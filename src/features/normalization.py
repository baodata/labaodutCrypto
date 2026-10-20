"""QNT-004/005: per-feature XS and strictly lagged rolling normalization.

Streaming state stores raw previous sessions, not normalized values. Batch
transformation creates a fresh stream; it cannot contaminate later inference.
Already bounded features retain their level unless explicitly configured.
"""
from collections import deque
from copy import deepcopy
from statistics import NormalDist
import numpy as np
import pandas as pd
from src.features.scaler import MarketFeatureScaler
from src.utils.data_types import MarketDataTensor


class CausalFeatureScaler(MarketFeatureScaler):
    MODES = {"static_train", "identity", "xs_zscore", "xs_rank", "gaussian_rank", "rolling_zscore"}

    def __init__(self, modes=None, window=252, min_periods=None, eps=1e-8,
                 clip_range=(-10., 10.), default_mode="static_train", preserve_market_context=True):
        super().__init__(eps=eps, clip_range=clip_range)
        self.modes = dict(modes or {})
        self.window = int(window)
        self.min_periods = self.window if min_periods is None else int(min_periods)
        self.default_mode = default_mode
        self.preserve_market_context = bool(preserve_market_context)
        if self.window < 2 or not 2 <= self.min_periods <= self.window or eps <= 0 or not np.isfinite(eps):
            raise ValueError("Invalid normalization window/epsilon.")
        if clip_range is not None and (len(clip_range) != 2 or not np.isfinite(clip_range).all() or clip_range[0] >= clip_range[1]):
            raise ValueError("Invalid normalization clip range.")
        if any(m not in self.MODES for m in [default_mode, *self.modes.values()]):
            raise ValueError("Unknown normalization mode.")
        self.tickers = []
        self.session_dates = []
        self._session_positions = {}
        self.reset_stream()

    def reset_stream(self):
        self.history = deque(maxlen=self.window)
        self.normalized_history = deque(maxlen=max(self.window, 512))
        self.last_date = None
        self.last_output = None
        self.last_input = None
        self.ready = False
        self.clipped = 0
        self.values_seen = 0

    def fit(self, train_data, feature_names=None):
        if not isinstance(train_data, MarketDataTensor):
            raise TypeError("Causal scaler requires named MarketDataTensor.")
        if set(self.modes) - set(train_data.feature_names):
            raise ValueError("Normalization config contains unknown features.")
        super().fit(train_data, feature_names)
        self.tickers = list(train_data.tickers)
        return self

    def mode(self, feature):
        if feature in self.modes:
            return self.modes[feature]
        if feature.startswith('xs_rank_') or feature.startswith('rsi_'):
            return "identity"
        return self.default_mode

    @property
    def context_features(self):
        return [f for f in self.feature_names if self.mode(f) in {'xs_zscore','xs_rank','gaussian_rank'}] if self.preserve_market_context else []

    def market_context(self, values, valid_assets=None):
        x = np.asarray(values, dtype=float)
        valid = np.ones(len(x), dtype=bool) if valid_assets is None else np.asarray(valid_assets,dtype=bool)
        output = []
        for feature in self.context_features:
            values = x[:, self.feature_names.index(feature)]
            values = values[valid & np.isfinite(values)]
            mean, std = (float(values.mean()),float(values.std())) if len(values) else (self.means[feature],0.)
            output.extend([(mean-self.means[feature])/(self.stds[feature]+self.eps),std/(self.stds[feature]+self.eps)])
        return np.clip(np.asarray(output,dtype=np.float32), *(self.clip_range or (-np.inf,np.inf)))

    def set_session_calendar(self, dates):
        index = pd.DatetimeIndex(dates)
        if not index.is_unique or not index.is_monotonic_increasing or index.hasnans:
            raise ValueError('Expected ordered unique trading sessions.')
        if self.last_date is not None and pd.Timestamp(self.last_date) not in index:
            raise ValueError('Session calendar must include the stored last date.')
        self.session_dates = [str(d) for d in index]
        self._session_positions = {date:i for i,date in enumerate(self.session_dates)}

    def transform_day(self, values, date, *, tickers=None, valid_assets=None, update=True):
        if not self.is_fitted:
            raise RuntimeError("Fit on training data first.")
        if tickers is not None and list(tickers) != self.tickers:
            raise ValueError("Normalization ticker order mismatch.")
        x = np.asarray(values, dtype=float)
        if x.shape != (len(self.tickers), len(self.feature_names)):
            raise ValueError("Expected named [N,F] daily features.")
        date = pd.Timestamp(date)
        if self.last_date is not None and date <= pd.Timestamp(self.last_date):
            if date == pd.Timestamp(self.last_date) and np.array_equal(x, self.last_input, equal_nan=True):
                return self.last_output.copy()
            raise ValueError("Streaming dates must increase; replay requires a fresh state.")
        if self.session_dates:
            position = self._session_positions.get(str(date))
            if position is None:
                raise ValueError('Date not in declared session calendar; extend calendar before inference.')
            expected = 0 if self.last_date is None else self._session_positions[self.last_date]+1
            if position != expected:
                raise ValueError('Missing normalization sessions; stream every intervening raw daily feature first.')
        valid = np.ones(len(x), dtype=bool) if valid_assets is None else np.asarray(valid_assets, dtype=bool)
        if valid.shape != (len(x),):
            raise ValueError("Universe mask shape must be [N].")
        previous = np.asarray(self.history) if self.history else None
        output = np.zeros_like(x)
        for f, feature in enumerate(self.feature_names):
            v = x[:, f]
            eligible = valid & np.isfinite(v)
            mode = self.mode(feature)
            if mode == 'identity':
                output[eligible, f] = v[eligible]
            elif mode == 'static_train':
                output[eligible, f] = (v[eligible] - self.means[feature]) / (self.stds[feature] + self.eps)
            elif mode == 'xs_zscore':
                if eligible.any():
                    output[eligible, f] = (v[eligible] - v[eligible].mean()) / (v[eligible].std() + self.eps)
            elif mode in {'xs_rank', 'gaussian_rank'}:
                n = int(eligible.sum())
                if n:
                    ranks = pd.Series(v[eligible]).rank(method='average').to_numpy()
                    if mode == 'xs_rank':
                        output[eligible, f] = (ranks-1)/(n-1) if n > 1 else .5
                    else:
                        output[eligible, f] = [NormalDist().inv_cdf(float(p)) for p in (ranks-.5)/n]
            elif previous is not None:
                h = previous[:, :, f]
                for asset in np.flatnonzero(eligible):
                    past = h[:, asset]
                    past = past[np.isfinite(past)]
                    if len(past) >= self.min_periods:
                        output[asset, f] = (v[asset] - past.mean()) / (past.std() + self.eps)
        clipped_count = 0
        if self.clip_range:
            clipped_count = int(np.count_nonzero((output < self.clip_range[0]) | (output > self.clip_range[1])))
            output = np.clip(output, *self.clip_range)
        if update:
            self.clipped += clipped_count
            self.values_seen += int(np.count_nonzero(np.isfinite(x) & valid[:, None]))
            self.history.append(np.where(valid[:, None], x, np.nan))
            self.last_date = str(date)
            self.last_input = x.copy()
            self.last_output = output.astype(np.float32)
            rolling = [i for i,f in enumerate(self.feature_names) if self.mode(f) == 'rolling_zscore']
            counts = np.isfinite(np.asarray(self.history)).sum(axis=0)
            self.ready = bool(valid.any()) and (not rolling or bool(np.all(counts[valid][:,rolling] >= self.min_periods)))
            self.normalized_history.append((str(date), output.astype(np.float32).tolist()))
        return output.astype(np.float32)

    def transform(self, data):
        if not isinstance(data, MarketDataTensor):
            raise TypeError("Causal scaler requires dates/ticker order; use transform_day for streaming.")
        if data.feature_names != self.feature_names or data.tickers != self.tickers:
            raise ValueError("Normalization feature/ticker order mismatch.")
        stream = deepcopy(self)
        stream.reset_stream()
        stream.set_session_calendar(data.dates)
        output = []
        diagnostics = []
        for i, date in enumerate(data.dates):
            # Feature warm-up is missing history, not a zero return sample.
            before = stream.clipped
            output.append(stream.transform_day(data.tensor[i], date, valid_assets=np.full(len(self.tickers), bool(data.feature_ready[i]))))
            diagnostics.append(dict(date=str(date),state_ready_for_next_day=stream.ready,
                clipped_values=stream.clipped-before,eligible_values=int(data.feature_ready[i])*len(self.tickers)*len(self.feature_names)))
        rolling = any(self.mode(f) == 'rolling_zscore' for f in self.feature_names)
        warmup = data.warmup_sessions + (self.min_periods if rolling else 0)
        result = MarketDataTensor(np.asarray(output), list(data.tickers), list(data.feature_names), list(data.dates), warmup)
        result.validate()
        self.last_batch_state = stream.to_dict()
        self.last_batch_diagnostics = pd.DataFrame(diagnostics)
        return result

    def to_dict(self):
        payload = super().to_dict()
        payload.update(version='2.0', kind='causal_feature_scaler', modes=self.modes,
            window=self.window, min_periods=self.min_periods, default_mode=self.default_mode,
            preserve_market_context=self.preserve_market_context,
            tickers=self.tickers, session_dates=self.session_dates, state={
                'history': [np.where(np.isfinite(x), x, 0).tolist() for x in self.history],
                'valid_history': [np.isfinite(x).tolist() for x in self.history],
                'last_date': self.last_date,
                'last_output': self.last_output.tolist() if self.last_output is not None else None,
                'last_input': np.nan_to_num(self.last_input).tolist() if self.last_input is not None else None,
                'last_input_valid': np.isfinite(self.last_input).tolist() if self.last_input is not None else None,
                'normalized_history': list(self.normalized_history),
                'ready': self.ready, 'clipped': self.clipped, 'values_seen': self.values_seen})
        return payload

    @classmethod
    def from_dict(cls, payload):
        validation = dict(payload)
        validation.pop('kind',None)
        MarketFeatureScaler.from_dict(validation)
        obj = cls(payload['modes'], payload['window'], payload['min_periods'], payload['eps'],
                  tuple(payload['clip_range']) if payload['clip_range'] else None, payload['default_mode'],
                  payload.get('preserve_market_context',True))
        obj.is_fitted = payload['is_fitted']
        obj.feature_names, obj.tickers = list(payload['feature_names']), list(payload['tickers'])
        obj.means, obj.stds = dict(payload['means']), dict(payload['stds'])
        state = payload['state']
        for values, valid in zip(state['history'], state['valid_history']):
            obj.history.append(np.where(valid, values, np.nan))
        obj.normalized_history.extend(state.get('normalized_history', []))
        obj.last_date, obj.ready = state['last_date'], state['ready']
        obj.last_output = np.asarray(state['last_output'], dtype=np.float32) if state['last_output'] is not None else None
        obj.last_input = np.where(state['last_input_valid'], state['last_input'], np.nan) if state['last_input'] is not None else None
        obj.clipped, obj.values_seen = state['clipped'], state['values_seen']
        if payload.get('session_dates'):
            obj.set_session_calendar(payload['session_dates'])
        return obj
