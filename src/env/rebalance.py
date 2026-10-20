"""
src/env/rebalance.py
Ticket: ENV-003 (Sprint 4 - Thành viên A)

Bộ máy xử lý ràng buộc tỷ trọng (Rebalance Engine).
Đảm bảo các hành động từ Agent không vi phạm nguyên tắc toán học của danh mục đầu tư.
"""
import numpy as np

class RebalanceEngine:
    """Bộ xử lý ràng buộc và tái cân bằng danh mục."""
    
    @staticmethod
    def project_weights(
        weights: np.ndarray,
        max_asset_weight: float | None = None,
    ) -> np.ndarray:
        """
        Chiếu (Project) mảng tỷ trọng về không gian hợp lệ:
        1. Long-only: Mọi tỷ trọng đều phải >= 0.
        2. Mỗi tài sản rủi ro không vượt max_asset_weight (nếu được cấu hình).
        3. Fully-invested (kể cả cash): Tổng tỷ trọng = 1.0.
        
        Nếu mô hình xuất ra toàn số âm (hoặc 0), 
        hệ thống tự động chuyển 100% tỷ trọng vào Tiền mặt (Cash - node cuối).
        
        Args:
            weights: Mảng tỷ trọng thô do AI sinh ra (kích thước N+1).
            max_asset_weight: Trần tỷ trọng cho mỗi cổ phiếu; tiền mặt không bị giới hạn.
            
        Returns:
            Mảng tỷ trọng hợp lệ, tổng đúng bằng 1.0.
        """
        raw_weights = np.asarray(weights, dtype=np.float64)
        if raw_weights.ndim != 1 or raw_weights.size < 2:
            raise ValueError("weights phải là vector chứa ít nhất một tài sản và tiền mặt.")
        if not np.all(np.isfinite(raw_weights)):
            raise ValueError("weights phải chứa các giá trị hữu hạn.")
        if max_asset_weight is not None and (
            not np.isfinite(max_asset_weight) or not 0.0 < max_asset_weight <= 1.0
        ):
            raise ValueError("max_asset_weight phải hữu hạn và thuộc (0, 1].")

        # Normalize after scaling by the maximum to avoid overflow for large finite inputs.
        nonnegative = np.maximum(raw_weights, 0.0)
        scale = float(np.max(nonnegative))
        if scale > 0.0:
            scaled = nonnegative / scale
            projected = scaled / np.sum(scaled)
        else:
            projected = np.zeros_like(nonnegative)
            projected[-1] = 1.0

        if max_asset_weight is not None:
            assets = np.minimum(projected[:-1], float(max_asset_weight))
            excess = float(np.sum(projected[:-1] - assets))
            projected = np.concatenate((assets, [projected[-1] + excess]))

        # Accounting stays float64; the policy observation adapter casts to float32.
        return projected


class ExecutionController:
    """QNT-013: choose speeds and notional caps at Close, fill at next Open.

    Limits refer to decision NAV and trailing ADV, fixed before the gap.
    At Open only the mechanical quantity cap is applied; no speed is refit.
    A no-trade stock leg holds its quantity through the overnight gap.
    """
    CONFIG_KEYS = {'enabled', 'no_trade_band', 'participation_cap', 'cost_aversion',
                   'risk_aversion', 'speed_mode', 'alpha_min', 'alpha_max',
                   'alpha_buy', 'alpha_sell', 'er_window', 'cost_risk_brake'}

    def __init__(self, config):
        if (not isinstance(config, dict) or set(config)-self.CONFIG_KEYS
                or type(config.get('enabled', False)) is not bool):
            raise ValueError('Unknown execution controller configuration keys.')
        self.config = dict(config)
        self.enabled = bool(config.get('enabled', False))
        self.band = float(config.get('no_trade_band', 0.))
        self.participation_cap = float(config.get('participation_cap', .01))
        self.cost_aversion = float(config.get('cost_aversion', 1.))
        self.risk_aversion = float(config.get('risk_aversion', 1.))
        self.speed_mode = config.get('speed_mode', 'cost_aware')
        self.alpha_min = float(config.get('alpha_min', .05))
        self.alpha_max = float(config.get('alpha_max', .8))
        self.alpha_buy = float(config.get('alpha_buy', .25))
        self.alpha_sell = float(config.get('alpha_sell', .75))
        self.er_window = config.get('er_window', 10)
        self.cost_risk_brake = config.get('cost_risk_brake', True)
        if not all(np.isfinite(v) and v >= 0 for v in [self.band, self.cost_aversion, self.risk_aversion]) or not 0 < self.participation_cap <= 1:
            raise ValueError("Invalid execution controller configuration.")
        if (self.speed_mode not in {'cost_aware', 'fixed', 'efficiency_ratio', 'asymmetric'}
                or not np.isfinite([self.alpha_min, self.alpha_max, self.alpha_buy, self.alpha_sell]).all()
                or not 0 < self.alpha_min <= self.alpha_max <= 1
                or not 0 < self.alpha_buy <= 1 or not 0 < self.alpha_sell <= 1
                or type(self.er_window) is not int or self.er_window < 1
                or type(self.cost_risk_brake) is not bool):
            raise ValueError("Invalid execution speed mode/alpha/ER configuration.")

    @property
    def replaces_action_alpha(self):
        """Old cost_aware behavior remains blend-then-brake for compatibility."""
        return self.enabled and self.speed_mode != 'cost_aware'

    def decide(self, current, target, nav, adv, sigma, covariance, ready, base_cost,
               *, base_alpha=1., efficiency_ratio=None, trend_ready=None,
               return_diagnostics=False, alpha_override=None):
        delta = np.asarray(target[:-1]) - np.asarray(current[:-1])
        daily_risk = np.sqrt(np.maximum(np.diag(covariance), 0))
        participation = np.divide(np.abs(delta)*nav, adv, out=np.zeros_like(delta), where=adv > 0)
        brake = 1 / (1 + self.cost_aversion * (base_cost + sigma*np.sqrt(participation)) / np.maximum(daily_risk, 1e-6)
                     + self.risk_aversion * daily_risk)
        requested = np.ones_like(delta)
        ready = np.broadcast_to(np.asarray(ready, dtype=bool), delta.shape).copy()
        if self.speed_mode == 'fixed':
            if not np.isfinite(base_alpha) or not 0 < base_alpha <= 1:
                raise ValueError('base_alpha must be in (0,1].')
            requested.fill(base_alpha)
        elif self.speed_mode == 'asymmetric':
            # A target reduction is not necessarily a crash/risk-off signal.
            requested = np.where(delta < 0, self.alpha_sell, self.alpha_buy)
        elif self.speed_mode == 'efficiency_ratio':
            er = np.asarray(efficiency_ratio, dtype=float)
            er_ready = np.asarray(trend_ready, dtype=bool)
            if er.shape != delta.shape or er_ready.shape != delta.shape:
                raise ValueError('ER mode requires an aligned causal trend bank.')
            ready &= er_ready & np.isfinite(er)
            requested = self.alpha_min + (self.alpha_max-self.alpha_min)*np.clip(er, 0., 1.)**2
        if alpha_override is not None and self.replaces_action_alpha:
            if not np.isfinite(alpha_override) or not 0 < alpha_override <= 1:
                raise ValueError('alpha_override must be in (0,1].')
            requested.fill(alpha_override)
        speed = requested * (brake if self.speed_mode == 'cost_aware' or self.cost_risk_brake else 1.)
        speed = np.where(ready & np.isfinite(speed), speed, 0.)
        hold = (np.abs(delta) <= self.band) | ~ready
        proposed = current[:-1] + np.where(hold, 0., speed * delta)
        if self.replaces_action_alpha:
            # Per-leg alphas need cash funding. Never renormalize assets after
            # faster sells: the residual cash is intentional.
            change = proposed-current[:-1]
            buys = np.maximum(change, 0.)
            available = max(0., float(current[-1]-np.minimum(change, 0.).sum()))
            if buys.sum() > available:
                change = np.minimum(change, 0.) + buys*(available/buys.sum())
            proposed = current[:-1]+change
        result = (proposed, hold, self.participation_cap * np.maximum(adv, 0))
        if return_diagnostics:
            return (*result, dict(speed_mode=self.speed_mode, requested_alpha=requested.copy(),
                scheduled_alpha=np.where(hold, 0., speed),
                effective_alpha=np.divide(proposed-current[:-1], delta,
                    out=np.zeros_like(delta), where=np.abs(delta) > 1e-15), hold_mask=hold.copy(),
                close_proposed_weights=np.append(proposed, max(0., 1-proposed.sum()))))
        return result

    @staticmethod
    def fill(proposed, hold, notional_caps, open_weights, open_nav, max_weight):
        delta = proposed - open_weights[:-1]
        delta = np.clip(delta, -notional_caps/open_nav, notional_caps/open_nav)
        delta[hold] = 0.
        # A hold above the cap remains a hold; only new buys must respect cap.
        delta = np.minimum(delta, np.maximum(max_weight-open_weights[:-1], 0))
        delta = np.maximum(delta, -open_weights[:-1])
        sells = -np.minimum(delta, 0).sum()
        buys = np.maximum(delta, 0).sum()
        available = open_weights[-1] + sells
        if buys > available:
            delta = np.where(delta > 0, delta * available/buys, delta)
        assets = open_weights[:-1] + delta
        return np.append(assets, max(0., 1-assets.sum()))
