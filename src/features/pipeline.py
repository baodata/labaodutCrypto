"""
src/features/pipeline.py
Động cơ đường ống hợp nhất đặc trưng thị trường (Unified Feature Pipeline).

Ticket: FEAT-006 (P0 - Thành viên A)
Sprint: 3

Mục đích:
1. Kết nối và đồng bộ hóa toàn bộ các động cơ đặc trưng:
   - ReturnsEngine: Tính toán lợi suất hàng ngày (FEAT-001)
   - VolatilityEngine: Độ biến động trượt 20 phiên (FEAT-002)
   - RSIEngine: Chỉ báo Wilder RSI-14 (FEAT-003)
   - MACDEngine: MACD Line và Signal Line (FEAT-004)
   - VolumeEngine: Khối lượng Log-Zscore chuẩn hóa (FEAT-005)
2. Đúc thành đối tượng MarketDataTensor kích thước [T, N, F] theo đúng Data Contract
   cam kết với Thành viên B (src/utils/data_types.py):
   - Trục 0 (T): Số ngày giao dịch (2,765 ngày)
   - Trục 1 (N): 24 cổ phiếu có thể giao dịch (SPY được giữ riêng làm benchmark)
   - Trục 2 (F): feature kỹ thuật, động lượng, rủi ro và tương đối cắt ngang
3. Xuất dữ liệu đã tinh chế ra tệp parquet tại data/processed/features.parquet.
4. Tự động kiểm tra tính toàn vẹn (không có NaN, không có Inf) trước khi xuất xưởng.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.alignment import CalendarAligner
from src.data.validator import DataValidator
from src.features.macd import MACDEngine
from src.features.momentum import MOMENTUM_FEATURE_NAMES, calculate_momentum_features
from src.features.relative import RELATIVE_FEATURE_NAMES, calculate_relative_features
from src.features.risk import RISK_FEATURE_NAMES, calculate_risk_features
from src.features.returns import ReturnsEngine
from src.features.rsi import RSIEngine
from src.features.volatility import VolatilityEngine
from src.features.volume import VolumeEngine
from src.utils.config import AssetsConfig
from src.utils.data_types import MarketDataTensor


class FeaturePipeline:
    """Đường ống tính toán và đóng gói đặc trưng đa tài sản."""
    WARMUP_SESSIONS = 252

    FEATURE_NAMES: List[str] = [
        "return",
        "volatility_20d",
        "rsi_14",
        "macd",
        "macd_signal",
        "volume_norm",
        *RELATIVE_FEATURE_NAMES,
        *MOMENTUM_FEATURE_NAMES,
        *RISK_FEATURE_NAMES,
    ]

    def __init__(
        self,
        return_type: str = "log",
        vol_window: int = 20,
        rsi_period: int = 14,
        rsi_scaled: bool = True,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        volume_window: int = 20,
        price_col: str = "close",
        volume_col: str = "volume",
        assets_config_path: str = "configs/core/assets.yaml",
    ) -> None:
        """Khởi tạo FeaturePipeline với các tham số cấu hình."""
        self.assets_config_path = assets_config_path
        self.parameters = {
            "return_type": return_type, "vol_window": vol_window,
            "rsi_period": rsi_period, "rsi_scaled": rsi_scaled,
            "macd_fast": macd_fast, "macd_slow": macd_slow,
            "macd_signal": macd_signal, "volume_window": volume_window,
            "price_col": price_col, "volume_col": volume_col,
            "warmup_sessions": self.WARMUP_SESSIONS,
            "downside_volatility_definition": "sqrt_mean_squared_negative_returns_zero_threshold",
        }
        self.returns_engine = ReturnsEngine(return_type=return_type, price_col=price_col, fill_zero=True)
        self.vol_engine = VolatilityEngine(window=vol_window, annualized=False, return_col="return", fill_zero=True)
        self.rsi_engine = RSIEngine(period=rsi_period, scaled=rsi_scaled, fill_na=True, price_col=price_col)
        self.macd_engine = MACDEngine(
            fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal, fill_zero=True, price_col=price_col
        )
        self.volume_engine = VolumeEngine(window=volume_window, fill_zero=True, volume_col=volume_col)

    def process_asset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính feature đơn mã; feature cắt ngang được hoàn thiện trong process_all."""
        DataValidator().validate_dataframe(df).raise_if_invalid()
        df_out = df.copy()

        # 1. Return
        df_out["return"] = self.returns_engine.calculate(df_out)

        # 2. Volatility (dựa trên cột return vừa tính)
        df_out["volatility_20d"] = self.vol_engine.calculate(df_out)

        # 3. RSI
        df_out["rsi_14"] = self.rsi_engine.calculate(df_out)

        # 4 & 5. MACD & Signal
        macd, signal, _ = self.macd_engine.calculate(df_out)
        df_out["macd"] = macd
        df_out["macd_signal"] = signal

        # 6. Volume Norm
        df_out["volume_norm"] = self.volume_engine.calculate(df_out)

        close = pd.to_numeric(df_out[self.returns_engine.price_col], errors="coerce")
        high_col = next((c for c in df_out.columns if str(c).lower() == "high"), None)
        low_col = next((c for c in df_out.columns if str(c).lower() == "low"), None)
        if high_col is None or low_col is None:
            raise ValueError("OHLC cần có cột High và Low để tính Parkinson volatility và ATR.")
        high = pd.to_numeric(df_out[high_col], errors="coerce")
        low = pd.to_numeric(df_out[low_col], errors="coerce")
        single_asset_features = pd.concat(
            [
                calculate_momentum_features(close),
                calculate_risk_features(close, high, low),
            ],
            axis=1,
        )
        df_out[single_asset_features.columns] = single_asset_features

        # Neutral standalone defaults; process_all replaces them using the full universe.
        for feature in RELATIVE_FEATURE_NAMES:
            df_out[feature] = 0.5 if feature.startswith("xs_rank_") else 0.0
        for feature in self.FEATURE_NAMES:
            values = pd.to_numeric(df_out[feature], errors="coerce")
            df_out[feature] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df_out

    def process_all(
        self,
        aligned_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Xử lý toàn bộ các mã cổ phiếu trong từ điển dữ liệu đã căn chỉnh lịch.

        Args:
            aligned_data: Từ điển {ticker: DataFrame OHLCV}.

        Returns:
            Từ điển {ticker: DataFrame đã có đủ feature; SPY chỉ là chuỗi tham chiếu}.
        """
        processed_data: dict[str, pd.DataFrame] = {}
        expected_index = None
        for ticker, df in aligned_data.items():
            if expected_index is None:
                expected_index = df.index
            elif not df.index.equals(expected_index):
                raise ValueError(f"Lịch giao dịch của {ticker} không khớp với các mã còn lại.")
            processed_data[ticker] = self.process_asset(df)

        relative_features = calculate_relative_features(
            processed_data,
            AssetsConfig(self.assets_config_path).get_benchmark_ticker(),
        )
        for ticker, features in relative_features.items():
            processed_data[ticker][list(RELATIVE_FEATURE_NAMES)] = features[
                list(RELATIVE_FEATURE_NAMES)
            ]
        return processed_data

    def build_market_data_tensor(
        self,
        processed_data: dict[str, pd.DataFrame],
        tickers_order: Optional[List[str]] = None,
    ) -> MarketDataTensor:
        """Đúc dữ liệu đã tinh chế thành MarketDataTensor [T, N, F].

        Args:
            processed_data: Từ điển {ticker: DataFrame có đủ FEATURE_NAMES}.
            tickers_order: Thứ tự cụ thể của N mã cổ phiếu (mặc định lấy sorted keys).

        Returns:
            MarketDataTensor thỏa mãn 100% Data Contract.
        """
        if not processed_data:
            raise ValueError("processed_data không được rỗng.")

        benchmark_ticker = AssetsConfig(self.assets_config_path).get_benchmark_ticker().upper()
        normalized_data = {str(ticker).upper(): frame for ticker, frame in processed_data.items()}
        if len(normalized_data) != len(processed_data):
            raise ValueError("processed_data chứa ticker trùng lặp sau chuẩn hóa chữ hoa.")
        tickers = (
            [str(ticker).upper() for ticker in tickers_order]
            if tickers_order is not None
            else sorted(
                ticker for ticker in normalized_data
                if ticker != benchmark_ticker
            )
        )
        if not tickers:
            raise ValueError("Không có tài sản giao dịch để đưa vào MarketDataTensor.")
        if len(set(tickers)) != len(tickers):
            raise ValueError("tickers_order chứa mã trùng lặp.")
        if benchmark_ticker in tickers:
            raise ValueError(
                f"'{benchmark_ticker}' là benchmark và không được nằm trong MarketDataTensor/action space."
            )
        missing_tickers = [ticker for ticker in tickers if ticker not in normalized_data]
        if missing_tickers:
            raise KeyError(
                "Thiếu dữ liệu feature cho ticker: " + ", ".join(missing_tickers)
            )
        first_df = normalized_data[tickers[0]]
        if not isinstance(first_df.index, pd.DatetimeIndex) or not first_df.index.is_monotonic_increasing or not first_df.index.is_unique:
            raise ValueError("Feature index phải là DatetimeIndex tăng dần, không trùng lặp.")
        dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in first_df.index]

        num_timesteps = len(dates)
        num_assets = len(tickers)
        num_features = len(self.FEATURE_NAMES)

        # Khởi tạo ma trận [T, N, F]
        tensor_data = np.zeros((num_timesteps, num_assets, num_features), dtype=np.float32)

        for n_idx, ticker in enumerate(tickers):
            df = normalized_data[ticker]
            if not df.index.equals(first_df.index):
                raise ValueError(f"Lịch giao dịch của {ticker} không khớp; không được gán feature theo vị trí.")
            if len(df) != num_timesteps:
                raise ValueError(
                    f"Số dòng của ticker '{ticker}' ({len(df)}) không khớp với T={num_timesteps}."
                )
            for f_idx, feat_name in enumerate(self.FEATURE_NAMES):
                if feat_name not in df.columns:
                    raise KeyError(f"Đặc trưng '{feat_name}' không có trong DataFrame của '{ticker}'.")
                tensor_data[:, n_idx, f_idx] = df[feat_name].to_numpy(dtype=np.float32)

        # Đóng gói và validate
        market_tensor = MarketDataTensor(
            tensor=tensor_data,
            tickers=tickers,
            feature_names=list(self.FEATURE_NAMES),
            dates=dates,
            warmup_sessions=self.WARMUP_SESSIONS,
        )
        market_tensor.validate()
        return market_tensor

    def to_flat_dataframe(
        self,
        processed_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Xuất feature của tài sản giao dịch; benchmark chỉ dùng tính feature tham chiếu."""
        records = []
        benchmark = AssetsConfig(self.assets_config_path).get_benchmark_ticker().upper()
        for ticker, df in processed_data.items():
            if str(ticker).upper() == benchmark:
                continue
            sub_df = df[self.FEATURE_NAMES].copy()
            sub_df["ticker"] = ticker
            sub_df["date"] = sub_df.index
            records.append(sub_df)

        combined = pd.concat(records, axis=0)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.set_index(["date", "ticker"]).sort_index()
        return combined

    def export_processed_data(
        self,
        processed_data: dict[str, pd.DataFrame],
        output_file: Union[Path, str] = "data/processed/features.parquet",
    ) -> Path:
        """Xuất dữ liệu đã tinh chế ra tệp parquet tại data/processed/features.parquet.

        Args:
            processed_data: Từ điển DataFrame đặc trưng.
            output_file: Đường dẫn tệp đầu ra.

        Returns:
            Path tới tệp đã lưu.
        """
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        flat_df = self.to_flat_dataframe(processed_data)
        flat_df.to_parquet(path)
        return path

    def run_from_raw(
        self,
        raw_dir: Union[Path, str] = "data/raw",
        export_parquet: bool = True,
        output_file: Union[Path, str] = "data/processed/features.parquet",
        end_date: str | None = None,
    ) -> Tuple[dict[str, pd.DataFrame], MarketDataTensor]:
        """Căn lịch, tính feature, tạo tensor; SPY là tham chiếu ngoài action space."""
        aligner = CalendarAligner(method="intersection")
        assets_config = AssetsConfig(self.assets_config_path)
        tradable_tickers = assets_config.get_tickers()
        benchmark_ticker = assets_config.get_benchmark_ticker()
        requested_tickers = list(tradable_tickers)
        if benchmark_ticker.upper() not in {ticker.upper() for ticker in requested_tickers}:
            requested_tickers.append(benchmark_ticker)
        alignment_res = aligner.align_from_directory(
            raw_dir,
            tickers=requested_tickers,
            end_date=end_date,
        )

        processed_data = self.process_all(alignment_res.aligned_data)
        market_tensor = self.build_market_data_tensor(processed_data)

        if export_parquet:
            self.export_processed_data(processed_data, output_file=output_file)

        return processed_data, market_tensor

