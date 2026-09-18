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
   - Trục 1 (N): 25 tài sản (24 cổ phiếu + SPY benchmark)
   - Trục 2 (F): 6 đặc trưng kỹ thuật chuẩn hóa
3. Xuất dữ liệu đã tinh chế ra tệp parquet tại data/processed/features.parquet.
4. Tự động kiểm tra tính toàn vẹn (không có NaN, không có Inf) trước khi xuất xưởng.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.alignment import CalendarAligner
from src.features.macd import MACDEngine
from src.features.returns import ReturnsEngine
from src.features.rsi import RSIEngine
from src.features.volatility import VolatilityEngine
from src.features.volume import VolumeEngine
from src.utils.data_types import MarketDataTensor


class FeaturePipeline:
    """Đường ống tính toán và đóng gói đặc trưng đa tài sản."""

    FEATURE_NAMES: List[str] = [
        "return",
        "volatility_20d",
        "rsi_14",
        "macd",
        "macd_signal",
        "volume_norm",
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
    ) -> None:
        """Khởi tạo FeaturePipeline với các tham số cấu hình."""
        self.returns_engine = ReturnsEngine(return_type=return_type, price_col=price_col, fill_zero=True)
        self.vol_engine = VolatilityEngine(window=vol_window, annualized=False, return_col="return", fill_zero=True)
        self.rsi_engine = RSIEngine(period=rsi_period, scaled=rsi_scaled, fill_na=True, price_col=price_col)
        self.macd_engine = MACDEngine(
            fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal, fill_zero=True, price_col=price_col
        )
        self.volume_engine = VolumeEngine(window=volume_window, fill_zero=True, volume_col=volume_col)

    def process_asset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán toàn bộ 6 đặc trưng cho một DataFrame cổ phiếu."""
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

        return df_out

    def process_all(
        self,
        aligned_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Xử lý toàn bộ các mã cổ phiếu trong từ điển dữ liệu đã căn chỉnh lịch.

        Args:
            aligned_data: Từ điển {ticker: DataFrame OHLCV}.

        Returns:
            Từ điển {ticker: DataFrame đã có đủ 6 cột đặc trưng}.
        """
        processed_data: dict[str, pd.DataFrame] = {}
        for ticker, df in aligned_data.items():
            processed_data[ticker] = self.process_asset(df)
        return processed_data

    def build_market_data_tensor(
        self,
        processed_data: dict[str, pd.DataFrame],
        tickers_order: Optional[List[str]] = None,
    ) -> MarketDataTensor:
        """Đúc dữ liệu đã tinh chế thành MarketDataTensor [T, N, F].

        Args:
            processed_data: Từ điển {ticker: DataFrame có đủ 6 đặc trưng}.
            tickers_order: Thứ tự cụ thể của N mã cổ phiếu (mặc định lấy sorted keys).

        Returns:
            MarketDataTensor thỏa mãn 100% Data Contract.
        """
        if not processed_data:
            raise ValueError("processed_data không được rỗng.")

        tickers = tickers_order or sorted(processed_data.keys())
        first_df = processed_data[tickers[0]]
        dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in first_df.index]

        num_timesteps = len(dates)
        num_assets = len(tickers)
        num_features = len(self.FEATURE_NAMES)

        # Khởi tạo ma trận [T, N, F]
        tensor_data = np.zeros((num_timesteps, num_assets, num_features), dtype=np.float32)

        for n_idx, ticker in enumerate(tickers):
            df = processed_data[ticker]
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
        )
        market_tensor.validate()
        return market_tensor

    def to_flat_dataframe(
        self,
        processed_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Chuyển đổi dữ liệu tinh chế thành bảng phẳng Multi-index (Date, Ticker)."""
        records = []
        for ticker, df in processed_data.items():
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
    ) -> Tuple[dict[str, pd.DataFrame], MarketDataTensor]:
        """Thực thi toàn bộ quy trình: Căn chỉnh lịch -> Tính 6 đặc trưng -> Đúc Tensor -> Xuất tệp."""
        aligner = CalendarAligner(method="intersection")
        alignment_res = aligner.align_from_directory(raw_dir)

        processed_data = self.process_all(alignment_res.aligned_data)
        market_tensor = self.build_market_data_tensor(processed_data)

        if export_parquet:
            self.export_processed_data(processed_data, output_file=output_file)

        return processed_data, market_tensor

