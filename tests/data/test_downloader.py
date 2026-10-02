"""
tests/test_downloader.py
Unit tests cho module OHLCV Downloader (DATA-002).

- Ticket: DATA-002 (Sprint 1 - Member A)

FILE NÀY ĐỂ LÀM GÌ?
- Kiểm thử các hàm tải dữ liệu OHLCV từ Yahoo Finance.
- Sử dụng pytest và unittest.mock để mô phỏng các tình huống tải dữ liệu
- Kiểm tra chuẩn hóa dữ liệu, lưu file Parquet/CSV, và xử lý lỗi khi tải dữ liệu.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.downloader import (
    STANDARD_COLUMNS,
    OHLCVDownloader,
    download_market_data,
    normalize_ohlcv_dataframe,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture tạo dữ liệu OHLCV thô mô phỏng đầu ra của yfinance."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Open": [150.0, 152.0, 151.0, 153.0, 155.0],
            "High": [153.0, 154.0, 153.5, 156.0, 157.0],
            "Low": [149.0, 150.5, 150.0, 152.0, 154.0],
            "Close": [152.0, 151.0, 153.0, 155.0, 156.5],
            "Adj Close": [151.5, 150.5, 152.5, 154.5, 156.0],
            "Volume": [1000000, 1200000, 900000, 1100000, 1300000],
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def sample_multiindex_df() -> pd.DataFrame:
    """Fixture mô phỏng DataFrame với MultiIndex columns từ yfinance."""
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    cols = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Adj Close", "AAPL"),
            ("Volume", "AAPL"),
        ]
    )
    df = pd.DataFrame(
        [
            [150.0, 153.0, 149.0, 152.0, 151.5, 1000000],
            [152.0, 154.0, 150.5, 151.0, 150.5, 1200000],
            [151.0, 153.5, 150.0, 153.0, 152.5, 900000],
        ],
        index=dates,
        columns=cols,
    )
    return df


class TestNormalizeOHLCV:
    def test_normalize_empty_dataframe(self):
        """Kiểm tra xử lý khi dataframe rỗng."""
        empty_df = pd.DataFrame()
        result = normalize_ohlcv_dataframe(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == STANDARD_COLUMNS
        assert len(result) == 0

    def test_normalize_standard_dataframe(self, sample_raw_df):
        """Kiểm tra chuẩn hóa dataframe thông thường."""
        normalized = normalize_ohlcv_dataframe(sample_raw_df, ticker="AAPL")

        # Kiểm tra đầy đủ các cột chuẩn
        assert list(normalized.columns) == STANDARD_COLUMNS
        assert len(normalized) == 5

        # Kiểm tra kiểu dữ liệu date
        assert pd.api.types.is_datetime64_any_dtype(normalized["date"])
        # Kiểm tra giá trị không bị biến dạng
        assert normalized["open"].iloc[0] == 150.0
        assert normalized["close"].iloc[0] == 152.0
        assert normalized["adj_close"].iloc[0] == 151.5

    def test_normalize_multiindex_columns(self, sample_multiindex_df):
        """Kiểm tra xử lý MultiIndex columns từ yfinance."""
        normalized = normalize_ohlcv_dataframe(sample_multiindex_df, ticker="AAPL")
        assert list(normalized.columns) == STANDARD_COLUMNS
        assert len(normalized) == 3
        assert normalized["close"].iloc[0] == 152.0

    def test_timezone_removal(self, sample_raw_df):
        """Kiểm tra việc loại bỏ múi giờ tz-aware thành tz-naive."""
        sample_raw_df.index = sample_raw_df.index.tz_localize("America/New_York")
        normalized = normalize_ohlcv_dataframe(sample_raw_df)
        assert normalized["date"].dt.tz is None

    def test_deduplication_and_sorting(self):
        """Kiểm tra việc loại bỏ ngày trùng lặp và sắp xếp ngày tăng dần."""
        dates = pd.to_datetime(["2023-01-05", "2023-01-02", "2023-01-02"])
        df = pd.DataFrame(
            {
                "Open": [10.0, 20.0, 21.0],
                "High": [11.0, 21.0, 22.0],
                "Low": [9.0, 19.0, 20.0],
                "Close": [10.5, 20.5, 21.5],
                "Volume": [100, 200, 300],
            },
            index=dates,
        )
        normalized = normalize_ohlcv_dataframe(df)
        assert len(normalized) == 2
        # Kiểm tra thứ tự tăng dần
        assert normalized["date"].iloc[0] == pd.Timestamp("2023-01-02")
        assert normalized["date"].iloc[1] == pd.Timestamp("2023-01-05")
        # Giữ dòng cuối khi trùng lặp
        assert normalized["close"].iloc[0] == 21.5


class TestOHLCVDownloader:
    def test_invalid_format_raises_error(self, tmp_path):
        """Kiểm tra ném lỗi khi định dạng không hợp lệ."""
        with pytest.raises(ValueError, match="không hợp lệ"):
            OHLCVDownloader(output_dir=tmp_path, save_format="invalid_fmt")

    def test_save_ticker_data_parquet(self, tmp_path, sample_raw_df):
        """Kiểm tra lưu file Parquet."""
        downloader = OHLCVDownloader(output_dir=tmp_path, save_format="parquet")
        normalized = normalize_ohlcv_dataframe(sample_raw_df)
        paths = downloader.save_ticker_data(normalized, "AAPL")

        assert len(paths) == 1
        parquet_file = paths[0]
        assert parquet_file.suffix == ".parquet"
        assert parquet_file.exists()

        # Đọc lại và kiểm tra nội dung
        loaded = pd.read_parquet(parquet_file)
        assert len(loaded) == 5
        assert list(loaded.columns) == STANDARD_COLUMNS

    def test_save_ticker_data_csv(self, tmp_path, sample_raw_df):
        """Kiểm tra lưu file CSV."""
        downloader = OHLCVDownloader(output_dir=tmp_path, save_format="csv")
        normalized = normalize_ohlcv_dataframe(sample_raw_df)
        paths = downloader.save_ticker_data(normalized, "MSFT")

        assert len(paths) == 1
        csv_file = paths[0]
        assert csv_file.suffix == ".csv"
        assert csv_file.exists()

        loaded = pd.read_csv(csv_file)
        assert len(loaded) == 5
        assert "close" in loaded.columns

    def test_save_ticker_data_both(self, tmp_path, sample_raw_df):
        """Kiểm tra lưu cả 2 định dạng Parquet và CSV."""
        downloader = OHLCVDownloader(output_dir=tmp_path, save_format="both")
        normalized = normalize_ohlcv_dataframe(sample_raw_df)
        paths = downloader.save_ticker_data(normalized, "NVDA")

        assert len(paths) == 2
        extensions = {p.suffix for p in paths}
        assert extensions == {".parquet", ".csv"}

    @patch("yfinance.download")
    def test_download_ticker_success(self, mock_yf_download, tmp_path, sample_raw_df):
        """Kiểm tra tải dữ liệu thành công với mock."""
        mock_yf_download.return_value = sample_raw_df

        downloader = OHLCVDownloader(output_dir=tmp_path, save_format="parquet")
        df = downloader.download_ticker("AAPL", "2023-01-01", "2023-01-06")

        assert not df.empty
        assert len(df) == 5
        assert mock_yf_download.call_count == 1

    @patch("yfinance.download")
    def test_download_ticker_retry_on_failure(self, mock_yf_download, tmp_path, sample_raw_df):
        """Kiểm tra cơ chế retry khi gặp lỗi tạm thời."""
        # Lần 1 raise exception, lần 2 trả về dataframe thành công
        mock_yf_download.side_effect = [Exception("Network error"), sample_raw_df]

        downloader = OHLCVDownloader(output_dir=tmp_path, save_format="parquet")
        df = downloader.download_ticker("AAPL", "2023-01-01", "2023-01-06", max_retries=2, retry_delay=0.01)

        assert not df.empty
        assert len(df) == 5
        assert mock_yf_download.call_count == 2

    @patch("yfinance.download")
    def test_download_and_save_pipeline(self, mock_yf_download, tmp_path, sample_raw_df):
        """Kiểm tra pipeline download_and_save cho danh sách tickers."""
        mock_yf_download.return_value = sample_raw_df

        downloader = OHLCVDownloader(output_dir=tmp_path, save_format="parquet")
        results = downloader.download_and_save(["AAPL", "MSFT"], "2023-01-01", "2023-01-06", max_retries=1)

        assert results == {"AAPL": True, "MSFT": True}
        assert (tmp_path / "AAPL.parquet").exists()
        assert (tmp_path / "MSFT.parquet").exists()


class TestDownloadMarketDataAPI:
    @patch("yfinance.download")
    def test_high_level_api(self, mock_yf_download, tmp_path, sample_raw_df):
        """Kiểm tra hàm API cấp cao download_market_data."""
        mock_yf_download.return_value = sample_raw_df

        results = download_market_data(
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-06",
            output_dir=tmp_path,
            save_format="parquet",
            include_benchmark=False,
        )

        assert results == {"AAPL": True}
        assert (tmp_path / "AAPL.parquet").exists()

