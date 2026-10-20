"""
src/utils/config.py
Module quản lý cấu hình và tiện ích đường dẫn dự án.

FILE NÀY ĐỂ LÀM GÌ?
- Có hàm tiện ích thì ghi ra đây, các module khác có thể import và dùng chung.
- Tự động tìm đường dẫn tuyệt đối của dự án (tránh lỗi đường dẫn tương đối khi chạy script ở các thư mục khác nhau).
- Đọc các file cấu hình YAML, chuyển nó thành dạng Object 
  để các module khác dễ dàng truy xuất thông tin (ví dụ: lấy danh sách các mã cổ phiếu, phân ngành).
"""

from pathlib import Path
from typing import Any, Dict, List
import yaml


def get_project_root() -> Path:
    """Trả về đường dẫn tuyệt đối đến thư mục gốc của dự án."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_config_path(file_path: Path | str | None) -> Path:
    """Tìm và trả về đường dẫn file cấu hình, hỗ trợ fallback giữa configs/ và các thư mục con."""
    if file_path is None:
        raise ValueError("file_path không được là None.")
    path = Path(file_path)
    root = get_project_root()
    if path.is_absolute() and path.exists():
        return path
    rel_root = root / path
    if rel_root.exists():
        return rel_root

    # Kiểm tra trong các thư mục con: core, research, test
    filename = path.name
    for sub in ("core", "research", "test"):
        sub_candidate = root / "configs" / sub / filename
        if sub_candidate.exists():
            return sub_candidate

    return rel_root if not path.is_absolute() else path


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Đọc file cấu hình YAML và trả về dict."""
    path = resolve_config_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


class AssetsConfig:
    """Class tiện ích quản lý thông tin vũ trụ cổ phiếu từ assets.yaml."""

    def __init__(self, config_path: str = "configs/core/assets.yaml"):
        self.raw_config = load_yaml(config_path)
        self.assets = self.raw_config.get("assets", [])
        self.sectors = self.raw_config.get("sectors", {})
        self.benchmark = self.raw_config.get("benchmark", {})
        self.date_range = self.raw_config.get("date_range", {})
        self.cash = self.raw_config.get("cash", {})

    def get_tickers(self) -> List[str]:
        """Trả về danh sách toàn bộ các mã cổ phiếu trong danh mục."""
        return [asset["ticker"] for asset in self.assets]

    def get_ticker_to_sector(self) -> Dict[str, str]:
        """Trả về mapping từ ticker -> tên sector (ví dụ: {'AAPL': 'technology'})."""
        return {asset["ticker"]: asset["sector"] for asset in self.assets}

    def get_sector_to_tickers(self) -> Dict[str, List[str]]:
        """Trả về mapping từ sector -> danh sách tickers thuộc sector đó."""
        mapping: Dict[str, List[str]] = {s: [] for s in self.sectors}
        for asset in self.assets:
            sec = asset["sector"]
            if sec not in mapping:
                mapping[sec] = []
            mapping[sec].append(asset["ticker"])
        return mapping

    def get_benchmark_ticker(self) -> str:
        """Trả về mã benchmark (mặc định: SPY)."""
        return self.benchmark.get("ticker", "SPY")

    def get_date_range(self) -> tuple[str, str]:
        """Trả về tuple (start_date, end_date), mặc định ('2015-01-01', '2025-12-31')."""
        start = self.date_range.get("start_date", "2015-01-01")
        end = self.date_range.get("end_date", "2025-12-31")
        return start, end

    def get_split_dates(self) -> Dict[str, str]:
        """
        Trả về dictionary các mốc phân chia nếu được cấu hình trong assets.yaml:
        {'train_end': ..., 'val_start': ..., 'val_end': ..., 'test_start': ...}
        Tìm kiếm ưu tiên trong khối date_range, sau đó trong khối split riêng.
        """
        split_cfg = self.raw_config.get("split", {})
        train_end = self.date_range.get("train_end") or split_cfg.get("train_end")
        val_start = self.date_range.get("val_start") or split_cfg.get("val_start")
        val_end = self.date_range.get("val_end") or split_cfg.get("val_end")
        test_start = self.date_range.get("test_start") or split_cfg.get("test_start")

        result = {}
        if train_end:
            result["train_end"] = str(train_end)
        if val_start:
            result["val_start"] = str(val_start)
        if val_end:
            result["val_end"] = str(val_end)
        if test_start:
            result["test_start"] = str(test_start)
        return result

    def get_split_or_default_dates(self) -> tuple[str, str, str, str]:
        """
        Trả về tuple (train_end, val_start, val_end, test_start).
        Nếu file config đã chỉ định đủ 4 mốc thì trả về trực tiếp.
        Nếu thiếu, tự động tính toán dựa trên start_date và end_date theo tỷ lệ 70% / 15% / 15%.
        """
        splits = self.get_split_dates()
        if len(splits) == 4:
            return splits["train_end"], splits["val_start"], splits["val_end"], splits["test_start"]

        import pandas as pd
        start_dt = pd.Timestamp(self.date_range.get("start_date", "2015-01-01"))
        end_dt = pd.Timestamp(self.date_range.get("end_date", "2025-12-31"))
        total_days = max(1, (end_dt - start_dt).days)

        train_end_dt = start_dt + pd.Timedelta(days=int(total_days * 0.70))
        val_start_dt = train_end_dt + pd.Timedelta(days=1)
        val_end_dt = start_dt + pd.Timedelta(days=int(total_days * 0.85))
        test_start_dt = val_end_dt + pd.Timedelta(days=1)

        return (
            splits.get("train_end", str(train_end_dt.date())),
            splits.get("val_start", str(val_start_dt.date())),
            splits.get("val_end", str(val_end_dt.date())),
            splits.get("test_start", str(test_start_dt.date())),
        )


