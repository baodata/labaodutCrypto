"""
src/utils/config.py
Module quản lý cấu hình và tiện ích đường dẫn dự án.

FILE NÀY ĐỂ LÀM GÌ?
- Cung cấp các hàm tiện ích (utilities) dùng chung cho toàn dự án.
- Tự động tìm đường dẫn tuyệt đối của dự án (tránh lỗi đường dẫn tương đối khi chạy script ở các thư mục khác nhau).
- Đọc các file cấu hình YAML (như `assets.yaml`, `env.yaml`) và chuyển nó thành dạng Object 
  để các module khác dễ dàng truy xuất thông tin (ví dụ: lấy danh sách các mã cổ phiếu, phân ngành).
"""

from pathlib import Path
from typing import Any, Dict, List
import yaml


def get_project_root() -> Path:
    """Trả về đường dẫn tuyệt đối đến thư mục gốc của dự án."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Đọc file cấu hình YAML và trả về dict."""
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


class AssetsConfig:
    """Class tiện ích quản lý thông tin vũ trụ cổ phiếu từ assets.yaml."""

    def __init__(self, config_path: str = "configs/assets.yaml"):
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

