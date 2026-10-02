import unittest
from pathlib import Path
from src.utils.config import AssetsConfig, get_project_root, load_yaml


class TestAssetsConfig(unittest.TestCase):
    """Kiểm thử tiêu chuẩn nghiệm thu cho Ticket DATA-001 (Chọn vũ trụ cổ phiếu)."""

    def setUp(self):
        self.config_path = get_project_root() / "configs/assets.yaml"
        self.assertTrue(self.config_path.exists(), "File configs/assets.yaml chưa được tạo!")
        self.assets_config = AssetsConfig("configs/assets.yaml")

    def test_yaml_syntax_and_structure(self):
        """Kiểm tra cú pháp YAML hợp lệ và có các khối chính."""
        data = load_yaml(self.config_path)
        self.assertIn("assets", data)
        self.assertIn("sectors", data)
        self.assertIn("benchmark", data)
        self.assertIn("cash", data)

    def test_minimum_asset_count(self):
        """Tiêu chuẩn nghiệm thu: Ít nhất 20 mã cổ phiếu (ở đây có 24 mã)."""
        tickers = self.assets_config.get_tickers()
        self.assertGreaterEqual(len(tickers), 20, f"Cần ít nhất 20 mã, hiện có {len(tickers)}")

    def test_no_duplicate_tickers(self):
        """Đảm bảo không có mã cổ phiếu nào bị trùng lặp."""
        tickers = self.assets_config.get_tickers()
        self.assertEqual(len(tickers), len(set(tickers)), "Phát hiện mã cổ phiếu bị trùng lặp!")

    def test_minimum_sectors_count(self):
        """Tiêu chuẩn nghiệm thu: Ít nhất 4 nhóm ngành (ở đây có 5 nhóm ngành)."""
        sector_mapping = self.assets_config.get_sector_to_tickers()
        self.assertGreaterEqual(len(sector_mapping), 4, f"Cần ít nhất 4 sectors, hiện có {len(sector_mapping)}")
        for sector, tickers in sector_mapping.items():
            self.assertGreater(len(tickers), 0, f"Sector {sector} không có cổ phiếu nào!")

    def test_asset_fields_complete(self):
        """Mỗi cổ phiếu phải có đầy đủ: ticker, name, sector, exchange."""
        for asset in self.assets_config.assets:
            self.assertIn("ticker", asset)
            self.assertIn("name", asset)
            self.assertIn("sector", asset)
            self.assertIn("exchange", asset)
            self.assertTrue(asset["ticker"].isupper(), f"Ticker {asset['ticker']} phải viết hoa!")


if __name__ == "__main__":
    unittest.main()

