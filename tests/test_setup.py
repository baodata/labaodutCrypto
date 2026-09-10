"""
tests/test_setup.py
Unit tests cho việc thiết lập dự án và cấu trúc thư mục (Epic 0).

- Ticket: EPIC-0 (Sprint 1 - Member A)

FILE NÀY ĐỂ LÀM GÌ?
- Kiểm tra phiên bản Python, import các package cốt lõi, và cấu trúc thư mục của dự án.
"""


import sys
import unittest
from pathlib import Path


class TestProjectFoundation(unittest.TestCase):
    """Kiểm tra nền tảng dự án và cấu trúc thư mục (Epic 0)."""

    def test_python_version(self):
        """Kiểm tra phiên bản Python >= 3.10."""
        self.assertGreaterEqual(
            sys.version_info[:2],
            (3, 10),
            f"Yêu cầu Python >= 3.10, hiện tại là {sys.version}",
        )

    def test_src_package_import(self):
        """Kiểm tra import thành công package src và các sub-modules."""
        import src
        import src.data
        import src.features
        import src.graph
        import src.env
        import src.models
        import src.agents
        import src.training
        import src.evaluation
        import src.utils

        self.assertIsNotNone(src)

    def test_directory_structure(self):
        """Kiểm tra sự tồn tại của các thư mục cốt lõi trong dự án."""
        project_root = Path(__file__).resolve().parent.parent
        expected_dirs = [
            "configs",
            "data/raw",
            "data/interim",
            "data/processed",
            "src",
            "tests",
            "notebooks",
            "experiments",
            "scripts",
        ]
        for dir_path in expected_dirs:
            full_path = project_root / dir_path
            self.assertTrue(
                full_path.exists() and full_path.is_dir(),
                f"Thư mục {dir_path} chưa được tạo!",
            )

    def test_core_dependencies(self):
        """Kiểm tra khả năng import các thư viện cốt lõi đã cài trong môi trường."""
        import pandas as pd
        import numpy as np
        import yfinance as yf

        self.assertIsNotNone(pd)
        self.assertIsNotNone(np)
        self.assertIsNotNone(yf)


if __name__ == "__main__":
    unittest.main()

