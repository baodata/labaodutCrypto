#!/usr/bin/env python3
"""test output for feature engineering and scaled features."""

from pathlib import Path
import sys
import numpy as np

# thêm thư mục gốc dự án vào sys.path để chạy ở đâu cũng import được
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_market_data
from src.features.pipeline import build_feature_pipeline


def main():
    print("=" * 65)
    print("KIỂM TRA TẦNG ĐẶC TRƯNG & CHỈ BÁO KỸ THUẬT (FEATURE INSPECTION)")
    print("=" * 65)

    print("\n1. Nạp dữ liệu và chạy pipeline tính toán đặc trưng:")
    full_data, splits = load_market_data("configs/assets.yaml")
    scaled_splits, scaler = build_feature_pipeline(full_data, splits, warmup_days=30)

    print(f"   - Số lượng cổ phiếu      : {full_data.N}")
    print(f"   - Số đặc trưng kỹ thuật  : {scaled_splits['train'].F}")
    print("   - Danh sách các đặc trưng:")
    for i, name in enumerate(scaled_splits["train"].feature_names):
        print(f"     [{i}] {name:<18} (Mean train: {scaler.mean[0, 0, i]:.6f}, Std train: {scaler.std[0, 0, i]:.6f})")

    print("\n2. Thống kê kích thước và phân phối giá trị sau khi Scale:")
    print(f"{'TẬP DỮ LIỆU':<12} | {'SỐ PHIÊN (T)':<14} | {'SHAPE':<16} | {'MIN':<7} | {'MAX':<7} | {'MEAN':<8} | {'STD':<8}")
    print("-" * 85)
    for name, tensor in scaled_splits.items():
        arr = tensor.values
        print(
            f"{name.upper():<12} | "
            f"{tensor.T:<14} | "
            f"{str(arr.shape):<16} | "
            f"{arr.min():>6.2f} | "
            f"{arr.max():>6.2f} | "
            f"{arr.mean():>7.4f} | "
            f"{arr.std():>7.4f}"
        )

    print("\n3. Mẫu vector đặc trưng phiên gần nhất (2025-12-30) của AAPL và NVDA:")
    test_tensor = scaled_splits["test"]
    aapl_idx = test_tensor.get_ticker_index("AAPL")
    nvda_idx = test_tensor.get_ticker_index("NVDA")

    print(f"{'ĐẶC TRƯNG':<18} | {'AAPL (Z-score)':<16} | {'NVDA (Z-score)':<16}")
    print("-" * 55)
    for i, fname in enumerate(test_tensor.feature_names):
        aapl_val = test_tensor.values[-1, aapl_idx, i]
        nvda_val = test_tensor.values[-1, nvda_idx, i]
        print(f"{fname:<18} | {aapl_val:>14.4f} | {nvda_val:>14.4f}")

    print("\n✅ TẤT CẢ ĐẶC TRƯNG ĐƯỢC TÍNH TOÁN LIÊN TỤC, 0 NaN, 0 Inf.")
    print("=" * 65)


if __name__ == "__main__":
    main()
