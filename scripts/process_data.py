#!/usr/bin/env python3
"""
scripts/process_data.py
Script thực hiện Pipeline làm sạch dữ liệu và tính toán đặc trưng.

Ticket liên quan: DATA-003 (Validate), DATA-004 (Align), FEAT-001 (Returns)
Sprint: 2, 3

Quy trình:
1. Quét toàn bộ file trong data/raw/
2. Chạy qua DataValidator để đảm bảo dữ liệu thô không có rác (QC).
3. Chạy qua CalendarAligner để đồng bộ ngày giao dịch (giao nhau) cho tất cả cổ phiếu.
4. Xuất ra bảng giá chung [T, N].
5. Tính toán Tỷ suất lợi nhuận (Return) [T, N].
6. (Tương lai) Tính thêm RSI, MACD, Volatility... và gom thành [T, N, F].
7. Lưu kết quả ra thư mục data/processed/.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Thêm đường dẫn gốc của dự án vào sys.path để import dễ dàng
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.validator import DataValidator
from src.data.alignment import CalendarAligner
from src.features.returns import compute_simple_returns

def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(" H-MARL-GNN: QUY TRÌNH LÀM SẠCH VÀ TẠO FEATURE")
    print("="*60)

    # ---------------------------------------------------------
    # BƯỚC 1: KIỂM ĐỊNH DỮ LIỆU THÔ (DATA-003)
    # ---------------------------------------------------------
    print("\n[1/4] Đang kiểm định chất lượng dữ liệu thô (Validation)...")
    validator = DataValidator(allow_zero_volume=True)
    validation_results = validator.validate_raw_dir(raw_dir)
    
    valid_tickers = []
    for ticker, res in validation_results.items():
        if res.is_valid:
            valid_tickers.append(ticker)
        else:
            print(f"  ❌ {ticker} bị loại do lỗi: {res.errors}")
            
    if not valid_tickers:
        print("\n[LỖI] Không có mã cổ phiếu nào vượt qua bài kiểm tra chất lượng!")
        sys.exit(1)
        
    print(f"  -> Có {len(valid_tickers)} mã hợp lệ đi tiếp vào vòng trong.")

    # ---------------------------------------------------------
    # BƯỚC 2: ĐỒNG BỘ LỊCH GIAO DỊCH (DATA-004)
    # ---------------------------------------------------------
    print("\n[2/4] Đang đồng bộ lịch giao dịch (Alignment)...")
    aligner = CalendarAligner(method="intersection")
    
    # Ở đây ta giả định aligner.align_from_directory nhận danh sách file hoặc thư mục
    # Vì ta đã có valid_tickers, ta sẽ tự load lên dictionary để an toàn hơn
    raw_data_dict = {}
    for ticker in valid_tickers:
        file_path = raw_dir / f"{ticker}.parquet"
        if not file_path.exists():
            file_path = raw_dir / f"{ticker}.csv"
        
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        raw_data_dict[ticker] = df

    align_result = aligner.align(raw_data_dict)
    print(f"  -> {align_result.summary()}")

    # ---------------------------------------------------------
    # BƯỚC 3: XUẤT BẢNG GIÁ CHUNG VÀ TÍNH RETURN (FEAT-001)
    # ---------------------------------------------------------
    print("\n[3/4] Trích xuất bảng giá [T, N] và tính Return...")
    # Lấy cột giá đóng cửa để tạo bảng giá (Mặc định hàm to_price_panel của A có thể lấy 'close')
    try:
        price_panel = aligner.to_price_panel(align_result.aligned_data, price_col='close')
    except Exception:
        price_panel = aligner.to_price_panel(align_result.aligned_data, price_col='Close')
        
    # Tính toán Lợi suất (Returns)
    returns_panel = compute_simple_returns(price_panel, fill_zero=True)
    print(f"  -> Bảng Return đã sẵn sàng. Kích thước: {returns_panel.shape} [T, N]")

    # ---------------------------------------------------------
    # BƯỚC 4: LƯU TRỮ (DATA-005)
    # ---------------------------------------------------------
    print("\n[4/4] Lưu kết quả ra data/processed/...")
    
    price_path = processed_dir / "aligned_prices.parquet"
    returns_path = processed_dir / "returns.parquet"
    
    price_panel.to_parquet(price_path)
    returns_panel.to_parquet(returns_path)
    
    print(f"  ✅ Đã lưu {price_path}")
    print(f"  ✅ Đã lưu {returns_path}")
    print("\n🎉 QUY TRÌNH HOÀN TẤT. Dữ liệu đã sẵn sàng cho hệ thống Graph!")

if __name__ == "__main__":
    main()
