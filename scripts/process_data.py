"""
scripts/process_data.py
Kịch bản chạy Mạng đường ống tạo Đặc trưng (Feature Pipeline) của Thành viên A.
"""
import sys
from pathlib import Path

# Thêm đường dẫn dự án
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.pipeline import FeaturePipeline

def main():
    print("="*60)
    print(" H-MARL-GNN: ĐANG KHỞI ĐỘNG FEATURE PIPELINE (Thành viên A)")
    print("="*60)
    
    pipeline = FeaturePipeline()
    
    print("1. Đang nạp dữ liệu thô và Căn chỉnh lịch giao dịch...")
    print("2. Đang nạp 5 cỗ máy (Return, Volatility, RSI, MACD, Volume)...")
    print("3. Đang đúc Khối Rubik MarketDataTensor [T, N, F]...")
    
    # Hàm run_from_raw làm từ A-Z mọi thứ
    processed_data, market_tensor = pipeline.run_from_raw(
        raw_dir="data/raw", 
        export_parquet=True,
        output_file="data/processed/features.parquet"
    )
    
    print("\n✅ HOÀN TẤT XUẤT SẮC!")
    print(f"-> Tensor đã tạo: {market_tensor.tensor.shape} (T: {market_tensor.tensor.shape[0]} ngày, N: {market_tensor.tensor.shape[1]} cổ phiếu, F: {market_tensor.tensor.shape[2]} tính năng)")
    print(f"-> Danh sách tính năng (F): {market_tensor.feature_names}")
    print("-> Đã lưu thành phẩm vào: data/processed/features.parquet")
    print("="*60)

if __name__ == "__main__":
    main()
