"""
src/graph/builder.py
Module tạo cấu trúc Graph (Node, Edge) từ ma trận tương quan.

- Ticket: GRAPH-002 (Sprint 2 - Member B)
- Nâng cấp: GRAPH-006 (Sprint 4 - Member B)
  + Thêm chế độ k-NN Graph để giữ độ thưa ổn định trong mọi điều kiện thị trường.
  + Thêm Dynamic Threshold thích ứng theo phân phối tương quan hiện tại,
    ngăn đồ thị tiệm cận Fully Connected (fully connected graph) trong giai đoạn khủng hoảng
    (market correlation spike — hiện tượng tương quan đồng loạt tăng lên ~1.0).

FILE NÀY ĐỂ LÀM GÌ?
- Input: Nhận ma trận tương quan, lấy ở correlation.py (graph-001).
- Lọc các cặp cổ phiếu có mức độ tương quan >= or > một threshold nhất định.
- Output: Chuyển đổi dữ liệu thành cấu trúc chuẩn của PyTorch Geometric bao gồm:
  + `edge_index`: Mảng 2D lưu các cặp đỉnh nối với nhau.
  + `edge_weight`: Mảng 1D lưu trọng số của các cạnh (chính là hệ số tương quan).
"""

import numpy as np
import torch
from typing import Tuple, Literal


def generate_correlation_edges(
    corr_matrix: np.ndarray,
    threshold: float = 0.5,
    mode: Literal["threshold", "knn", "dynamic_threshold"] = "threshold",
    top_k: int = 5,
    dynamic_alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tạo edge_index và edge_weight từ một ma trận tương quan 2D tại một thời điểm t.

    Args:
        corr_matrix:
            Numpy array (N x N) biểu diễn ma trận tương quan.
        threshold:
            Ngưỡng tương quan tối thiểu để tạo cạnh — chỉ dùng ở mode="threshold"
            (mặc định 0.5).
        mode:
            Chiến lược xây dựng đồ thị. Có 3 tuỳ chọn:

            • "threshold" (Baseline gốc):
                Tạo cạnh khi |corr| >= threshold cố định.
                Rủi ro: Khi thị trường khủng hoảng (market spike), tương quan
                đồng loạt tăng lên gần 1.0, làm đồ thị tiệm cận Fully-Connected
                và kéo theo hiện tượng Over-smoothing trong GATv2.

            • "knn" (K Nearest Neighbors — KHUYẾN NGHỊ):
                Mỗi node chỉ kết nối với đúng K hàng xóm có |corr| cao nhất.
                Ưu điểm: Giữ độ thưa (sparsity) cố định = N*K cạnh, bất kể thị
                trường đang bình thường hay khủng hoảng. Là lá chắn hiệu quả nhất
                chống Over-smoothing do Market Correlation Spike.

            • "dynamic_threshold" (Ngưỡng thích ứng):
                Tự điều chỉnh ngưỡng dựa trên phân phối tương quan hiện tại:
                    threshold_t = mean(|corr_off_diag|) + alpha * std(|corr_off_diag|)
                Ưu điểm: Khi thị trường spike (mean tăng cao), ngưỡng tự động
                tăng theo, lọc bỏ các cạnh "giả tạo" do xu hướng chung.

        top_k:
            Số hàng xóm K cho chế độ "knn" (mặc định 5).
            Khuyến nghị: K = round(log2(N)) để cân bằng giữa thông tin và sparsity.
        dynamic_alpha:
            Hệ số alpha trong công thức dynamic_threshold (mặc định 1.0).
            Tăng alpha lên (ví dụ: 1.5) để đồ thị thưa hơn trong thời kỳ khủng hoảng.

    Returns:
        edge_index: Tensor kích thước [2, E], kiểu torch.long
        edge_weight: Tensor kích thước [E], kiểu torch.float32
    """
    # Thay thế các giá trị NaN bằng 0 để an toàn
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    N = corr_matrix.shape[0]
    sources = []
    targets = []
    weights = []

    # ── Chế độ 1: Threshold cố định (baseline gốc) ───────────────────────────
    if mode == "threshold":
        for i in range(N):
            for j in range(N):
                if i != j:
                    weight = corr_matrix[i, j]
                    # Chỉ giữ các cạnh có độ lớn tương quan >= threshold
                    # (Dùng abs() nếu bạn cho rằng tương quan ngược chiều cũng là 1 mối liên hệ mạnh)
                    if abs(weight) >= threshold:
                        sources.append(i)
                        targets.append(j)
                        weights.append(weight)

    # ── Chế độ 2: k-NN Graph (KHUYẾN NGHỊ để chống Market Correlation Spike) ─
    elif mode == "knn":
        # Đặt đường chéo về -inf để tránh node tự kết nối với chính nó
        abs_corr = np.abs(corr_matrix.copy())
        np.fill_diagonal(abs_corr, -np.inf)

        k = min(top_k, N - 1)  # Không thể có nhiều hàng xóm hơn số node - 1

        for i in range(N):
            # Lấy K chỉ số có giá trị |corr| cao nhất (không tính chính nó)
            top_k_indices = np.argpartition(abs_corr[i], -k)[-k:]
            for j in top_k_indices:
                weight = corr_matrix[i, j]
                sources.append(i)
                targets.append(j)
                weights.append(float(weight))

    # ── Chế độ 3: Dynamic Threshold thích ứng ───────────────────────────────
    elif mode == "dynamic_threshold":
        abs_corr = np.abs(corr_matrix.copy())

        # Lấy các giá trị ngoài đường chéo (off-diagonal) để tính thống kê
        mask = ~np.eye(N, dtype=bool)
        off_diag_values = abs_corr[mask]

        if len(off_diag_values) == 0:
            pass  # Không có dữ liệu, thoát sớm
        else:
            # Công thức: threshold_t = mean + alpha * std
            # Khi thị trường spike: mean tăng cao → ngưỡng tự tăng theo,
            # lọc bỏ tự động các cạnh "giả tạo" do thị trường cùng pha.
            adaptive_threshold = off_diag_values.mean() + dynamic_alpha * off_diag_values.std()

            for i in range(N):
                for j in range(N):
                    if i != j:
                        weight = corr_matrix[i, j]
                        if abs(weight) >= adaptive_threshold:
                            sources.append(i)
                            targets.append(j)
                            weights.append(weight)
    else:
        raise ValueError(
            f"mode='{mode}' không hợp lệ. Chọn một trong: 'threshold', 'knn', 'dynamic_threshold'."
        )

    # Chuyển đổi sang Torch Tensor theo chuẩn của thư viện PyTorch Geometric
    if len(sources) > 0:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    else:
        # Xử lý an toàn cho trường hợp không có cạnh nào thỏa mãn ngưỡng
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)

    return edge_index, edge_weight
