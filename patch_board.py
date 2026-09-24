with open('manual/BANG_DIEU_PHOI_2_NGUOI.md', 'r') as f:
    text = f.read()

# Sprint 1
text = text.replace(
    "• **DATA-001:** Chọn 20–30 mã cổ phiếu `[CORE]`",
    "• **DATA-001:** Chọn 20–30 mã cổ phiếu `[CORE]` (N=24, SPY chỉ là benchmark)"
)

# Sprint 3
text = text.replace(
    "• **SPLIT-002:** Chuẩn hóa (fit Train) `[CORE]`",
    "• **SPLIT-002:** Chuẩn hóa (fit Train) `[CORE]`<br>• **LEAK-002:** Test mở rộng rò rỉ (MACD, scaler...) `[CORE]`"
)

# Sprint 4
text = text.replace(
    "• **ENV-007:** Thiết kế hàm Reward V1 `[CORE]`<br>• _(A học GCN & viết Toy GCN)_",
    "• **ENV-007:** Thiết kế hàm Reward V1 `[CORE]`<br>• **ENV-009:** Quy ước khớp lệnh Open t+1 `[CORE]`<br>• _(A học GCN & viết Toy GCN)_"
)

# Move EXP-001 from Sprint 6 to Sprint 5
text = text.replace(
    "<br>• **EXP-001:** Bộ đo lường tài chính `[CORE]`",
    ""
)
text = text.replace(
    "• **BASE-004: Single-Agent PPO `[CORE]`** (Primary: A, Secondary: B)",
    "• **BASE-004: Single-Agent PPO `[CORE]`** (Primary: A, Secondary: B)<br>• **EXP-001:** Cài chỉ số tài chính (Sharpe, MDD...) `[CORE]`"
)

# MARL-002/003/004 are already Primary: A in Sprint 6. No change needed.

# Sprint 7
text = text.replace(
    "• **HMARL-001:** State High-Level `[CORE]` (Primary: A)",
    "• **HMARL-000:** Tài liệu thiết kế phân cấp `[CORE]`<br>• **HMARL-001:** State High-Level `[CORE]` (Primary: A)"
)
text = text.replace(
    "🤝 **HMARL-004:** Ghép High-Level Agent (A) và Low-Level Agents (B) vào vòng lặp huấn luyện H-MARL chung",
    "🤝 **HMARL-000:** Đồng bộ tài liệu thiết kế phân cấp<br>🤝 **HMARL-004:** Ghép High-Level Agent (A) và Low-Level Agents (B) vào vòng lặp huấn luyện H-MARL chung"
)

# Sprint 8/9
text = text.replace(
    "<br>• **ADV-EXP-001:** Advanced Graph Ablation `[MỐC C]`",
    ""
)
text = text.replace(
    "• **ADV-GRAPH-001..004:** True Heterogeneous Graph (Supply-chain / Ownership / Macro) `[MỐC C - ADVANCED]`",
    "• **ADV-GRAPH-001..004:** True Heterogeneous Graph (Supply-chain / Ownership / Macro) `[MỐC C - ADVANCED]`<br>• **ADV-GNN-001:** Heterogeneous GNN/GAT `[MỐC C]`<br>• **ADV-EXP-001:** Advanced Graph Ablation `[MỐC C]`"
)

# Mốc dừng
text = text.replace(
    "**Đủ điều kiện hoàn thành và bảo vệ tốt đồ án.** Hệ thống chứng minh trọn vẹn giá trị của phân cấp H-MARL và đồ thị tương quan. Điểm số thực tế phụ thuộc chất lượng mã nguồn, độ sâu thực nghiệm và phần phản biện trước hội đồng.",
    "**Đủ điều kiện hoàn thành và bảo vệ tốt đồ án.** Trả lời RQ1-RQ4. Hệ thống chứng minh trọn vẹn giá trị của phân cấp H-MARL và đồ thị tương quan. Điểm số thực tế phụ thuộc chất lượng mã nguồn, độ sâu thực nghiệm và phần phản biện trước hội đồng."
)
text = text.replace(
    "**Mục tiêu xuất sắc.** Đồ thị đa quan hệ V1 kết hợp cơ chế Attention (GAT) và đánh giá thống kê đa hạt giống (5 seeds) giúp củng cố vững chắc luận điểm khoa học khi phản biện.",
    "**Mục tiêu xuất sắc.** Trả lời RQ5 (cần EXP-008) và RQ6 (cần EXP-010). Đồ thị đa quan hệ V1 kết hợp cơ chế Attention (GAT) và đánh giá thống kê đa hạt giống (5 seeds) giúp củng cố vững chắc luận điểm khoa học khi phản biện."
)

with open('manual/BANG_DIEU_PHOI_2_NGUOI.md', 'w') as f:
    f.write(text)
