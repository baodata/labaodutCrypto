with open("tests/test_actor_critic.py", "r") as f:
    content = f.read()

docstring = '''"""
tests/test_actor_critic.py
Bộ kiểm thử Kiến trúc Đầu não ra Quyết định (GNN Actor-Critic Tests).

Ticket: BASE-004 (Sprint 5 - Thành viên B & A)
Sprint: 5

Mục đích kiểm thử:
1. Đảm bảo luồng đi tiến (Forward Pass) của mô hình chạy trơn tru với kích thước đúng: Actor xuất [N], Critic xuất [1].
2. Kiểm tra việc kết nối GNN Encoder (GCN/GAT) vào Actor-Critic có đồng bộ về kích thước chiều (hidden dimension).
3. Đảm bảo luồng đi lùi (Backward Pass) có thể truyền gradient xuyên suốt từ đầu ra (Actor/Critic) về tận trọng số của GNN Encoder.
"""
'''
with open("tests/test_actor_critic.py", "w") as f:
    f.write(docstring + content)
