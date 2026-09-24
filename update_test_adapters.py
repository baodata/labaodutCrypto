with open("tests/test_adapters.py", "r") as f:
    content = f.read()

docstring = '''"""
tests/test_adapters.py
Bộ kiểm thử Dịch thuật dữ liệu (Adapter Tests) giữa Gym Env và GNN Model.

Ticket: MODEL-001 (P0 - Thành viên B)
Sprint: 5

Mục đích kiểm thử:
1. Đảm bảo ObservationAdapter biến đổi đúng ma trận đặc trưng Numpy thành PyTorch Tensor cho GNN.
2. Kiểm tra ActionAdapter lấy đầu ra Softmax từ logits của Actor.
3. Kiểm tra tính năng tự động chèn logit Tiền mặt (Cash logit) vào Action của Môi trường.
"""
'''
with open("tests/test_adapters.py", "w") as f:
    f.write(docstring + content)
