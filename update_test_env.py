with open("tests/test_portfolio_env.py", "r") as f:
    content = f.read()

docstring = '''"""
tests/test_portfolio_env.py
Bộ kiểm thử Môi trường Giao dịch Chứng khoán (Trading Environment Tests).

Ticket: ENV-001 -> ENV-007 (Sprint 4 - Thành viên A)
Sprint: 4

Mục đích kiểm thử:
1. Kiểm tra tính đúng đắn của Không gian Hành động (Action Space) gồm N tài sản và 1 Tiền mặt.
2. Kiểm tra tính năng ép tỷ trọng (Weight Projection) để đảm bảo không âm và tổng bằng 1.0.
3. Kiểm tra chi phí giao dịch (Transaction Cost) có áp dụng logic Drifted Weight theo thị trường thực tế.
4. Kiểm tra lợi nhuận danh mục (Net Return) và phần thưởng (Reward) theo thiết kế hệ số beta.
"""
'''
with open("tests/test_portfolio_env.py", "w") as f:
    f.write(docstring + content)
