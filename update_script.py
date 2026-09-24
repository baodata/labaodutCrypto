import re

# 1. Update assets.yaml
with open('configs/assets.yaml', 'r') as f:
    content = f.read()

agent_group_section = """# Phân nhóm Agent cho Multi-Agent RL (MARL-001)
# Energy và Consumer gộp chung do số lượng tài sản nhỏ (3+5=8)
agent_groups:
  technology:
    name: "Technology Agent"
    sectors: ["technology"]
  financials:
    name: "Financials Agent"
    sectors: ["financials"]
  healthcare:
    name: "Healthcare Agent"
    sectors: ["healthcare"]
  energy_consumer:
    name: "Energy & Consumer Agent"
    sectors: ["energy", "consumer"]

"""

# Insert agent_groups after sectors:
sectors_end_match = re.search(r'    description: "Hàng tiêu dùng thiết yếu và hàng tiêu dùng không thiết yếu"\n\n', content)
if sectors_end_match:
    content = content[:sectors_end_match.end()] + "\n" + agent_group_section + content[sectors_end_match.end():]

# Add agent_group to each asset
def repl_asset(m):
    sector = m.group(1)
    if sector in ['energy', 'consumer']:
        ag = 'energy_consumer'
    else:
        ag = sector
    return f'sector: "{sector}"\n    agent_group: "{ag}"'

content = re.sub(r'sector:\s*"([^"]+)"', repl_asset, content)

with open('configs/assets.yaml', 'w') as f:
    f.write(content)


# 2. Update env.yaml
with open('configs/env.yaml', 'r') as f:
    env_content = f.read()

env_content = env_content.replace(
    "slippage_pct: 0.0005           # Trượt giá (Slippage): 0.05%",
    "slippage_pct: 0.0005           # Trượt giá (Slippage): 0.05%\n  transaction_cost_rate: 0.001   # Tỷ lệ phí giao dịch c dùng trong công thức TC = c · Σ|w - w̃|"
)
env_content = env_content.replace(
    "allow_short_selling: false     # Đồ án hiện tại chỉ đánh Long (Mua/Bán), không Short (Bán khống)",
    "allow_short_selling: false     # Đồ án hiện tại chỉ đánh Long (Mua/Bán), không Short (Bán khống)\n  execution_lag: \"open_t+1\"      # Quy ước khớp lệnh: quyết định cuối ngày t, khớp tại giá Open ngày t+1"
)
env_content = env_content.replace(
    "lookback_window: 20            # Môi trường sẽ cung cấp dữ liệu 20 ngày gần nhất cho AI mỗi bước",
    "lookback_window: 20            # Môi trường sẽ cung cấp dữ liệu 20 ngày gần nhất cho AI mỗi bước\n  covariance_window: 60          # Cửa sổ ước lượng ma trận hiệp phương sai (60 ngày để tránh suy biến với N=24)"
)
env_content = env_content.replace(
    "reward_scaling: 100.0          # Nhân phần thưởng lên 100 lần để Gradient của PPO không bị biến mất",
    "reward_scaling: 100.0          # Nhân phần thưởng lên 100 lần để Gradient của PPO không bị biến mất\n  reward_beta: 1.0               # Hệ số phạt phí giao dịch (β=1 là phạt đúng phí thật, β>1 là reward shaping)"
)

with open('configs/env.yaml', 'w') as f:
    f.write(env_content)
