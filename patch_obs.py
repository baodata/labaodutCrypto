import re

with open('src/env/trading_env.py', 'r') as f:
    content = f.read()

new_obs_func = '''
    def _get_obs(self):
        """Lấy quan sát tại thời điểm t (Market State + Portfolio State)."""
        return {
            "market_state": self.market_tensor.tensor[self.current_step].astype(np.float32),
            "portfolio_weights": self.portfolio.asset_weights.astype(np.float32),
            "cash_ratio": np.array([self.portfolio.cash_weight], dtype=np.float32)
        }
'''

content = re.sub(r'    def _get_obs\(self\):.*?return self\.market_tensor\.tensor\[self\.current_step\]\.astype\(np\.float32\)', new_obs_func.strip('\n'), content, flags=re.DOTALL)

with open('src/env/trading_env.py', 'w') as f:
    f.write(content)
