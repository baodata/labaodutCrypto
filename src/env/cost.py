"""
src/env/cost.py
Ticket: ENV-004 (Sprint 4 - Thành viên A)

Bộ máy Mô hình hóa chi phí giao dịch (Transaction Cost Engine).
Xử lý sự kiện "Trôi tỷ trọng" (Drifted Weights) và tính toán phần trăm phí bị trừ
do hành động tái cân bằng danh mục của AI.
"""
import numpy as np


def resolve_transaction_cost_components(
    environment_config: dict,
) -> tuple[float, float, float]:
    """Return (commission, slippage, total) rates from env configuration.

    New configs specify the two components directly. Older configs that only
    define transaction_cost_rate remain supported as a single all-in rate.
    """
    has_components = (
        "transaction_fee_pct" in environment_config
        or "slippage_pct" in environment_config
    )
    if has_components:
        commission = float(environment_config.get("transaction_fee_pct", 0.0))
        slippage = float(environment_config.get("slippage_pct", 0.0))
    else:
        commission = float(environment_config.get("transaction_cost_rate", 0.001))
        slippage = 0.0

    if (
        not np.isfinite(commission)
        or not np.isfinite(slippage)
        or commission < 0.0
        or slippage < 0.0
    ):
        raise ValueError("transaction fee và slippage phải hữu hạn, không âm.")
    return commission, slippage, commission + slippage

class TransactionCostEngine:
    """Động cơ tính toán chi phí giao dịch dựa trên mức độ Turnover."""
    
    def __init__(self, cost_rate: float = 0.001):
        """
        Khởi tạo bộ máy tính phí.
        
        Args:
            cost_rate (c): Tỷ lệ phí giao dịch mặc định (ví dụ 0.001 = 0.1%).
        """
        if not np.isfinite(cost_rate) or cost_rate < 0:
            raise ValueError("cost_rate phải hữu hạn và không âm.")
        self.c = float(cost_rate)
        
    def compute_drifted_weights(
        self, 
        old_weights: np.ndarray, 
        asset_returns: np.ndarray, 
        cash_return: float, 
        portfolio_gross_return: float
    ) -> np.ndarray:
        """
        Tính tỷ trọng đã trôi (Drifted weight).
        Khi giá cổ phiếu tăng/giảm, tỷ trọng danh mục sẽ tự động thay đổi dù AI không làm gì.
        
        Công thức: w̃_i = w_{i,t-1} * (1 + r_i) / (1 + R_p)
        
        Args:
            old_weights: Tỷ trọng kỳ trước (t-1) có kích thước [N+1] (cash ở cuối).
            asset_returns: Lợi nhuận của nhóm cổ phiếu kỳ t [N].
            cash_return: Lợi nhuận sinh ra từ tiền mặt (r_f) kỳ t.
            portfolio_gross_return: Lợi nhuận gộp toàn danh mục (R_p).
            
        Returns:
            Mảng tỷ trọng bị trôi [N+1].
        """
        w_assets = old_weights[:-1]
        w_cash = old_weights[-1]
        
        # Mẫu số chung
        denominator = 1.0 + portfolio_gross_return
        
        # Trôi Cổ phiếu
        drifted_assets = w_assets * (1.0 + asset_returns) / denominator
        
        # Trôi Tiền mặt
        drifted_cash = w_cash * (1.0 + cash_return) / denominator
        
        drifted_weights = np.append(drifted_assets, drifted_cash)
        return drifted_weights
        
    def compute_cost_rate(self, target_weights: np.ndarray, drifted_weights: np.ndarray) -> float:
        """
        Tính Tỷ lệ Phí giao dịch (TC_t) cho lệnh tái cân bằng.
        
        Công thức: TC_t = c * sum_{i=1..N}(|w_{i,t} - w̃_i|); cash là nguồn tiền,
        không phải một chân giao dịch tính phí riêng.
        
        Args:
            target_weights: Tỷ trọng đích do AI quyết định (w_{t}).
            drifted_weights: Tỷ trọng do thị trường tự trôi về (w̃_i).
            
        Returns:
            TC_t (Phần trăm tài sản bị mất vì phí giao dịch).
        """
        target_weights = np.asarray(target_weights, dtype=np.float64)
        drifted_weights = np.asarray(drifted_weights, dtype=np.float64)
        if target_weights.ndim != 1 or target_weights.shape != drifted_weights.shape:
            raise ValueError("target_weights và drifted_weights phải là vector cùng kích thước.")
        if target_weights.size < 2:
            raise ValueError("Weights phải chứa ít nhất một tài sản và tiền mặt.")
        if not np.all(np.isfinite(target_weights)) or not np.all(np.isfinite(drifted_weights)):
            raise ValueError("Weights phải chứa các giá trị hữu hạn.")

        # Turnover counts traded risky assets once; the cash leg is their funding source.
        turnover = np.sum(np.abs(target_weights[:-1] - drifted_weights[:-1]))
        
        # Phí giao dịch tỷ lệ thuận với Turnover
        return float(self.c * turnover)


def square_root_impact(notionals, dollar_adv, daily_volatility, eta=0.):
    """Return per-stock dollar impact and participation, using causal ADV.

    Trade notionals are measured at execution. No cash leg is accepted.
    Missing capacity blocks active impact trades, rather than charging zero.
    """
    q, adv, sigma = np.broadcast_arrays(np.asarray(notionals, float),
        np.asarray(dollar_adv, float), np.asarray(daily_volatility, float))
    if not np.isfinite(eta) or eta < 0 or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Invalid impact coefficient/notional.")
    active = q > 1e-12
    if eta > 0 and np.any(active & (~np.isfinite(adv) | (adv <= 0) | ~np.isfinite(sigma) | (sigma < 0))):
        raise ValueError("Active impact trades require valid causal ADV and volatility.")
    participation = np.divide(q, adv, out=np.zeros_like(q), where=np.isfinite(adv) & (adv > 0))
    impact = np.zeros_like(q)
    if eta > 0:
        impact[active] = q[active] * eta * sigma[active] * np.sqrt(participation[active])
    return impact, participation


def execute_cash_funded_trade(open_weights, target_weights, nav, commission, slippage,
                              adv, sigma, eta=0., max_asset_weight=1.):
    """Execute actual stock dollar legs and deduct fees from cash.

    Scale only buys if fees cannot be funded; holds and sales retain their
    intended quantities. Also cap new buys against NAV AFTER costs.
    """
    old = np.asarray(open_weights[:-1], float) * nav
    desired = np.asarray(target_weights[:-1], float) * nav
    delta = desired - old
    sells = np.minimum(delta, 0.)
    buys = np.maximum(delta, 0.)
    cash_available = float(open_weights[-1]*nav - sells.sum())
    scale = 1.
    for _ in range(100):
        traded = sells + buys*scale
        q = np.abs(traded)
        impact, participation = square_root_impact(q, adv, sigma, eta)
        fee = q.sum()*(commission+slippage) + impact.sum()
        if fee >= nav:
            raise ValueError("Execution costs exhaust NAV; reduce participation/impact.")
        capped = np.maximum(max_asset_weight*(nav-fee)-old-sells, 0.)
        limited_buys = np.minimum(buys, capped)
        # All buy legs are scaled together only for cash funding.
        if np.any(limited_buys < buys-1e-12):
            buys = limited_buys
            scale = min(scale, 1.)
            continue
        affordable = max(0., cash_available-fee)
        next_scale = min(scale, affordable / buys.sum()) if buys.sum() > 0 else 1.
        if abs(next_scale-scale) <= 1e-12:
            break
        scale = next_scale
    else:
        raise RuntimeError("Execution cash/fee fixed point did not converge.")
    traded = sells + buys*scale
    q = np.abs(traded)
    impact, participation = square_root_impact(q, adv, sigma, eta)
    commission_cost, slippage_cost = q.sum()*commission, q.sum()*slippage
    cost = float(commission_cost+slippage_cost+impact.sum())
    new_assets = old+traded
    new_cash = nav-new_assets.sum()-cost
    if new_cash < -1e-7 or cost >= nav:
        raise ValueError("Execution cannot fund fees without a negative cash balance.")
    weights = np.append(new_assets, max(0.,new_cash))/(nav-cost)
    return weights, dict(transaction_cost=cost, commission_cost=float(commission_cost),
        slippage_cost=float(slippage_cost), impact_cost=float(impact.sum()),
        participation=participation, stock_costs=q*(commission+slippage)+impact, turnover=float(q.sum()/nav), cost_rate=cost/nav)
