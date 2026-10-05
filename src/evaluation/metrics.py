import numpy as np
from typing import Dict, Optional

class FinancialMetrics:
    """
    Lớp tính toán các chỉ số đánh giá hiệu quả đầu tư từ chuỗi lợi nhuận hàng ngày.
    """
    def __init__(
        self,
        daily_net_returns: np.ndarray,
        risk_free_rate_annual: float = 0.02,
        trading_days_per_year: int = 252,
        daily_turnover: Optional[np.ndarray] = None,
        daily_costs: Optional[np.ndarray] = None
    ):
        """
        Khởi tạo và tính toán các chỉ số.
        
        Args:
            daily_net_returns: Mảng lợi nhuận ròng hàng ngày.
            risk_free_rate_annual: Lãi suất phi rủi ro hàng năm.
            trading_days_per_year: Số ngày giao dịch trong năm.
            daily_turnover: Tổng tỷ lệ giao dịch hai chiều hàng ngày (tùy chọn).
            daily_costs: Chi phí giao dịch quy ra tiền hàng ngày (tùy chọn).
        """
        self.daily_net_returns = np.asarray(daily_net_returns, dtype=np.float64)
        self.risk_free_rate_annual = risk_free_rate_annual
        self.trading_days_per_year = trading_days_per_year

        if self.daily_net_returns.ndim != 1:
            raise ValueError("daily_net_returns phải là mảng một chiều.")
        if not np.all(np.isfinite(self.daily_net_returns)):
            raise ValueError("daily_net_returns chứa NaN hoặc Inf.")
        if np.any(self.daily_net_returns < -1.0):
            raise ValueError("Lợi nhuận ngày không thể thấp hơn -100% với danh mục giới hạn trách nhiệm.")
        if trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year phải lớn hơn 0.")
        if not np.isfinite(risk_free_rate_annual):
            raise ValueError("risk_free_rate_annual phải hữu hạn.")
        
        T = len(self.daily_net_returns)
        
        # 1. Tỷ suất lợi nhuận tích lũy (Cumulative return)
        cumulative_wealth = float(np.prod(1.0 + self.daily_net_returns))
        if not np.isfinite(cumulative_wealth):
            raise ValueError("Tổng wealth bị tràn số; kiểm tra daily_net_returns.")
        self.cumulative_return = cumulative_wealth - 1.0
        
        # 2. Lợi nhuận kỳ vọng hàng năm (Annualized return)
        if T > 0 and cumulative_wealth > 0.0:
            self.annualized_return = float(
                np.expm1(np.log(cumulative_wealth) * (self.trading_days_per_year / T))
            )
        elif T > 0:
            # A fully wiped-out portfolio has a CAGR of -100%; log(0) is undefined.
            self.annualized_return = -1.0
        else:
            self.annualized_return = 0.0
            
        # 3. Biến động hàng năm (Annualized volatility)
        self.annualized_volatility = np.std(self.daily_net_returns) * np.sqrt(self.trading_days_per_year)
        
        # 4. Tỷ lệ Sharpe (Sharpe ratio)
        if self.annualized_volatility > 1e-6:
            self.sharpe_ratio = (self.annualized_return - self.risk_free_rate_annual) / self.annualized_volatility
        else:
            self.sharpe_ratio = 0.0
            
        # 5. Sụt giảm tối đa (Max drawdown)
        wealth_path = np.concatenate(([1.0], np.cumprod(1.0 + self.daily_net_returns)))
        peak = np.maximum.accumulate(wealth_path)
        drawdowns = (wealth_path - peak) / np.maximum(peak, 1e-8)
        self.max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # 6. Tỷ lệ Calmar (Calmar ratio)
        if abs(self.max_drawdown) > 1e-6:
            self.calmar_ratio = self.annualized_return / abs(self.max_drawdown)
        else:
            self.calmar_ratio = 0.0
            
        # 7 & 8. Turnover và Costs
        self.total_turnover = self._sum_nonnegative(
            daily_turnover, "daily_turnover", expected_length=T
        )
        self.total_costs = self._sum_nonnegative(daily_costs, "daily_costs", expected_length=T)

    @staticmethod
    def _sum_nonnegative(
        values: Optional[np.ndarray], name: str, expected_length: int
    ) -> float:
        if values is None:
            return 0.0
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} phải là mảng một chiều hữu hạn.")
        if len(values) != expected_length:
            raise ValueError(f"{name} phải có cùng số phần tử với daily_net_returns.")
        if np.any(values < 0.0):
            raise ValueError(f"{name} không thể âm.")
        return float(np.sum(values))

    def summary_dict(self) -> Dict[str, float]:
        """Trả về dictionary chứa tất cả các chỉ số."""
        return {
            "cumulative_return": self.cumulative_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "total_turnover": self.total_turnover,
            "total_costs": self.total_costs
        }
        
    def summary_table(self) -> str:
        """In và trả về bảng tóm tắt các chỉ số."""
        lines = [
            "=========================================",
            "          FINANCIAL METRICS SUMMARY      ",
            "=========================================",
            f"Cumulative Return:     {self.cumulative_return:.4%}",
            f"Annualized Return:     {self.annualized_return:.4%}",
            f"Annualized Volatility: {self.annualized_volatility:.4%}",
            f"Sharpe Ratio:          {self.sharpe_ratio:.4f}",
            f"Max Drawdown:          {self.max_drawdown:.4%}",
            f"Calmar Ratio:          {self.calmar_ratio:.4f}",
            f"Total Turnover:        {self.total_turnover:.4f}",
            f"Total Costs:           {self.total_costs:.4f}",
            "========================================="
        ]
        table = "\n".join(lines)
        print(table)
        return table
