import numpy as np
from typing import Dict, Optional

class FinancialMetrics:
    """
    Lớp tính toán các chỉ số đánh giá hiệu quả đầu tư từ chuỗi lợi nhuận hàng ngày.
    """
    SHARPE_CONVENTION = "daily_arithmetic_excess_sample_std"

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
        daily_std = float(np.std(self.daily_net_returns, ddof=1)) if T > 1 else 0.0
        self.annualized_volatility = daily_std * np.sqrt(self.trading_days_per_year)
        
        # 4. Tỷ lệ Sharpe (Sharpe ratio)
        # Use arithmetic daily excess returns. CAGR remains a separate metric.
        # The daily risk-free convention matches PortfolioReturnEngine.
        if daily_std > 1e-12:
            excess = self.daily_net_returns - self.risk_free_rate_annual / self.trading_days_per_year
            self.sharpe_ratio = float(
                np.sqrt(self.trading_days_per_year) * np.mean(excess) / daily_std
            )
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


def block_bootstrap_excess_returns(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    block_size: int = 5,
    num_samples: int = 2000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """
    Circular Block Bootstrap (CBB) trên chuỗi excess return hàng ngày (strategy - benchmark).
    
    Returns dict gồm:
        mean_excess_annualized: Lợi nhuận vượt trội trung bình năm hóa.
        std_excess_annualized: Sai số chuẩn của excess return năm hóa.
        ci_lower_95: Cận dưới khoảng tin cậy 95%.
        ci_upper_95: Cận trên khoảng tin cậy 95%.
        p_value_positive_excess: One-sided test under centered H0: mean excess = 0.
        bootstrap_fraction_excess_le_0: Descriptive tail fraction, not a null p-value.
    """
    strategy_returns = np.asarray(strategy_returns, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)
    if strategy_returns.ndim != 1 or strategy_returns.shape != benchmark_returns.shape:
        raise ValueError("Strategy và benchmark phải là chuỗi một chiều cùng độ dài.")
    if not np.isfinite(strategy_returns).all() or not np.isfinite(benchmark_returns).all():
        raise ValueError("Returns bootstrap phải hữu hạn.")
    if num_samples <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Số mẫu bootstrap/confidence_level không hợp lệ.")
    excess = strategy_returns - benchmark_returns
    T = len(excess)
    if T < block_size or block_size <= 0:
        raise ValueError("T phải lớn hơn block_size.")
    
    rng = np.random.default_rng(seed)
    num_blocks = int(np.ceil(T / block_size))
    boot_means = np.empty(num_samples, dtype=np.float64)
    offsets = np.arange(block_size)
    for first in range(0, num_samples, 256):
        count = min(256, num_samples - first)
        starts = rng.integers(0, T, size=(count, num_blocks))
        indices = (starts[..., None] + offsets) % T
        samples = excess[indices].reshape(count, -1)[:, :T]
        boot_means[first:first + count] = samples.mean(axis=1)
    annualized_means = boot_means * 252.0
    observed = float(np.mean(excess) * 252.0)
    # The same block draws from centered returns implement the boundary null.
    centered_null_means = annualized_means - observed
    p_positive = (1 + np.count_nonzero(centered_null_means >= observed)) / (num_samples + 1)
    tail = (1.0 - confidence_level) / 2.0
    
    return {
        "mean_excess_annualized": observed,
        "std_excess_annualized": float(np.std(annualized_means)),
        "ci_lower_95": float(np.quantile(annualized_means, tail)),
        "ci_upper_95": float(np.quantile(annualized_means, 1.0 - tail)),
        "p_value_positive_excess": float(p_positive),
        "bootstrap_fraction_excess_le_0": float(np.mean(annualized_means <= 0.0)),
    }
