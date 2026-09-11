"""Raw Market Data Validator.

Validates OHLCV historical time series data to ensure financial logic consistency,
timeline integrity, non-negativity of prices/volume, and absence of NaN/Inf values.
Supports both TitleCase ('Open', 'High', ...) and lowercase ('open', 'high', ...)
conventions, and automatically handles 'Date' / 'date' columns or DatetimeIndex.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Represents the outcome of a dataset validation."""

    is_valid: bool
    ticker: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        """Raise ValueError if the dataset is not valid."""
        if not self.is_valid:
            error_details = "\n - ".join(self.errors)
            msg = f"Validation failed for asset '{self.ticker}':\n - {error_details}"
            raise ValueError(msg)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [f"Validation [{status}] for '{self.ticker}':"]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"    - {err}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"    - {warn}")
        lines.append(f"  Stats: {self.stats}")
        return "\n".join(lines)


class DataValidator:
    """Validates OHLCV financial data against financial and data-integrity rules."""

    REQUIRED_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, allow_zero_volume: bool = True) -> None:
        """Initialize DataValidator.

        Args:
            allow_zero_volume: If True, volume == 0 generates a warning instead of an error.
        """
        self.allow_zero_volume = allow_zero_volume

    def _normalize_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Normalize column names and ensure DatetimeIndex."""
        errors: list[str] = []
        df_norm = df.copy()

        # Handle Date column if index is not DatetimeIndex
        if not isinstance(df_norm.index, pd.DatetimeIndex):
            date_col = next((c for c in df_norm.columns if str(c).lower() == "date"), None)
            if date_col is not None:
                try:
                    df_norm[date_col] = pd.to_datetime(df_norm[date_col])
                    df_norm = df_norm.set_index(date_col)
                    df_norm.index.name = "Date"
                except Exception as exc:
                    errors.append(f"Failed to parse '{date_col}' column into DatetimeIndex: {exc}")
            else:
                errors.append(f"DataFrame index must be a pd.DatetimeIndex or contain a 'date' column, got {type(df_norm.index).__name__}.")

        # Standardize column naming to TitleCase (Open, High, Low, Close, Volume)
        col_map: dict[str, str] = {}
        for c in df_norm.columns:
            c_str = str(c).strip()
            c_lower = c_str.lower()
            if c_lower == "open":
                col_map[c] = "Open"
            elif c_lower == "high":
                col_map[c] = "High"
            elif c_lower == "low":
                col_map[c] = "Low"
            elif c_lower == "close":
                col_map[c] = "Close"
            elif c_lower == "adj_close" or c_lower == "adj close":
                col_map[c] = "Adj Close"
            elif c_lower == "volume":
                col_map[c] = "Volume"

        df_norm = df_norm.rename(columns=col_map)
        return df_norm, errors

    def validate_dataframe(self, df: pd.DataFrame, ticker: str = "") -> ValidationResult:
        """Validate an in-memory OHLCV DataFrame.

        Rules verified:
        1. Non-empty DataFrame.
        2. Presence of required OHLCV columns (case-insensitive).
        3. DatetimeIndex is monotonic increasing (chronological) and unique (no duplicates).
        4. Prices (Open, High, Low, Close) are strictly positive (> 0).
        5. Volume is non-negative (>= 0).
        6. OHLC consistency: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close.
        7. No NaN, null, or infinite (Inf / -Inf) values.

        Args:
            df: DataFrame containing OHLCV series.
            ticker: Optional ticker symbol for identification.

        Returns:
            ValidationResult containing status, errors, warnings, and metadata stats.
        """
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {
            "num_rows": len(df),
            "ticker": ticker,
        }

        # Rule 1: Non-empty DataFrame
        if df.empty:
            errors.append("DataFrame is completely empty (0 rows).")
            return ValidationResult(is_valid=False, ticker=ticker, errors=errors, warnings=warnings, stats=stats)

        # Normalize schema (handle date column & case-insensitive column names)
        df_clean, norm_errors = self._normalize_dataframe(df)
        if norm_errors:
            errors.extend(norm_errors)
            return ValidationResult(is_valid=False, ticker=ticker, errors=errors, warnings=warnings, stats=stats)

        # Rule 2: Required OHLCV columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df_clean.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}. Expected at least {self.REQUIRED_COLUMNS}.")
            return ValidationResult(is_valid=False, ticker=ticker, errors=errors, warnings=warnings, stats=stats)

        # Rule 3: Timeline & Index Integrity
        if not df_clean.index.is_monotonic_increasing:
            errors.append("Dates are not sorted chronologically (index is not monotonic increasing).")
        if not df_clean.index.is_unique:
            dup_count = df_clean.index.duplicated().sum()
            errors.append(f"Index contains {dup_count} duplicate timestamp(s).")

        stats["start_date"] = str(df_clean.index.min().date())
        stats["end_date"] = str(df_clean.index.max().date())

        # Rule 4: No NaN / Null or Infinite values in required columns
        for col in self.REQUIRED_COLUMNS:
            nan_count = df_clean[col].isna().sum()
            if nan_count > 0:
                errors.append(f"Column '{col}' contains {nan_count} NaN/null value(s).")

            # Check infinite values on numeric types
            if np.issubdtype(df_clean[col].dtype, np.number):
                inf_count = np.isinf(df_clean[col]).sum()
                if inf_count > 0:
                    errors.append(f"Column '{col}' contains {inf_count} infinite value(s).")

        # If there are NaN values in price/volume, subsequent numeric checks will fail or behave unexpectedly
        if errors:
            return ValidationResult(is_valid=False, ticker=ticker, errors=errors, warnings=warnings, stats=stats)

        # Rule 5: Strictly positive prices
        for price_col in ["Open", "High", "Low", "Close"]:
            le_zero = (df_clean[price_col] <= 0).sum()
            if le_zero > 0:
                errors.append(f"Column '{price_col}' has {le_zero} non-positive values (<= 0).")

        # Rule 6: Non-negative volume
        neg_vol = (df_clean["Volume"] < 0).sum()
        if neg_vol > 0:
            errors.append(f"Column 'Volume' has {neg_vol} negative value(s) (< 0).")

        zero_vol = (df_clean["Volume"] == 0).sum()
        stats["zero_volume_days"] = int(zero_vol)
        if zero_vol > 0:
            msg = f"Asset '{ticker}' has {zero_vol} trading days with Volume == 0."
            if self.allow_zero_volume:
                warnings.append(msg)
            else:
                errors.append(msg)

        # Rule 7: OHLC Consistency
        # High >= Low
        invalid_hl = (df_clean["High"] < df_clean["Low"]).sum()
        if invalid_hl > 0:
            errors.append(f"High < Low on {invalid_hl} rows.")

        # High >= Open and High >= Close
        invalid_ho = (df_clean["High"] < df_clean["Open"]).sum()
        if invalid_ho > 0:
            errors.append(f"High < Open on {invalid_ho} rows.")

        invalid_hc = (df_clean["High"] < df_clean["Close"]).sum()
        if invalid_hc > 0:
            errors.append(f"High < Close on {invalid_hc} rows.")

        # Low <= Open and Low <= Close
        invalid_lo = (df_clean["Low"] > df_clean["Open"]).sum()
        if invalid_lo > 0:
            errors.append(f"Low > Open on {invalid_lo} rows.")

        invalid_lc = (df_clean["Low"] > df_clean["Close"]).sum()
        if invalid_lc > 0:
            errors.append(f"Low > Close on {invalid_lc} rows.")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, ticker=ticker, errors=errors, warnings=warnings, stats=stats)

    def validate_file(self, file_path: Path | str, ticker: str | None = None) -> ValidationResult:
        """Load and validate an OHLCV Parquet or CSV file.

        Args:
            file_path: Path to parquet or csv file.
            ticker: Optional ticker name (defaults to file stem if None).

        Returns:
            ValidationResult.
        """
        path = Path(file_path)
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                ticker=ticker or path.stem,
                errors=[f"File not found: {path}"],
            )

        symbol = ticker if ticker is not None else path.stem.upper()

        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".csv":
                df = pd.read_csv(path)
            else:
                return ValidationResult(
                    is_valid=False,
                    ticker=symbol,
                    errors=[f"Unsupported file format '{path.suffix}'. Expected .parquet or .csv."],
                )
        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                ticker=symbol,
                errors=[f"Failed to read file '{path}': {exc}"],
            )

        return self.validate_dataframe(df, ticker=symbol)

    def validate_raw_dir(self, raw_dir: Path | str) -> dict[str, ValidationResult]:
        """Validate all parquet files in a directory.

        Args:
            raw_dir: Path to directory containing raw asset parquet files.

        Returns:
            Dictionary mapping ticker symbol to ValidationResult.
        """
        dir_path = Path(raw_dir)
        results: dict[str, ValidationResult] = {}
        parquet_files = sorted(dir_path.glob("*.parquet"))

        for file_path in parquet_files:
            ticker = file_path.stem.upper()
            result = self.validate_file(file_path, ticker=ticker)
            results[ticker] = result

        return results

