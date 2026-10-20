#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/import_alpha_sources.py
MỤC ĐÍCH:
    Nhập và làm sạch (import & clean) các file dữ liệu Cboe/French đã tải sẵn về máy
    cục bộ (dành cho môi trường không có mạng Internet).
    Hỗ trợ giải nén file ZIP của Ken French, căn chỉnh ngày giao dịch, kiểm tra
    các giá trị rỗng/sentinel (-99.99, -999) và chuẩn hóa thành chuỗi chỉ số.

HƯỚNG DẪN CHẠY (CLI USAGE):
    python scripts/import_alpha_sources.py \\
        --vix-csv path/to/VIX_History.csv \\
        --vix3m-csv path/to/VIX3M_History.csv \\
        --french-daily path/to/F-F_Research_Data_Factors_daily_CSV.zip \\
        --output-dir data/external_alpha/

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - Các file CSV hoặc ZIP gốc từ Cboe hoặc Ken French

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - <output-dir>/
        + macro_volatility.csv, ff3_daily.csv
        + import_manifest.json: Mã băm SHA-256 các file đã nhập
==============================================================================
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import re
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory


def cboe_close(path, symbol):
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    value = "CLOSE" if "CLOSE" in frame else symbol
    if "DATE" not in frame or value not in frame:
        raise ValueError(f"Expected official Cboe DATE/CLOSE or DATE/{symbol} columns: {path}")
    result = pd.DataFrame({"date": pd.to_datetime(frame.DATE), symbol.lower(): pd.to_numeric(frame[value])})
    if result.date.duplicated().any() or result.isna().any().any() or (result[symbol.lower()] <= 0).any():
        raise ValueError("Invalid or duplicate Cboe daily observations.")
    return result


def french_daily(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError("Expected one daily factors CSV in the French ZIP.")
            content = archive.read(names[0]).decode("utf-8-sig")
    else:
        content = path.read_text(encoding="utf-8-sig")
    rows = [line for line in content.splitlines() if re.match(r"^\s*\d{8}\s*,", line)]
    if not rows or any(len(line.split(",")) != 5 for line in rows):
        raise ValueError("Expected FF3 daily date/Mkt-RF/SMB/HML/RF, not monthly or FF5.")
    frame = pd.read_csv(StringIO("\n".join(rows)), header=None, names=["date", "mkt_rf", "smb", "hml", "rf"])
    frame["date"] = pd.to_datetime(frame.date.astype(str), format="%Y%m%d")
    if frame.date.duplicated().any() or frame.isna().any().any() or (frame.iloc[:, 1:].abs() >= 100).any().any():
        raise ValueError("Invalid French factors or missing-value sentinels.")
    # French has historically used -99.99/-999 for missing entries.
    if frame.iloc[:, 1:].isin([-99.99, -999.]).any().any():
        raise ValueError("Missing factor sentinel; refuse to treat it as a return.")
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vix-csv", type=Path)
    parser.add_argument("--vix3m-csv", type=Path)
    parser.add_argument("--french-daily", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--research-lag-days", type=int, default=None,
        help="Explicit hypothetical observation-to-release lag, NOT historical publication evidence. Still research-only.")
    args = parser.parse_args(argv)
    if bool(args.vix_csv) != bool(args.vix3m_csv) or not (args.vix_csv or args.french_daily):
        parser.error("Provide both Cboe series and/or --french-daily.")
    if args.research_lag_days is not None and args.research_lag_days < 1:
        parser.error("Research lag must be at least one calendar day.")
    output = args.output_dir or new_run_directory(ROOT / "data/external", "alpha_import")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Import directory is not empty; preserve existing vintages.")
    output.mkdir(parents=True, exist_ok=True)
    fetched = datetime.now(timezone.utc).isoformat()
    sources = []

    def save(frame, name, unit, url, inputs):
        frame = frame.sort_values("date").copy()
        frame["available_at"] = fetched
        if args.research_lag_days is not None:
            assumed = pd.DatetimeIndex(frame.date) + pd.Timedelta(days=args.research_lag_days, hours=16, minutes=30)
            frame["available_at"] = assumed.tz_localize("America/New_York").tz_convert("UTC").astype(str)
        frame["source"], frame["vintage_id"] = url, fetched
        frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
        atomic_csv(frame, output / f"{name}.csv", index=False)
        metadata = dict(availability="current_vintage_research_only", unit=unit, fetched_at=fetched,
            availability_evidence=None, hypothetical_lag_days=args.research_lag_days,
            warning="Lag does not recover historical vintages or establish actual release time.",
            source=url, input_hashes=file_hashes(inputs, ROOT))
        atomic_json(output / f"{name}.json", metadata)
        sources.append(name)

    if args.vix_csv:
        vix, vix3m = cboe_close(args.vix_csv, "VIX"), cboe_close(args.vix3m_csv, "VIX3M")
        # Outer join makes a one-series gap explicit instead of silently dropping it.
        frame = vix.merge(vix3m, on="date", how="outer", validate="one_to_one").sort_values("date")
        # Different historical inception dates are allowed; interior gaps are not.
        first = max(vix.date.min(), vix3m.date.min())
        last = min(vix.date.max(), vix3m.date.max())
        frame = frame.loc[frame.date.between(first, last)]
        if frame.empty or frame.isna().any().any():
            raise ValueError("Unpaired VIX/VIX3M observations inside common historical coverage.")
        save(frame, "macro_volatility", "index_points", "https://www.cboe.com/tradable_products/vix/vix_historical_data",
             [args.vix_csv, args.vix3m_csv])
    if args.french_daily:
        save(french_daily(args.french_daily), "ff3_daily", "percent_return",
             "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html", [args.french_daily])
    atomic_json(output / "import_manifest.json", dict(status="completed", sources=sources,
        availability="current_vintage_research_only", fetched_at=fetched,
        artifact_hashes=file_hashes(output.glob("*.csv"), output.resolve())))
    print(f"Imported: {output}. Current vintage only; strict PIT research will reject these files.")


if __name__ == "__main__":
    main()
