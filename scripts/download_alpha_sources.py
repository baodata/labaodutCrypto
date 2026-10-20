"""Download official alpha inputs with provenance; never manufacture PIT releases.

Current Cboe/French files are current vintages. Fed annual announcements provide
dated schedule evidence, but cannot establish full earnings/calendar coverage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import urllib.request

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.import_alpha_sources import cboe_close, french_daily
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory
from src.utils.config import AssetsConfig


SERIES = {
    "VIX_History.csv": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX3M_History.csv": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "F-F_Research_Data_Factors_daily_CSV.zip": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
}
FRED_SERIES = {
    "FRED_VIXCLS.csv": "https://fred.stlouisfed.org/graph/graph.csv?id=VIXCLS",
    "FRED_VXVCLS.csv": "https://fred.stlouisfed.org/graph/graph.csv?id=VXVCLS",
}
YAHOO_SERIES = {
    "Yahoo_VIX.json": "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?period1=1388534400&period2=1672531200&interval=1d",
    "Yahoo_VIX3M.json": "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX3M?period1=1388534400&period2=1672531200&interval=1d",
}
FOMC = {
    2015: "20140605b", 2016: "20150511a", 2017: "20160628a", 2018: "20170511b",
    2019: "20180525a", 2020: "20190517a", 2021: "20200702a", 2022: "20210604a",
}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")


class PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pieces = []

    def handle_data(self, data):
        self.pieces.append(data)


def fetch(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (academic historical data research)"})
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = response.read()
        if not payload:
            raise ValueError(f"Empty official source: {url}")
        destination.write_bytes(payload)
        return dict(source=url, file=str(destination), bytes=len(payload), http_status=response.status,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            content_type=response.headers.get("Content-Type"),
            last_modified=response.headers.get("Last-Modified"), final_url=response.url)


def fomc_records(year, path, url):
    parser = PageText()
    parser.feed(path.read_text(encoding="utf-8"))
    text = " ".join(parser.pieces)
    stamp = re.search(r"monetary(\d{8})[a-z]\.htm", url).group(1)
    release_day = datetime.strptime(stamp, "%Y%m%d")
    # Entire schedule must actually appear in the downloaded primary document.
    months = "|".join(MONTHS)
    expression = rf"\b({months})\s+(\d{{1,2}})\s*[-–]\s*(?:({months})\s+)?(\d{{1,2}})(?:,\s*(\d{{4}}))?\s*\((?:Monday|Tuesday|Wednesday|Thursday|Friday)"
    matches = list(re.finditer(expression, text))
    if len(matches) != 9:
        raise ValueError(f"Expected 8 meetings plus next January in Fed announcement {year}; found {len(matches)}.")
    # A day-end known_at is conservative even when the source has an exact hour.
    known = pd.Timestamp(release_day).tz_localize("America/New_York") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    result = []
    for match in matches:
        first_month, first_day, end_month, end_day, explicit_year = match.groups()
        last_year = int(explicit_year) if explicit_year else year
        end_month = end_month or first_month
        day = pd.Timestamp(year=last_year, month=MONTHS.index(end_month)+1, day=int(end_day))
        scheduled = day.tz_localize("America/New_York") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        result.append(dict(event_id=f"fomc_regular_{day:%Y%m%d}", event_type="fomc", ticker="*",
            scheduled_at=scheduled.isoformat(), known_at=known.isoformat(), status="scheduled",
            source=url, vintage_id=f"Fed_announcement_{stamp}"))
    return result


def fred_close(path, series, name):
    frame = pd.read_csv(path)
    date_column = "observation_date" if "observation_date" in frame else "DATE"
    if date_column not in frame or series not in frame:
        raise ValueError(f"Unexpected official FRED export schema: {path}")
    result = pd.DataFrame({"date": pd.to_datetime(frame[date_column]),
                           name: pd.to_numeric(frame[series], errors="coerce")})
    if result.date.isna().any() or result.date.duplicated().any() or (result[name].dropna() <= 0).any():
        raise ValueError("Invalid FRED daily index values/dates.")
    return result


def yahoo_close(path, symbol, name):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["chart"].get("error") or not payload["chart"].get("result"):
        raise ValueError(f"Yahoo chart error: {path}")
    result = payload["chart"]["result"][0]
    if result["meta"]["symbol"] != symbol:
        raise ValueError("Yahoo response ticker differs from requested volatility index.")
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    frame = pd.DataFrame({"date": dates, name: result["indicators"]["quote"][0]["close"]})
    if frame.date.duplicated().any() or (frame[name].dropna() <= 0).any():
        raise ValueError("Invalid Yahoo volatility index dates/prices.")
    return frame


def materialize(output, acquisition):
    raw, normalized = output / "raw", output / "normalized"
    normalized.mkdir(parents=True, exist_ok=True)
    sources = {row["name"]: row for row in acquisition if row["status"] == "downloaded"}
    statuses = []

    def save_series(frame, stem, unit, input_names):
        # Per-row availability is actual acquisition, never the historical date.
        received = max(sources[name]["retrieved_at"] for name in input_names)
        frame = frame.sort_values("date").copy()
        frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
        frame["available_at"] = received
        frame["source"] = " | ".join(sources[name]["source"] for name in input_names)
        frame["vintage_id"] = received
        atomic_csv(frame, normalized / f"{stem}.csv", index=False)
        atomic_json(normalized / f"{stem}.json", dict(availability="current_vintage_research_only",
            availability_evidence=None, unit=unit, retrieved_at=received,
            input_hashes=file_hashes([raw/name for name in input_names], output.resolve()),
            warning="Current release downloaded now. No historical available_at/vintage reconstruction.",
            source_records=[sources[name] for name in input_names]))
        statuses.append(dict(group=stem, status="downloaded_current_vintage", rows=len(frame),
            first_date=frame.date.iloc[0], last_date=frame.date.iloc[-1], strict_history_ready=False))

    macro_names = None
    if {"VIX_History.csv", "VIX3M_History.csv"} <= set(sources):
        a, b = cboe_close(raw/"VIX_History.csv", "VIX"), cboe_close(raw/"VIX3M_History.csv", "VIX3M")
        macro_names = ["VIX_History.csv", "VIX3M_History.csv"]
    elif set(FRED_SERIES) <= set(sources):
        a = fred_close(raw/"FRED_VIXCLS.csv", "VIXCLS", "vix")
        b = fred_close(raw/"FRED_VXVCLS.csv", "VXVCLS", "vix3m")
        macro_names = list(FRED_SERIES)
    elif set(YAHOO_SERIES) <= set(sources):
        a = yahoo_close(raw/"Yahoo_VIX.json", "^VIX", "vix")
        b = yahoo_close(raw/"Yahoo_VIX3M.json", "^VIX3M", "vix3m")
        macro_names = list(YAHOO_SERIES)
    if macro_names:
        common = a.merge(b, on="date", how="outer", validate="one_to_one").sort_values("date")
        first, last = max(a.dropna().date.min(), b.dropna().date.min()), min(a.dropna().date.max(), b.dropna().date.max())
        common = common.loc[common.date.between(first, last)]
        no_quotes = common[["vix", "vix3m"]].isna().all(axis=1)
        atomic_csv(common.loc[no_quotes], normalized / "macro_dates_without_quotes.csv", index=False)
        common = common.loc[~no_quotes]
        # Keep and report gaps in paired source files; do not fabricate a price.
        missing = common[common.isna().any(axis=1)]
        atomic_csv(missing, normalized / "macro_unpaired_dates.csv", index=False)
        if not missing.empty:
            raise ValueError("Cboe series have interior unpaired observations; see macro_unpaired_dates.csv.")
        save_series(common, "macro_volatility", "index_points", macro_names)
    else:
        statuses.append(dict(group="macro_volatility", status="download_failed", rows=0, strict_history_ready=False))
    if "F-F_Research_Data_Factors_daily_CSV.zip" in sources:
        save_series(french_daily(raw/"F-F_Research_Data_Factors_daily_CSV.zip"), "ff3_daily", "percent_return",
            ["F-F_Research_Data_Factors_daily_CSV.zip"])
    else:
        statuses.append(dict(group="ff3_daily", status="download_failed", rows=0, strict_history_ready=False))

    records, evidence = [], []
    for year, stamp in FOMC.items():
        name = f"fomc_{year}_{stamp}.html"
        if name in sources:
            rows = fomc_records(year, raw/name, sources[name]["source"])
            records.extend(rows)
            evidence.append(dict(year=year, rows=len(rows), source=sources[name]["source"],
                announcement_date=datetime.strptime(stamp[:8], "%Y%m%d").date().isoformat(),
                raw_hashes=file_hashes([raw/name], output.resolve())))
    columns = ["event_id", "event_type", "ticker", "scheduled_at", "known_at", "status", "source", "vintage_id"]
    events = pd.DataFrame(records, columns=columns).sort_values(["known_at", "event_id"])
    atomic_csv(events, normalized / "events.csv", index=False)
    # Evidence for schedule rows is real; completeness/cancellations and earnings
    # coverage have not been established. An empty coverage table keeps it blocked.
    atomic_csv(pd.DataFrame(columns=["ticker", "event_type", "covered_from", "covered_through", "known_at", "source", "vintage_id"]),
               normalized / "event_coverage.csv", index=False)
    atomic_json(normalized / "events.json", dict(availability="point_in_time",
        availability_evidence=evidence, completeness="partial_original_regular_FOMC_schedules_only",
        known_at_precision="conservative_end_of_announcement_day_New_York",
        scheduled_at_precision="meeting_end_date_only_23:59:59_is_not_actual_event_hour",
        coverage_status="unknown_no_assertions_written", earnings_status="needs_PIT_vendor_or_archived_announcements",
        revisions_status="subsequent_changes_cancellations_and_unscheduled_meetings_not_collected",
        use="source_records_only_not_ready_for_blackout_backtest"))
    statuses.append(dict(group="blackout", status="partial_fomc_no_earnings_or_coverage", rows=len(events), strict_history_ready=False))
    atomic_csv(pd.DataFrame(statuses), output / "data_inventory.csv", index=False)
    return statuses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--offline", action="store_true", help="Normalize already downloaded official files in output-dir/raw.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry missing sources in an existing download snapshot; keep successful originals.")
    parser.add_argument("--assets-config", type=Path, default=ROOT / "configs/assets.yaml")
    parser.add_argument("--vix-source", choices=["cboe", "fred", "yahoo"], default="cboe",
                        help="Explicit source: Cboe, FRED VIXCLS/VXVCLS, or Yahoo ^VIX/^VIX3M (2014-2022).")
    args = parser.parse_args(argv)
    if args.offline and args.retry_failed:
        parser.error("Choose --offline or --retry-failed.")
    if (args.offline or args.retry_failed) and args.output_dir is None:
        parser.error("Resuming acquisition requires an explicit --output-dir.")
    output = (args.output_dir or new_run_directory(ROOT / "data/external", "alpha_sources")).resolve()
    if args.offline:
        acquisition = json.loads((output / "acquisition.json").read_text(encoding="utf-8"))
    else:
        if output.exists() and any(output.iterdir()) and not args.retry_failed:
            raise ValueError("Download output exists; choose a fresh directory to retain source vintages.")
        raw = output / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        requests = dict(SERIES)
        if args.vix_source in {"fred", "yahoo"}:
            requests.pop("VIX_History.csv")
            requests.pop("VIX3M_History.csv")
            requests.update(FRED_SERIES if args.vix_source == "fred" else YAHOO_SERIES)
        requests.update({f"fomc_{year}_{stamp}.html": f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{stamp}.htm"
                         for year, stamp in FOMC.items()})
        acquisition = json.loads((output / "acquisition.json").read_text(encoding="utf-8")) if args.retry_failed else []
        completed = {item["name"] for item in acquisition if item["status"] == "downloaded"}
        requests = {name: url for name, url in requests.items() if name not in completed}
        if args.retry_failed:
            # Cboe now redirects its legacy cdn host to cdn-api. Try the official
            # target directly when the legacy host resets the TLS connection.
            requests = {name: url.replace("https://cdn.cboe.com/", "https://cdn-api.cboe.com/")
                        for name, url in requests.items()}
        with ThreadPoolExecutor(max_workers=3) as executor:
            pending = {executor.submit(fetch, url, raw/name): (name, url) for name, url in requests.items()}
            for future in as_completed(pending):
                name, url = pending[future]
                try:
                    item = dict(future.result(), name=name, status="downloaded")
                    print(f"Downloaded {name}: {item['bytes']} bytes", flush=True)
                except Exception as exc:
                    item = dict(name=name, source=url, status="failed", error=str(exc))
                    print(f"Download failed {name}: {exc}", flush=True)
                acquisition.append(item)
                atomic_json(output / "acquisition.json", acquisition)
    manifest = dict(status="normalizing", retrieved_sources=len([x for x in acquisition if x["status"] == "downloaded"]),
        data_root=str(output), no_historical_availability_invented=True, strict_history_ready=False)
    atomic_json(output / "manifest.json", manifest)
    try:
        statuses = materialize(output, acquisition)
        with (ROOT / "configs/alpha_features.yaml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config["vix"]["path"] = str(output / "normalized/macro_volatility.csv")
        config["fama_french"]["path"] = str(output / "normalized/ff3_daily.csv")
        config["blackout"].update(events_path=str(output / "normalized/events.csv"), coverage_path=str(output / "normalized/event_coverage.csv"))
        (output / "alpha_features_downloaded.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        # AssetConfig's public API supplies the same configured trading universe.
        assets = AssetsConfig(str(args.assets_config))
        tickers = assets.get_tickers()
        atomic_csv(pd.DataFrame([dict(ticker=ticker, status="needs_data", event_type="earnings",
            required="announced_schedule_known_at_revisions_and_coverage_not_just_realized_earnings_date")
            for ticker in tickers]), output / "earnings_requirements.csv", index=False)
        manifest.update(status="downloaded_partial_not_training_ready", inventory=statuses,
            source_hashes=file_hashes([Path(__file__), ROOT / "scripts/import_alpha_sources.py"], ROOT),
            artifact_hashes=file_hashes(list((output/"raw").glob("*")) + list((output/"normalized").glob("*")) +
                [output / "acquisition.json", output / "data_inventory.csv", output / "earnings_requirements.csv",
                 output / "alpha_features_downloaded.yaml"], output))
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        atomic_json(output / "manifest.json", manifest)
        raise
    atomic_json(output / "manifest.json", manifest)
    print(f"Source CSVs: {output}/normalized\nStrict historical training still requires PIT vintages/earnings coverage.", flush=True)


if __name__ == "__main__":
    main()
