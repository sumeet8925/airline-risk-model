#!/usr/bin/env python3
"""Flight risk model for CSE 6242 Team 185.

Getting started? Run this with Python 3.12.
    python model.py --setup

That sets up Python, grabs the 2022-2025 BTS flight data, and runs both models.
The first run downloads about 1.3 GB, so give it some time and disk space.
You'll find the detailed results in modelv4-work/artifacts.
In Visual.html, use Load JSON to pull in the two files from output/.

Already have the packages installed? Run the full process or pick a step.
    python model.py
    python model.py prepare --download
    python model.py run
    python model.py audit

Run python model.py --help for more options.
Treat the 2025 test results as exploratory.
"""
from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import copy
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import threading
import time
import types
import venv
import warnings
import zipfile
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve()
DEFAULT_WORK_DIR = SCRIPT.parent / 'modelv4-work'
DEFAULT_RUN = 'corrected-2025'
TEAM_UI_DIR = SCRIPT.parent / 'output'
DEPENDENCIES = ['numpy==2.1.1',
 'pandas==2.2.2',
 'pyarrow==20.0.0',
 'duckdb==1.5.6',
 'scikit-learn==1.8.0',
 'scipy==1.14.1',
 'joblib==1.4.2',
 'requests==2.32.3',
 'filelock==3.18.0',
 'airportsdata==20261103',
 'threadpoolctl==3.5.0']


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--setup', action='store_true',
                        help='Install the tested packages in a local virtual environment before running')
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument('--work-dir', type=Path, default=DEFAULT_WORK_DIR,
                       help='Location for downloads, prepared data, and artifacts')
    paths.add_argument('--config', type=Path,
                       help='Use a custom JSON configuration instead of the embedded defaults')
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('audit', help='Check the prepared flight data')
    prepare_parser = sub.add_parser('prepare', help='Ingest BTS archives and reconstruct rotations')
    prepare_parser.add_argument('--download', action='store_true',
                                help='Download any missing archives from BTS')
    run_parser = sub.add_parser('run', help='Train and evaluate both model groups and export results')
    run_parser.add_argument('--run', default=DEFAULT_RUN, help='Name for this run')
    run_parser.add_argument('--download', action='store_true',
                            help='Download and prepare data if any months are missing')
    run_parser.add_argument('--ui-dir', type=Path, default=TEAM_UI_DIR,
                            help='Directory for the JSON files read by Visual.html')
    export_parser = sub.add_parser('export-dashboard', help='Refresh Visual.html JSON from a completed run')
    export_parser.add_argument('--run', default=DEFAULT_RUN, help='Name of a completed run')
    export_parser.add_argument('--ui-dir', type=Path, default=TEAM_UI_DIR,
                               help='Directory for the JSON files read by Visual.html')
    predict_parser = sub.add_parser('predict', help='Score a CSV using a trusted saved model bundle')
    for name in ('bundle', 'input', 'output'):
        predict_parser.add_argument('--' + name, type=Path, required=True)
    return parser


def use_local_environment(args, cli_arguments):
    """Install only when requested and run in the same environment on later calls."""
    env_dir = (args.work_dir.expanduser().resolve() / '.venv')
    interpreter = env_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if args.setup:
        if not interpreter.is_file():
            print(f'Creating Python environment in {env_dir}', flush=True)
            venv.EnvBuilder(with_pip=True).create(env_dir)
        print('Installing the flight model packages', flush=True)
        subprocess.run([str(interpreter), '-m', 'pip', 'install', *DEPENDENCIES], check=True)
    if interpreter.is_file() and Path(sys.prefix).resolve() != env_dir:
        forwarded = [arg for arg in cli_arguments if arg != '--setup']
        child_env = os.environ.copy()
        child_env.pop('PYTHONHOME', None)
        raise SystemExit(subprocess.call([str(interpreter), str(SCRIPT), *forwarded], env=child_env))


if __name__ == '__main__':
    CLI_ARGS = build_parser().parse_args()
    use_local_environment(CLI_ARGS, sys.argv[1:])
    sys.modules['modelv4'] = sys.modules[__name__]


try:
    import airportsdata
    import duckdb
    import joblib
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests
    from filelock import FileLock
    from requests.adapters import HTTPAdapter
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln
    from scipy.stats import beta, betabinom
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.frozen import FrozenEstimator
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score, classification_report, confusion_matrix,
        log_loss, precision_recall_curve, roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from threadpoolctl import threadpool_limits
    from urllib3.util.retry import Retry
except ImportError as exc:
    if __name__ == '__main__':
        raise SystemExit(
            f'Python dependency unavailable: {exc}\n'
            'Run python model.py --setup to install the tested packages.'
        ) from None
    raise


DEFAULT_CONFIG = {'seed': 6242,
 'data_dir': 'prepared',
 'artifacts_dir': 'artifacts',
 'years': [2022, 2023, 2024, 2025],
 'threads': 4,
 'duckdb_memory_limit': '1GB',
 'score_batch_rows': 50000,
 'initial_train_rows': 100000,
 'validation_rows': 150000,
 'final_train_rows': 500000,
 'calibration_selection_rows': 150000,
 'calibration_rows': 300000,
 'min_route_frequency': 50,
 'logistic_candidates': [0.1, 1.0, 10.0],
 'forest_candidates': [{'n_estimators': 80,
                        'max_depth': 16,
                        'min_samples_leaf': 50,
                        'max_features': 'sqrt',
                        'max_leaf_nodes': 10000,
                        'max_samples': 100000},
                       {'n_estimators': 80,
                        'max_depth': 20,
                        'min_samples_leaf': 25,
                        'max_features': 0.1,
                        'max_leaf_nodes': 10000,
                        'max_samples': 100000}],
 'final_forest_trees': 150,
 'logistic_max_iter': 1000,
 'logistic_tol': 0.001,
 'hierarchy_levels': ['global',
                      'route',
                      'route+carrier',
                      'route+carrier+season',
                      'route+carrier+season+month+time_bucket'],
 'rotation_bucket_edges_minutes': [0, 40, 90, 180, 1440],
 'time_buckets': {'night': [21, 22, 23, 0, 1, 2, 3, 4],
                  'morning': [5, 6, 7, 8, 9, 10, 11],
                  'afternoon': [12, 13, 14, 15, 16],
                  'evening': [17, 18, 19, 20]},
 'test_interpretation': 'Exploratory reuse of previously inspected 2025. Tail-linked population is '
                        'retrospective and selected.',
 'splits': {'initial_fit': ['2023-01-01', '2023-12-31'],
            'validation': ['2024-01-01', '2024-06-30'],
            'final_fit': ['2023-01-01', '2024-06-30'],
            'calibration_fit': ['2024-07-01', '2024-09-30'],
            'calibration_selection': ['2024-10-01', '2024-12-31'],
            'calibration': ['2024-07-01', '2024-12-31'],
            'test': ['2025-01-01', '2025-12-31']},
 'hierarchy_history_period': ['2022-01-01', '2022-12-31'],
 'reference_implementation': 'modelv3.py',
 'design_notes': ['Dedicated 2022 hierarchy feature history prevents own-target encoding leakage '
                  'in 2023 fitting.',
                  'Joint Dirichlet pooling uses Beta marginal intervals and coherent four-outcome '
                  'probabilities.',
                  'No actual previous arrival delay or cancellation is used as a booking-time '
                  'input.',
                  'Rotation comparison is retrospective, limited to linked-tail population.'],
 'pooling_display_period': ['2024-01-01', '2024-12-31'],
 'raw_data_dir': 'bts',
 'csv_chunk_rows': 100000,
 'require_matched_rotation': True,
 'implementation_revision': 'modelv4-2026-11-12'}

SOURCE_PROVENANCE = {'source': 'modelv3.py',
 'source_sha256': '00cc53bbccaae37a7fb523c4a558fe89cd25f99ca51d6b18f8ed0ab0214dacbd',
 'revision': '2026-11-12',
 'support_modules_reused_from': '../flight-risk/src/flight_risk',
 'note': 'Original is archived unchanged. v4 refactors the teammate design using reviewed shared '
         'utilities plus new two-population evaluation and exports.',
 'distribution': 'Single-file v4 with embedded configuration'}


CLASS_NAMES = ["on_time", "delayed", "cancelled", "diverted"]
CAT_FEATURES = ["origin_id", "destination_id", "route_id", "carrier_id",
                "month", "departure_hour", "weekday"]
NUM_FEATURES = ["distance", "scheduled_duration"]
FEATURES = CAT_FEATURES + NUM_FEATURES


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def load_config(path):
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    result["_config_path"] = str(path)
    for key in ("data_dir", "artifacts_dir"):
        result[key] = str((path.parent / result[key]).resolve())
    validate_splits(result["splits"])
    return result


def validate_splits(splits):
    for name, (start, end) in splits.items():
        if start > end:
            raise ValueError(f"Invalid split {name}: {start} > {end}")
    ordered = ["initial_fit", "validation", "calibration_fit",
               "calibration_selection", "test"]
    for previous, following in zip(ordered, ordered[1:]):
        if splits[previous][1] >= splits[following][0]:
            raise ValueError(f"Overlapping chronological splits: {previous}, {following}")
    if splits["final_fit"][1] >= splits["calibration"][0]:
        raise ValueError("Final fitting must end before calibration begins")
    if splits["calibration"][1] >= splits["test"][0]:
        raise ValueError("Calibration must end before the test begins")
    for child in ("calibration_fit", "calibration_selection"):
        if not (splits["calibration"][0] <= splits[child][0]
                <= splits[child][1] <= splits["calibration"][1]):
            raise ValueError(f"{child} is outside the calibration period")
    for child in ("initial_fit", "validation"):
        if not (splits["final_fit"][0] <= splits[child][0]
                <= splits[child][1] <= splits["final_fit"][1]):
            raise ValueError(f"{child} is outside the final fitting period")


def versions():
    import importlib.metadata
    names = ["duckdb", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "joblib"]
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in names}}


BASE_URL = "https://transtats.bts.gov/PREZIP/"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


class DownloadIntegrityError(ValueError):
    """An incomplete transfer must be retried before accepting an archive."""


def filename(year, month):
    return f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        csvs = [info for info in archive.infolist() if info.filename.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {len(csvs)}")
        if csvs[0].file_size > 2 * 1024 ** 3:
            raise ValueError(f"Unreasonably large uncompressed CSV: {path}")
        bad = archive.testzip()
        if bad:
            raise DownloadIntegrityError(f"ZIP checksum failure in {bad}")
        return csvs[0].filename


def fetch_month(data_dir, year, month):
    raw = Path(data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lock_path = raw / (filename(year, month) + ".lock")
    with FileLock(str(lock_path), timeout=7200):
        for attempt in range(3):
            try:
                return _fetch_month(data_dir, year, month)
            except (requests.RequestException, zipfile.BadZipFile, zlib.error,
                    EOFError, DownloadIntegrityError):
                if attempt == 2:
                    raise
                print(f"Retrying verified download {year}-{month:02d}", flush=True)
                time.sleep(2 ** attempt)


def _fetch_month(data_dir, year, month):
    raw = Path(data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / filename(year, month)
    manifest = path.with_suffix(".manifest.json")
    if path.exists() and manifest.exists():
        info = json.loads(manifest.read_text())
        if info.get("sha256") == sha256(path):
            print(f"Cached {year}-{month:02d}", flush=True)
            return info
        raise ValueError(f"Cached archive changed: {path}; inspect before replacing")
    if path.exists():
        csv_name = verify_zip(path)
    else:
        session = requests.Session()
        retry = Retry(total=4, backoff_factor=1,
                      status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        part = raw / (path.name + f".{os.getpid()}.{threading.get_ident()}.part")
        started = time.monotonic()
        with session.get(BASE_URL + path.name, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            expected = int(response.headers.get("Content-Length", 0))
            if expected > MAX_ARCHIVE_BYTES:
                raise ValueError(f"Archive exceeds the monthly download limit: {expected}")
            size = 0
            with part.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise ValueError("Download exceeded the monthly archive limit")
                    output.write(chunk)
            if expected and size != expected:
                raise DownloadIntegrityError(f"Incomplete download: {size} != {expected}")
        csv_name = verify_zip(part)
        os.replace(part, path)
        print(f"Downloaded {year}-{month:02d}: {size / 1e6:.1f} MB "
              f"in {time.monotonic() - started:.1f}s", flush=True)
    info = {"year": year, "month": month, "url": BASE_URL + path.name,
            "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "csv_member": csv_name, "downloaded_at": utcnow()}
    write_json(manifest, info)
    return info


def download(data_dir, years, months=range(1, 13), workers=2):
    jobs = [(year, month) for year in years for month in months]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_month, data_dir, y, m) for y, m in jobs]
        return [future.result() for future in futures]


PIPELINE_VERSION = "1"
COLUMNS = [
    "FlightDate", "Year", "Month", "DayOfWeek", "Reporting_Airline",
    "DOT_ID_Reporting_Airline", "Flight_Number_Reporting_Airline",
    "OriginAirportID", "Origin", "DestAirportID", "Dest",
    "CRSDepTime", "Distance", "CRSElapsedTime", "Cancelled", "Diverted",
    "ArrDel15", "ArrDelay",
]
REQUIRED = {"FlightDate", "DOT_ID_Reporting_Airline", "OriginAirportID",
            "DestAirportID", "CRSDepTime", "Cancelled", "Diverted"}
SCHEMA = pa.schema([
    ("flight_date", pa.date32()), ("year", pa.int16()), ("month", pa.int8()),
    ("weekday", pa.int8()), ("departure_hour", pa.int8()),
    ("scheduled_minutes", pa.int16()),
    ("origin_id", pa.string()), ("destination_id", pa.string()),
    ("route_id", pa.string()), ("carrier_id", pa.string()),
    ("origin_code", pa.string()), ("destination_code", pa.string()),
    ("carrier_code", pa.string()), ("flight_number", pa.string()),
    ("distance", pa.float32()), ("scheduled_duration", pa.float32()),
    ("outcome", pa.int8()), ("raw_cancelled", pa.int8()),
    ("raw_diverted", pa.int8()), ("source_file", pa.string()),
    ("source_row", pa.int64()),
])


def prepare_chunk(raw, source_file, row_offset=0, expected_year=None, expected_month=None):
    raw = raw.copy()
    raw.columns = [str(c).strip().lstrip("\ufeff") for c in raw.columns]
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not ({"ArrDel15", "ArrDelay"} & set(raw.columns)):
        raise ValueError("Neither ArrDel15 nor ArrDelay is available")
    for name in COLUMNS:
        if name not in raw:
            raw[name] = pd.NA
    idx = raw.index
    reasons = pd.Series("", index=idx, dtype="string")
    audit = Counter()

    def reject(mask, reason):
        mask = mask.fillna(True)
        audit[reason] += int(mask.sum())
        reasons.loc[mask] = reasons.loc[mask] + reason + ";"

    numeric = lambda c: pd.to_numeric(raw[c], errors="coerce")
    date = pd.to_datetime(raw["FlightDate"], errors="coerce", format="mixed")
    reject(date.isna(), "invalid_date")
    if expected_year is not None:
        reject(date.dt.year.ne(expected_year), "wrong_year")
    if expected_month is not None:
        reject(date.dt.month.ne(expected_month), "wrong_month")
    identifiers = {}
    for field, output in [("OriginAirportID", "origin_id"),
                          ("DestAirportID", "destination_id"),
                          ("DOT_ID_Reporting_Airline", "carrier_id")]:
        values = numeric(field)
        valid = values.notna() & values.gt(0) & values.mod(1).eq(0)
        reject(~valid, f"invalid_{output}")
        identifiers[output] = values.where(valid).astype("Int64").astype("string")

    hhmm = numeric("CRSDepTime")
    valid_time = (hhmm.notna() & hhmm.mod(1).eq(0)
                  & (((hhmm >= 0) & (hhmm <= 2359) & (hhmm.mod(100) < 60))
                     | hhmm.eq(2400)))
    reject(~valid_time, "invalid_scheduled_time")
    audit["scheduled_2400"] += int(hhmm.eq(2400).sum())
    hour = np.floor(hhmm / 100).mod(24)
    minute = hhmm.mod(100)

    cancelled, diverted = numeric("Cancelled"), numeric("Diverted")
    reject(~cancelled.isin([0, 1]), "invalid_cancellation_flag")
    reject(~diverted.isin([0, 1]), "invalid_diversion_flag")
    audit["both_cancelled_and_diverted"] += int((cancelled.eq(1) & diverted.eq(1)).sum())
    delay_flag, delay_minutes = numeric("ArrDel15"), numeric("ArrDelay")
    ordinary = cancelled.eq(0) & diverted.eq(0)
    conflict = ordinary & delay_flag.isin([0, 1]) & delay_minutes.notna() \
        & delay_flag.ne(delay_minutes.ge(15).astype("Int64"))
    reject(conflict, "conflicting_arrival_outcome")
    reject(ordinary & delay_flag.notna() & ~delay_flag.isin([0, 1]),
           "invalid_arrival_indicator")
    missing_arrival = ordinary & delay_flag.isna() & delay_minutes.isna()
    reject(missing_arrival, "missing_arrival_outcome")
    audit["derived_arrival_indicator"] += int(
        (ordinary & delay_flag.isna() & delay_minutes.notna()).sum())
    delay = delay_flag.where(delay_flag.notna(), delay_minutes.ge(15).astype("Int64"))

    outcome = np.select(
        [condition.to_numpy(dtype=bool, na_value=False) for condition in
         [cancelled.eq(1), diverted.eq(1), delay.eq(1)]],
        [2, 3, 1], default=0).astype("int8")
    valid = reasons.eq("")
    clean = pd.DataFrame({
        "flight_date": date.dt.date, "year": date.dt.year,
        "month": date.dt.month, "weekday": date.dt.dayofweek + 1,
        "departure_hour": hour, "scheduled_minutes": hour * 60 + minute,
        **identifiers,
        "route_id": identifiers["origin_id"] + "_" + identifiers["destination_id"],
        "origin_code": raw["Origin"].astype("string").fillna(""),
        "destination_code": raw["Dest"].astype("string").fillna(""),
        "carrier_code": raw["Reporting_Airline"].astype("string").fillna(""),
        "flight_number": raw["Flight_Number_Reporting_Airline"].astype("string").fillna(""),
        "distance": numeric("Distance").where(numeric("Distance").gt(0)),
        "scheduled_duration": numeric("CRSElapsedTime").where(numeric("CRSElapsedTime").gt(0)),
        "outcome": outcome, "raw_cancelled": cancelled, "raw_diverted": diverted,
        "source_file": source_file,
        "source_row": np.arange(row_offset, row_offset + len(raw), dtype="int64"),
    }, index=idx).loc[valid].copy()
    for name, dtype in [("year", "int16"), ("month", "int8"), ("weekday", "int8"),
                        ("departure_hour", "int8"), ("scheduled_minutes", "int16"),
                        ("raw_cancelled", "int8"), ("raw_diverted", "int8")]:
        clean[name] = clean[name].astype(dtype)
    clean[["distance", "scheduled_duration"]] = clean[
        ["distance", "scheduled_duration"]].astype("float32")
    rejected = raw.loc[~valid, COLUMNS].astype("string").copy()
    rejected["rejection_reasons"] = reasons.loc[~valid]
    rejected["source_row"] = np.arange(row_offset, row_offset + len(raw))[
        ~valid.to_numpy(dtype=bool)]
    audit.update({f"outcome_{CLASS_NAMES[int(k)]}": int(v)
                  for k, v in clean["outcome"].value_counts().items()})
    audit["input_rows"] = len(raw)
    audit["accepted_rows"] = len(clean)
    audit["rejected_rows"] = len(rejected)
    return clean, rejected, audit


def ingest_archive(path, data_dir, chunk_rows=100000, force=False):
    path, data_dir = Path(path), Path(data_dir)
    pieces = path.stem.split("_")
    year, month = int(pieces[-2]), int(pieces[-1])
    destination = data_dir / "clean" / f"year={year}" / f"month={month:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    parquet_path, report_path = destination / "data.parquet", destination / "quality.json"
    digest = sha256(path)
    if report_path.exists() and parquet_path.exists() and not force:
        report = json.loads(report_path.read_text())
        if report["source_sha256"] != digest or report["pipeline_version"] != PIPELINE_VERSION:
            raise ValueError(f"Source or pipeline changed; explicitly re-ingest {path.name}")
        print(f"Ingest cached {year}-{month:02d}", flush=True)
        return report
    temp = parquet_path.with_suffix(".parquet.tmp")
    rejected_path = destination / "quarantine.parquet"
    rejected_temp = rejected_path.with_suffix(".parquet.tmp")
    reject_schema = pa.schema([(c, pa.string()) for c in COLUMNS]
                             + [("rejection_reasons", pa.string()), ("source_row", pa.int64())])
    counts, offset = Counter(), 0
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {path}")
        with archive.open(members[0]) as stream, \
                pq.ParquetWriter(temp, SCHEMA, compression="zstd") as writer, \
                pq.ParquetWriter(rejected_temp, reject_schema, compression="zstd") as rejected_writer:
            reader = pd.read_csv(
                stream, usecols=lambda c: str(c).strip().lstrip("\ufeff") in COLUMNS,
                dtype="string", chunksize=chunk_rows, encoding="utf-8-sig")
            for raw in reader:
                clean, rejected, audit = prepare_chunk(raw, path.name, offset, year, month)
                writer.write_table(pa.Table.from_pandas(clean, schema=SCHEMA, preserve_index=False))
                rejected_writer.write_table(pa.Table.from_pandas(
                    rejected, schema=reject_schema, preserve_index=False))
                counts.update(audit)
                offset += len(raw)
    os.replace(temp, parquet_path)
    os.replace(rejected_temp, rejected_path)
    report = {"pipeline_version": PIPELINE_VERSION, "source_file": path.name,
              "source_sha256": digest, "year": year, "month": month,
              "processed_at": utcnow(), "counts": dict(counts),
              "rejection_counts_may_overlap": True,
              "clean_parquet_bytes": parquet_path.stat().st_size}
    if counts["input_rows"] != counts["accepted_rows"] + counts["rejected_rows"]:
        raise AssertionError("Ingestion row reconciliation failed")
    write_json(report_path, report)
    print(f"Ingested {year}-{month:02d}: {counts['accepted_rows']:,} accepted, "
          f"{counts['rejected_rows']:,} quarantined", flush=True)
    return report


def ingest(data_dir, years, months=range(1, 13), chunk_rows=100000, force=False):
    reports = []
    for year in years:
        for month in months:
            path = Path(data_dir) / "raw" / filename(year, month)
            if not path.exists():
                raise FileNotFoundError(f"Download the source first: {path}")
            reports.append(ingest_archive(path, data_dir, chunk_rows, force))
    write_json(Path(data_dir) / "ingestion-manifest.json", {
        "pipeline_version": PIPELINE_VERSION, "created_at": utcnow(), "months": reports,
        "totals": {key: sum(r["counts"].get(key, 0) for r in reports)
                   for key in ["input_rows", "accepted_rows", "rejected_rows"]},
    })
    return reports


RAW = ['Tail_Number', 'FlightDate', 'Origin', 'Dest', 'CRSDepTime',
       'CRSArrTime', 'CRSElapsedTime']


def airport_timezones():
    data = airportsdata.load('ICAO')
    result = {}
    for airport in data.values():
        if airport.get('tz'):
            for code in [airport.get('iata'), airport.get('lid')]:
                if code:
                    result[code] = airport['tz']
    result['PBI'] = 'America/New_York'
    return result


def utc_departures(dates, hhmm, codes, zones):
    times = pd.to_numeric(hhmm, errors='coerce')
    valid = times.notna() & times.mod(1).eq(0) & (
        (times.between(0, 2359) & times.mod(100).lt(60)) | times.eq(2400))
    minutes = np.floor(times / 100) * 60 + times.mod(100)
    naive = pd.to_datetime(dates, errors='coerce') + pd.to_timedelta(minutes, unit='m')
    output = pd.Series(pd.NaT, index=dates.index, dtype='datetime64[ns, UTC]')
    mapped = codes.map(zones)
    for zone in mapped.dropna().unique():
        mask = mapped.eq(zone) & valid
        output.loc[mask] = naive.loc[mask].dt.tz_localize(
            zone, ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    return output


def extract_month(path, output, zones):
    path, output = Path(path), Path(output)
    parquet = output / 'legs' / (path.stem + '.parquet')
    manifest = parquet.with_suffix('.json')
    parquet.parent.mkdir(parents=True, exist_ok=True)
    if parquet.exists() and manifest.exists():
        record = json.loads(manifest.read_text())
        if record['source_sha256'] == sha256(path):
            return record
        raise ValueError(f'Source changed: {path.name}')
    schema = pa.schema([
        ('source_file', pa.string()), ('source_row', pa.int64()),
        ('tail_number', pa.string()), ('origin_code', pa.string()),
        ('destination_code', pa.string()),
        ('scheduled_departure_utc', pa.timestamp('ns', tz='UTC')),
        ('scheduled_arrival_utc', pa.timestamp('ns', tz='UTC')),
        ('arrival_clock_consistent', pa.bool_()),
    ])
    temporary = parquet.with_suffix('.tmp')
    rows, missing_zones, offset = 0, set(), 0
    with zipfile.ZipFile(path) as archive, pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
        member = next(n for n in archive.namelist() if n.lower().endswith('.csv'))
        with archive.open(member) as stream:
            for raw in pd.read_csv(stream, usecols=RAW, dtype='string', chunksize=100000):
                raw = raw.reset_index(drop=True)
                departure = utc_departures(raw.FlightDate, raw.CRSDepTime, raw.Origin, zones)
                duration = pd.to_numeric(raw.CRSElapsedTime, errors='coerce')
                duration = duration.where(duration.gt(0) & duration.le(1440))
                arrival = departure + pd.to_timedelta(duration, unit='m')
                arrival_hhmm = pd.to_numeric(raw.CRSArrTime, errors='coerce')
                clock_valid = arrival_hhmm.notna() & arrival_hhmm.mod(1).eq(0) & (
                    (arrival_hhmm.between(0, 2359) & arrival_hhmm.mod(100).lt(60)) | arrival_hhmm.eq(2400))
                arrival_minutes = (np.floor(arrival_hhmm / 100) * 60 + arrival_hhmm.mod(100)).mod(1440)
                consistent = pd.Series(False, index=raw.index)
                dest_zones = raw.Dest.map(zones)
                for zone in dest_zones.dropna().unique():
                    mask = dest_zones.eq(zone)
                    local = arrival.loc[mask].dt.tz_convert(zone)
                    difference = (local.dt.hour * 60 + local.dt.minute - arrival_minutes.loc[mask]).abs()
                    consistent.loc[mask] = (clock_valid.loc[mask] & np.minimum(difference, 1440 - difference).le(5)).fillna(False).to_numpy(dtype=bool)
                tail = raw.Tail_Number.str.strip().str.upper()
                valid_tail = tail.str.fullmatch(r'[A-Z0-9-]{3,12}', na=False) & ~tail.isin(['UNKNOWN', 'UNKNOW', 'NONE', 'NULL', '000000', 'N00000'])
                tail = tail.where(valid_tail)
                frame = pd.DataFrame({
                    'source_file': path.name, 'source_row': np.arange(offset, offset + len(raw)),
                    'tail_number': tail, 'origin_code': raw.Origin, 'destination_code': raw.Dest,
                    'scheduled_departure_utc': departure, 'scheduled_arrival_utc': arrival,
                    'arrival_clock_consistent': consistent,
                })
                writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
                missing_zones.update(raw.loc[raw.Origin.map(zones).isna(), 'Origin'].dropna())
                missing_zones.update(raw.loc[dest_zones.isna(), 'Dest'].dropna())
                rows += len(raw)
                offset += len(raw)
    temporary.replace(parquet)
    record = {'source_file': path.name, 'source_sha256': sha256(path),
              'rows': rows, 'missing_timezone_codes': sorted(missing_zones)}
    write_json(manifest, record)
    print(f'Rotation source {path.stem[-7:]}: {rows:,} rows', flush=True)
    return record


def rotation_query(legs):
    """Pure scheduled-time SQL, also exercised against adversarial test fixtures."""
    return f"""
    WITH ordered AS (
        SELECT *, count(*) OVER(PARTITION BY tail_number, scheduled_departure_utc) same_departures
        FROM {legs}
    ), previous AS (
        SELECT *, lag(destination_code) OVER w previous_destination,
          lag(scheduled_arrival_utc) OVER w previous_arrival,
          lag(arrival_clock_consistent) OVER w previous_clock_consistent,
          lag(same_departures) OVER w previous_same_departures
        FROM ordered
        WINDOW w AS (PARTITION BY tail_number ORDER BY scheduled_departure_utc, source_file, source_row)
    ), gaps AS (
        SELECT *, date_diff('second', previous_arrival, scheduled_departure_utc)/60.0 gap
        FROM previous
    ), classified AS (
        SELECT *, CASE
          WHEN tail_number IS NULL THEN 'missing_tail'
          WHEN scheduled_departure_utc IS NULL THEN 'unavailable_departure_time'
          WHEN same_departures>1 OR previous_same_departures>1 THEN 'ambiguous_tail_sequence'
          WHEN previous_arrival IS NULL THEN 'no_usable_previous_arrival'
          WHEN NOT previous_clock_consistent THEN 'previous_arrival_clock_mismatch'
          WHEN previous_destination<>origin_code THEN 'airport_discontinuity'
          WHEN gap<0 THEN 'overlapping_scheduled_legs'
          WHEN gap>1440 THEN 'gap_over_24_hours'
          ELSE 'matched' END rotation_status
        FROM gaps
    ) SELECT source_file, source_row, tail_number,
      scheduled_departure_utc, scheduled_arrival_utc, rotation_status,
      CASE WHEN rotation_status='matched' THEN gap END rotation_minutes,
      CASE WHEN rotation_status<>'matched' THEN 'unknown'
        WHEN gap<40 THEN 'lt40' WHEN gap<90 THEN '40_89'
        WHEN gap<180 THEN '90_179' ELSE '180_1440' END rotation_bucket
    FROM classified
    """


def build_rotations(config, output, workers=4):
    source, output = Path(config['data_dir']), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    zones = airport_timezones()
    paths = sorted((source / 'raw').glob('*.zip'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda p: extract_month(p, output, zones), paths))
    write_json(output / 'source-manifest.json', {'created_at': utcnow(), 'sources': records,
        'airportsdata_version': importlib.metadata.version('airportsdata'),
        'timezone_aliases': {'PBI': 'America/New_York'}})
    db = duckdb.connect()
    db.execute('SET threads=4')
    db.execute("SET memory_limit='2GB'")
    spill = output / 'spill'; spill.mkdir(exist_ok=True)
    db.execute('SET temp_directory=?', [str(spill.resolve())])
    try:
        db.from_parquet(str(output / 'legs' / '*.parquet')).create_view('legs')
        target = output / 'rotations.parquet'
        if not target.exists():
            temporary = target.with_suffix('.tmp')
            query = rotation_query('legs') + ' ORDER BY source_file, source_row'
            db.execute(f"COPY ({query}) TO '{str(temporary).replace(chr(39),chr(39)*2)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            temporary.replace(target)
        db.from_parquet(str(target)).create_view('rotations')
        for path in sorted((source / 'clean').glob('year=*/month=*/data.parquet')):
            destination = output / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            db.from_parquet(str(path)).create_view('month_clean', replace=True)
            query = """SELECT c.*, r.tail_number, r.scheduled_departure_utc,
                r.scheduled_arrival_utc, r.rotation_minutes, r.rotation_bucket,
                r.rotation_status, CASE WHEN c.departure_hour BETWEEN 5 AND 11 THEN 'morning'
                WHEN c.departure_hour BETWEEN 12 AND 16 THEN 'afternoon'
                WHEN c.departure_hour BETWEEN 17 AND 20 THEN 'evening' ELSE 'night' END time_bucket,
                CASE WHEN c.month IN (12,1,2) THEN 'winter'
                WHEN c.month IN (3,4,5) THEN 'spring' WHEN c.month IN (6,7,8) THEN 'summer'
                ELSE 'fall' END season
                FROM month_clean c JOIN rotations r USING(source_file,source_row)"""
            temporary = destination.with_suffix('.tmp')
            db.execute(f"COPY ({query}) TO '{str(temporary).replace(chr(39),chr(39)*2)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            temporary.replace(destination)
        db.from_parquet(str(output / 'clean/year=*/month=*/data.parquet'), hive_partitioning=False).create_view('features')
        quality = db.sql('SELECT year,outcome,rotation_status,count(*) n FROM features GROUP BY ALL ORDER BY ALL').fetchdf()
        summary = db.sql("""SELECT year,count(*) n,sum(rotation_status='matched') matched_rows,
            sum(outcome=2) cancellations,sum(outcome=2 AND rotation_status='matched') matched_cancellations
            FROM features GROUP BY year ORDER BY year""").fetchdf()
        contrast = db.sql("""SELECT rotation_bucket,count(*) n,sum(outcome=1) delayed,
            avg(CAST(outcome=1 AS INTEGER)) p_delayed FROM features
            WHERE year=2025 AND time_bucket='evening' AND rotation_status='matched'
            GROUP BY rotation_bucket ORDER BY rotation_bucket""").fetchdf()
        audit = {'created_at': utcnow(), 'row_count': int(summary.n.sum()),
                 'coverage_by_year': summary.to_dict('records'), 'quality': quality.to_dict('records'),
                 'evening_2025_observed_comparison': contrast.to_dict('records'),
                 'interpretation': 'Retrospective tail-assignment linkage; no actual arrival times or current outcomes enter rotation calculation.'}
        write_json(output / 'rotation-audit.json', audit)
        print(json.dumps({k:v for k,v in audit.items() if k!='quality'}, indent=2), flush=True)
        return audit
    finally:
        db.close()


def connect(config):
    connection = duckdb.connect()
    connection.execute("SET threads = ?", [int(config["threads"])])
    connection.execute("SET memory_limit = ?", [config["duckdb_memory_limit"]])
    temporary = Path(config["data_dir"]) / "duckdb-temporary"
    temporary.mkdir(parents=True, exist_ok=True)
    connection.execute("SET temp_directory = ?", [str(temporary)])
    return connection


def files(config):
    paths = sorted((Path(config["data_dir"]) / "clean").glob("year=*/month=*/data.parquet"))
    if not paths:
        raise FileNotFoundError("No cleaned data; run download and ingest first")
    return [str(path) for path in paths]


def normalise_features(frame):
    frame = frame.copy()
    for name in CAT_FEATURES:
        frame[name] = frame[name].astype(str)
    for name in ["distance", "scheduled_duration"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float32")
    return frame


def read_sample(config, period, limit, seed=None):
    connection = connect(config)
    seed = config["seed"] if seed is None else seed
    columns = ", ".join(FEATURES + ["outcome", "flight_date", "source_file", "source_row"])
    try:
        frame = connection.execute(
            f"SELECT {columns} FROM read_parquet(?, hive_partitioning=false) "
            "WHERE flight_date BETWEEN ?::DATE AND ?::DATE "
            "ORDER BY hash(source_file, source_row, ?) LIMIT ?",
            [files(config), *period, seed, int(limit)],
        ).fetchdf()
    finally:
        connection.close()
    if frame.empty:
        raise ValueError(f"No data for period {period}")
    return normalise_features(frame)


def iterate_period(config, period, columns=None):
    columns = columns or FEATURES + ["outcome", "flight_date", "origin_code",
                                     "destination_code", "carrier_code"]
    connection = connect(config)
    try:
        result = connection.execute(
            f"SELECT {', '.join(columns)} FROM read_parquet(?, hive_partitioning=false) "
            "WHERE flight_date BETWEEN ?::DATE AND ?::DATE",
            [files(config), *period],
        )
        reader = (result.to_arrow_reader(int(config["score_batch_rows"]))
                  if hasattr(result, "to_arrow_reader")
                  else result.fetch_record_batch(int(config["score_batch_rows"])))
        for batch in reader:
            frame = batch.to_pandas()
            if all(name in frame for name in FEATURES):
                frame = normalise_features(frame)
            yield frame
    finally:
        connection.close()


def coverage(config):
    connection = connect(config)
    try:
        by_month = connection.execute(
            "SELECT year, month, count(*) AS rows, "
            "sum(outcome=0) AS on_time, sum(outcome=1) AS delayed, "
            "sum(outcome=2) AS cancelled, sum(outcome=3) AS diverted "
            "FROM read_parquet(?, hive_partitioning=false) GROUP BY year, month "
            "ORDER BY year, month", [files(config)]).fetchdf()
        duplicates = connection.execute(
            "SELECT sum(n-1) AS possible_duplicate_rows FROM ("
            "SELECT count(*) n FROM read_parquet(?, hive_partitioning=false) "
            "GROUP BY flight_date, carrier_id, flight_number, origin_id, "
            "destination_id, scheduled_minutes HAVING count(*)>1)", [files(config)]
        ).fetchone()[0]
    finally:
        connection.close()
    return {"months": by_month.to_dict("records"),
            "possible_duplicate_rows": int(duplicates or 0),
            "duplicate_policy": "Flag only; no blind deduplication"}


def require_coverage(config):
    rows = coverage(config)
    actual = {(int(r["year"]), int(r["month"])) for r in rows["months"]}
    for name in ["initial_fit", "validation", "final_fit",
                 "calibration", "calibration_fit", "calibration_selection", "test"]:
        start, end = config["splits"][name]
        expected = {(d.year, d.month) for d in pd.date_range(
            start[:7] + "-01", end[:7] + "-01", freq="MS")}
        missing = expected - actual
        if missing:
            raise ValueError(f"Missing months for {name}: {sorted(missing)}")
    return rows


def estimate_concentration(counts, means):
    counts, means = np.asarray(counts, float), np.asarray(means, float)
    informative = counts.sum(axis=1) >= 2
    if informative.sum() < 2:
        return 50., {'status': 'insufficient_groups_default', 'groups': int(informative.sum())}
    counts, means = counts[informative], means[informative]
    means = np.clip(means, 1e-12, 1); means /= means.sum(axis=1, keepdims=True)
    n = counts.sum(axis=1)

    def objective(log_strength):
        strength = np.exp(log_strength)
        alpha = strength * means
        return -float((gammaln(strength) - gammaln(strength+n)
                      + (gammaln(alpha+counts)-gammaln(alpha)).sum(axis=1)).sum())
    result = minimize_scalar(objective, bounds=(np.log(.5), np.log(100000)), method='bounded')
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Empirical-Bayes concentration estimation failed')
    strength = float(np.exp(result.x))
    return strength, {'status': 'estimated', 'groups': len(counts),
                      'strength': strength, 'near_boundary': strength < .51 or strength > 99000}


@dataclass
class FourLevelPool:
    leaf_features: tuple = ('carrier_id', 'route_id', 'month', 'time_bucket', 'rotation_bucket')
    counts: list = field(default_factory=list)
    priors: list = field(default_factory=list)
    strengths: list = field(default_factory=list)
    fitting_report: list = field(default_factory=list)
    global_counts: np.ndarray | None = None
    grouping_levels: tuple | None = None

    @property
    def levels(self):
        return list(self.grouping_levels) if self.grouping_levels is not None else [('carrier_id',), ('carrier_id', 'route_id'), self.leaf_features]

    def fit(self, grouped):
        """Fit from a dataframe of leaf keys plus n0, n1, n2, n3."""
        self.counts, self.priors, self.strengths, self.fitting_report = [], [], [], []
        columns = ['n0', 'n1', 'n2', 'n3']
        if (grouped[columns].to_numpy() < 0).any():
            raise ValueError('Outcome counts must be nonnegative')
        self.global_counts = grouped[columns].sum().to_numpy(dtype=float)
        if not self.global_counts.sum():
            raise ValueError('No data for hierarchical pooling')
        global_alpha = self.global_counts + .5
        self.global_probabilities = global_alpha / global_alpha.sum()
        for level, dimensions in enumerate(self.levels):
            frame = grouped.groupby(list(dimensions), dropna=False, sort=True)[columns].sum().reset_index()
            level_counts, level_priors, all_counts, all_means = {}, {}, [], []
            for row in frame.itertuples(index=False, name=None):
                key = tuple(str(x) for x in row[:len(dimensions)])
                own = np.array(row[len(dimensions):], dtype=float)
                if level == 0:
                    prior_alpha = self.global_counts - own + .5
                else:
                    parent_dimensions = self.levels[level-1]
                    parent = tuple(key[list(dimensions).index(c)] for c in parent_dimensions)
                    remaining = np.maximum(0, self.counts[level-1][parent] - own)
                    prior_alpha = remaining + self.strengths[level-1] * self.priors[level-1][parent]
                mean = prior_alpha / prior_alpha.sum()
                level_counts[key], level_priors[key] = own, mean
                all_counts.append(own); all_means.append(mean)
            strength, report = estimate_concentration(all_counts, all_means)
            self.counts.append(level_counts); self.priors.append(level_priors)
            self.strengths.append(strength)
            self.fitting_report.append({'dimensions': list(dimensions), 'cells': len(frame), **report})
        return self

    def posterior(self, row):
        for level in range(len(self.levels)-1, -1, -1):
            key = tuple(str(row[c]) for c in self.levels[level])
            if key in self.counts[level]:
                counts = self.counts[level][key]
                alpha = counts + self.strengths[level] * self.priors[level][key]
                return alpha, int(counts.sum()), level+1
        return self.global_counts + .5, int(self.global_counts.sum()), 0

    def predict_proba(self, frame):
        keys = list(self.leaf_features)
        unique = frame[keys].astype(str).drop_duplicates()
        lookup = {}
        for row in unique.to_dict('records'):
            alpha, _, _ = self.posterior(row)
            lookup[tuple(row[c] for c in keys)] = alpha / alpha.sum()
        return np.array([lookup[tuple(str(v) for v in row)]
                         for row in frame[keys].itertuples(index=False, name=None)])

    def describe(self, row):
        row={c: str(row[c]) for c in self.leaf_features}
        alpha, n, level = self.posterior(row)
        total = alpha.sum()
        limits = beta.ppf(np.array([.025, .975])[:, None], alpha[None, :], total-alpha[None, :])
        return {'probabilities': (alpha/total).tolist(), 'intervals_95': limits.T.tolist(),
                'matching_fit_rows': n, 'hierarchy_level': level,
                'exact_leaf_fit_rows': int(self.counts[-1].get(tuple(row[c] for c in self.leaf_features),np.zeros(4)).sum()),
                'interval_scope': 'leaf_profile' if level==len(self.levels) else 'broader_fallback_group_only',
                'interval_kind': 'marginal_empirical_bayes_plugin_credible_interval',
                'caveat': 'Conditions on estimated pooling parameters; not calibrated RF uncertainty or a simultaneous four-outcome interval.'}


if __name__ == '__main__':
    FourLevelPool.__module__ = 'modelv4'


def make_pipeline(kind, parameters, config):
    categorical_features = config.get('categorical_features', CAT_FEATURES)
    numeric_features = config.get('numeric_features', NUM_FEATURES)
    numeric = [("impute", SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True))]
    if kind == "logistic_regression":
        numeric.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=int(config["min_route_frequency"]),
            sparse_output=True, dtype=np.float32), categorical_features),
        ("numeric", Pipeline(numeric), numeric_features),
    ], sparse_threshold=1.0)
    if kind == "random_forest":
        parameters = dict(parameters)
        estimator = RandomForestClassifier(
            **parameters, random_state=config["seed"],
            n_jobs=int(config["threads"]), class_weight=None)
    elif kind == "logistic_regression":
        estimator = LogisticRegression(
            C=parameters["C"], solver="saga",
            max_iter=int(config["logistic_max_iter"]),
            tol=float(config["logistic_tol"]),
            random_state=config["seed"], class_weight=None)
    else:
        raise ValueError(f"Unknown estimator: {kind}")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def fit_model(kind, parameters, frame, config):
    features = config.get('categorical_features', CAT_FEATURES) + config.get('numeric_features', NUM_FEATURES)
    y = frame["outcome"].to_numpy(dtype="int8")
    if set(np.unique(y)) != set(range(4)):
        raise ValueError("All four outcomes must occur in the fitting sample")
    parameters = dict(parameters)
    if kind == "random_forest":
        parameters["max_samples"] = min(parameters.get("max_samples", len(frame)), len(frame))
    pipeline = make_pipeline(kind, parameters, config)
    started = time.monotonic()
    converged = False
    with threadpool_limits(limits=int(config["threads"])):
        for attempt in range(2):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(frame[features], y)
            convergence_warnings = [w for w in caught
                                    if issubclass(w.category, ConvergenceWarning)]
            if not convergence_warnings:
                converged = True
                break
            if kind == "logistic_regression":
                pipeline.set_params(model__max_iter=pipeline["model"].max_iter * 2)
        if not converged:
            raise RuntimeError(f"{kind} did not converge; increase max_iter or inspect features")
    assert list(pipeline.classes_) == [0, 1, 2, 3]
    identity_columns = (["source_file", "source_row"]
                        if {"source_file", "source_row"}.issubset(frame.columns)
                        else features + ["outcome"])
    sample_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame[identity_columns], index=False).values.tobytes()
    ).hexdigest()
    return pipeline, {"fit_seconds": time.monotonic() - started,
                      "rows": len(frame), "parameters": parameters,
                      "converged": converged,
                      "sample_fingerprint": sample_hash,
                      "class_counts": np.bincount(y, minlength=4).tolist(),
                      "encoded_features": len(pipeline["preprocess"].get_feature_names_out()),
                      "iterations": (pipeline["model"].n_iter_.tolist()
                                     if kind == "logistic_regression" else None)}


def checked_proba(estimator, frame):
    features = list(getattr(estimator, 'feature_names_in_', FEATURES))
    ps = np.asarray(estimator.predict_proba(frame[features]), dtype=np.float64)
    if not np.array_equal(estimator.classes_, np.arange(4)):
        raise ValueError("Probability class order differs from the public class order")
    if not (np.isfinite(ps).all() and (ps >= 0).all() and (ps <= 1).all()
            and np.allclose(ps.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("Invalid probability vector")
    ps /= ps.sum(axis=1, keepdims=True)
    return ps


def fit_calibrator(estimator, frame):
    if set(frame["outcome"].unique()) != set(range(4)):
        raise ValueError("Calibration requires examples of all four classes")
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    features = list(getattr(estimator, 'feature_names_in_', FEATURES))
    calibrated.fit(frame[features], frame["outcome"].to_numpy())
    return calibrated


def probability_metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    truth = np.eye(4)[y]
    binary_brier = ((truth - probabilities) ** 2).mean(axis=0)
    return {
        "rows": len(y),
        "log_loss": float(log_loss(y, probabilities, labels=range(4))),
        "multiclass_brier": float(binary_brier.sum()),
        "brier_convention": "Mean summed squared error over four classes; range 0 to 2",
        "per_class_brier": dict(zip(CLASS_NAMES, binary_brier.tolist())),
    }


def evaluate(y, probabilities):
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    result = probability_metrics(y, probabilities)
    result["class_prevalence"] = {
        c: float(np.mean(y == k)) for k, c in enumerate(CLASS_NAMES)}
    result["confusion_matrix"] = confusion_matrix(
        y, predictions, labels=range(4)).tolist()
    result["classification"] = classification_report(
        y, predictions, labels=range(4), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)
    result["average_precision"] = {
        c: float(average_precision_score(y == k, probabilities[:, k]))
        if (y == k).any() else None for k, c in enumerate(CLASS_NAMES)}
    result["calibration"] = {}
    for k, name in enumerate(CLASS_NAMES):
        boundaries = np.unique(np.quantile(probabilities[:, k], np.linspace(0, 1, 11)))
        memberships = np.searchsorted(boundaries[1:-1], probabilities[:, k], side="right")
        bins = []
        for group in np.unique(memberships):
            mask = memberships == group
            bins.append({"n": int(mask.sum()),
                         "mean_probability": float(probabilities[mask, k].mean()),
                         "observed_rate": float((y[mask] == k).mean())})
        result["calibration"][name] = bins
    return result


def choose_cancellation_policy(y, probabilities, max_fraction=0.05, precision_lift=2):
    event = np.asarray(y) == 2
    prevalence = float(event.mean())
    base = {"event": "cancelled", "maximum_alert_fraction": max_fraction,
            "minimum_precision": precision_lift * prevalence,
            "selection_prevalence": prevalence}
    if not event.any():
        return {**base, "enabled": False, "threshold": None,
                "reason": "No cancellations in policy selection data"}
    precision, recall, thresholds = precision_recall_curve(event, probabilities[:, 2])
    sorted_scores = np.sort(probabilities[:, 2])
    fractions = (len(event) - np.searchsorted(sorted_scores, thresholds, side="left")) / len(event)
    feasible = ((fractions <= max_fraction) &
                (precision[:-1] >= precision_lift * prevalence) &
                (recall[:-1] > 0))
    candidates = np.flatnonzero(feasible)
    if len(candidates) == 0:
        return {**base, "enabled": False, "threshold": None,
                "reason": "No threshold satisfies the alert budget and precision target"}
    chosen = candidates[np.argmax(recall[candidates])]
    return {**base, "enabled": True, "threshold": float(thresholds[chosen]),
            "selection_precision": float(precision[chosen]),
            "selection_recall": float(recall[chosen]),
            "selection_alert_fraction": float(fractions[chosen]),
            "reason": "Maximize recall within the declared constraints"}


def evaluate_policy(y, probabilities, policy):
    if not policy["enabled"]:
        return {"enabled": False, "reason": policy["reason"]}
    event = np.asarray(y) == 2
    alert = probabilities[:, 2] >= policy["threshold"]
    tp = int((event & alert).sum())
    return {"enabled": True, "threshold": policy["threshold"],
            "alerts": int(alert.sum()), "true_positives": tp,
            "precision": tp / int(alert.sum()) if alert.any() else 0.0,
            "recall": tp / int(event.sum()) if event.any() else None,
            "alerts_per_1000": float(alert.mean() * 1000)}


VARIANTS = {
    'logistic_schedule': ('logistic_regression', False),
    'logistic_rotation': ('logistic_regression', True),
    'forest_schedule': ('random_forest', False),
    'forest_rotation': ('random_forest', True),
}
EXTRA = ['rotation_minutes','rotation_bucket','time_bucket','season']
EB_FEATURES = ['eb_'+c for c in CLASS_NAMES]
SAMPLE_COLUMNS = list(dict.fromkeys(CAT_FEATURES + NUM_FEATURES + EXTRA + [
    'outcome','flight_date','source_file','source_row','origin_code','destination_code','carrier_code']))


def normalized(frame):
    frame = frame.copy()
    for name in CAT_FEATURES + ['rotation_bucket','time_bucket','season']:
        frame[name] = frame[name].astype(str)
    for name in NUM_FEATURES + ['rotation_minutes']:
        frame[name] = pd.to_numeric(frame[name], errors='coerce').astype('float32')
    return frame


def sample(config, period, limit):
    db=connect(config)
    population_clause="AND rotation_status='matched'" if config.get('require_matched_rotation',True) else ''
    try:
        frame=db.execute(f"""SELECT {','.join(SAMPLE_COLUMNS)} FROM read_parquet(?,hive_partitioning=false)
            WHERE flight_date BETWEEN ?::DATE AND ?::DATE {population_clause}
            ORDER BY hash(source_file,source_row,?) LIMIT ?""",
            [files(config),*period,config['seed'],limit]).fetchdf()
    finally:db.close()
    if frame.empty:raise ValueError('No eligible fitting data')
    return normalized(frame)


def model_config(config, rotation):
    result=copy.deepcopy(config)
    result['categorical_features']=CAT_FEATURES+['season','time_bucket']+(['rotation_bucket'] if rotation else [])
    result['numeric_features']=NUM_FEATURES+EB_FEATURES+(['rotation_minutes'] if rotation else [])
    return result


def grouped_counts(config, period, matched=None):
    db=connect(config)
    dims=['route_id','carrier_id','season','month','time_bucket']
    if matched is None:matched=config.get('require_matched_rotation',True)
    clause="AND rotation_status='matched'" if matched else ''
    try:
        frame=db.execute('SELECT '+','.join(dims)+','+
            ','.join(f'sum(outcome={k}) n{k}' for k in range(4))+
            f' FROM read_parquet(?,hive_partitioning=false) WHERE flight_date BETWEEN ?::DATE AND ?::DATE {clause} GROUP BY ALL',
            [files(config),*period]).fetchdf()
    finally:db.close()
    for col in dims:frame[col]=frame[col].astype(str)
    return frame,dims


def train_team(config, root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'bundle.joblib').exists():
        existing=load_model_bundle(root/'bundle.joblib')
        if existing['config']!=config:
            raise ValueError('Run configuration changed; choose a new run name instead of reusing frozen models')
        return existing
    started=time.monotonic();splits=config['splits']
    matched=config.get('require_matched_rotation',True)
    variants={name:value for name,value in VARIANTS.items() if matched or not value[1]}
    coverage_report=require_coverage(config)
    actual={(int(r['year']),int(r['month'])) for r in coverage_report['months']}
    history_months={(d.year,d.month) for d in pd.date_range(config['hierarchy_history_period'][0],config['hierarchy_history_period'][1],freq='MS')}
    if history_months-actual:raise ValueError('Missing monthly source data for empirical-Bayes feature history')
    if config['hierarchy_history_period'][1]>=splits['initial_fit'][0]:
        raise ValueError('Hierarchy feature history must end before any model fitting rows')
    write_json(root/'coverage.json',coverage_report)
    print('Fitting past-only hierarchical features on dedicated history period',flush=True)
    counts,dims=grouped_counts(config,config['hierarchy_history_period'])
    hierarchy=(('route_id',),('route_id','carrier_id'),('route_id','carrier_id','season'),tuple(dims))
    feature_pool=FourLevelPool(tuple(dims),grouping_levels=hierarchy).fit(counts)
    all_counts,all_dims=grouped_counts(config,config['hierarchy_history_period'],False)
    display_counts,_=grouped_counts(config,config.get('pooling_display_period',splits['calibration']))
    pooling=FourLevelPool(tuple(dims),grouping_levels=hierarchy).fit(display_counts)
    all_counts,all_dims=grouped_counts(config,config.get('pooling_display_period',splits['calibration']),False)
    all_pooling=FourLevelPool(tuple(all_dims),grouping_levels=hierarchy).fit(all_counts)
    def with_history(frame):
        ps=feature_pool.predict_proba(frame)
        for j,name in enumerate(EB_FEATURES):frame[name]=ps[:,j].astype('float32')
        return frame
    initial=with_history(sample(config,splits['initial_fit'],config['initial_train_rows']))
    validation=with_history(sample(config,splits['validation'],config['validation_rows']))
    selection={};chosen={}
    for name,(kind,rotation) in variants.items():
        setting=model_config(config,rotation)
        candidates=[{'C':c} for c in config['logistic_candidates']] if kind=='logistic_regression' else config['forest_candidates']
        reports=[]
        for parameters in candidates:
            fitted,fit=fit_model(kind,parameters,initial,setting)
            score=probability_metrics(validation.outcome,checked_proba(fitted,validation))
            reports.append({'parameters':parameters,'fit':fit,**score})
            print(name,parameters,'validation log loss',score['log_loss'],flush=True)
        selection[name]=reports;chosen[name]=min(reports,key=lambda x:x['log_loss'])['parameters']
    del initial,validation
    write_json(root/'selection.json',selection)
    final=with_history(sample(config,splits['final_fit'],config['final_train_rows']))
    cal_fit=with_history(sample(config,splits['calibration_fit'],config['calibration_selection_rows']))
    cal_select=with_history(sample(config,splits['calibration_selection'],config['calibration_selection_rows']))
    cal_all=with_history(sample(config,splits['calibration'],config['calibration_rows']))
    models={};fitting={};calibration={}
    for name,(kind,rotation) in variants.items():
        setting=model_config(config,rotation);parameters=dict(chosen[name])
        if kind=='random_forest':parameters['n_estimators']=config['final_forest_trees']
        print('Final fitting',name,flush=True)
        fitted,fit=fit_model(kind,parameters,final,setting)
        raw=probability_metrics(cal_select.outcome,checked_proba(fitted,cal_select))
        candidate=fit_calibrator(fitted,cal_fit)
        adjusted=probability_metrics(cal_select.outcome,checked_proba(candidate,cal_select))
        selected='sigmoid' if adjusted['log_loss']<raw['log_loss'] else 'none'
        models[name]=fit_calibrator(fitted,cal_all) if selected=='sigmoid' else fitted
        fit['predictor_names']=setting['categorical_features']+setting['numeric_features']
        fitting[name]=fit;calibration[name]={'selected':selected,'raw':raw,'sigmoid':adjusted}
        if kind=='random_forest':
            models[name+'_sigmoid']=fit_calibrator(fitted,cal_all)
            fitting[name+'_sigmoid']={**fit,'explicit_calibrated_variant':True}
            calibration[name+'_sigmoid']={'selected':'sigmoid','raw':raw,'sigmoid':adjusted,
                                          'retained_for_team_method_comparison':True}
        print(name,'calibration',selected,flush=True)
    assert len({r['sample_fingerprint'] for r in fitting.values()})==1
    del final,cal_fit,cal_select,cal_all
    bundle={'run_name':root.name,'created_at':utcnow(),'config':config,'models':models,
            'pooling':pooling,'feature_pool':feature_pool,'all_flight_pooling':all_pooling,
            'fit_reports':fitting,'calibration':calibration,'selection':selection,'versions':versions(),
            'population':('Flights with valid retrospective scheduled tail rotation' if matched
                          else 'All accepted flights; no tail or rotation availability requirement'),
            'test_interpretation':config['test_interpretation']}
    temporary=root/'bundle.joblib.tmp';joblib.dump(bundle,temporary,compress=3);temporary.replace(root/'bundle.joblib')
    summary={k:v for k,v in bundle.items() if k not in ['models','pooling','feature_pool','all_flight_pooling','config','selection']}
    summary.update(training_seconds=time.monotonic()-started,
        hierarchical_rotation={'counts_rows':int(display_counts[['n0','n1','n2','n3']].sum().sum()),'levels':pooling.fitting_report},
        historical_feature_pool={'period':config['hierarchy_history_period'],'counts_rows':int(counts[['n0','n1','n2','n3']].sum().sum()),'levels':feature_pool.fitting_report},
        hierarchical_all_flights={'counts_rows':int(all_counts[['n0','n1','n2','n3']].sum().sum()),'levels':all_pooling.fitting_report})
    write_json(root/'training-summary.json',summary)
    return bundle


def evaluate_team(bundle, root, reuse_predictions=False):
    root=Path(root);config=bundle['config'];started=time.monotonic()
    if (root/'evaluation.json').exists():return json.loads((root/'evaluation.json').read_text())
    target=root/'predictions.parquet';temporary=target.with_suffix('.tmp')
    db=connect(config);writer=None;n=0
    cached=bool(reuse_predictions and target.exists())
    try:
        if cached:
            missing=[name for name in bundle['models'] if f'{name}_on_time' not in pq.ParquetFile(target).schema.names]
            if missing:
                for batch in pq.ParquetFile(target).iter_batches(batch_size=config['score_batch_rows']):
                    frame=batch.to_pandas()
                    for name in missing:
                        ps=checked_proba(bundle['models'][name],frame)
                        for j,c in enumerate(CLASS_NAMES):frame[f'{name}_{c}']=ps[:,j].astype('float32')
                    table=pa.Table.from_pandas(frame,preserve_index=False)
                    if writer is None:writer=pq.ParquetWriter(temporary,table.schema,compression='zstd')
                    writer.write_table(table)
                writer.close();writer=None;temporary.replace(target)
            n=pq.ParquetFile(target).metadata.num_rows
        population_clause="AND rotation_status='matched'" if config.get('require_matched_rotation',True) else ''
        query=f"SELECT {','.join(SAMPLE_COLUMNS)} FROM read_parquet(?,hive_partitioning=false) WHERE flight_date BETWEEN ?::DATE AND ?::DATE {population_clause}"
        reader=[] if cached else db.execute(query,[files(config),*config['splits']['test']]).to_arrow_reader(config['score_batch_rows'])
        for batch in reader:
            frame=normalized(batch.to_pandas())
            historical=bundle.get('feature_pool',bundle['pooling']).predict_proba(frame)
            for j,col in enumerate(EB_FEATURES):frame[col]=historical[:,j].astype('float32')
            out=frame.copy()
            for name,model in bundle['models'].items():
                ps=checked_proba(model,frame)
                for j,label in enumerate(CLASS_NAMES):out[f'{name}_{label}']=ps[:,j].astype('float32')
            ps=bundle['pooling'].predict_proba(frame)
            for j,label in enumerate(CLASS_NAMES):
                out[f'hierarchical_{label}']=ps[:,j].astype('float32')
                out[f'global_{label}']=np.float32(bundle['pooling'].global_probabilities[j])
            table=pa.Table.from_pandas(out,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(temporary,table.schema,compression='zstd')
            writer.write_table(table);n+=len(frame)
            if n%500000<config['score_batch_rows']:print('Scored eligible test flights',f'{n:,}',flush=True)
    finally:
        if writer:writer.close()
        db.close()
    if not cached:temporary.replace(target)
    report={'created_at':utcnow(),'reference_period':config['splits']['test'],'rows':n,
            'population':bundle['population'],'test_interpretation':bundle['test_interpretation'],
            'models':{},'subgroups':{},'interval_validation':{}}
    y=pq.read_table(target,columns=['outcome']).column(0).to_numpy().astype('int8')
    for name in [*bundle['models'],'hierarchical','global']:
        print('Metrics',name,flush=True)
        columns=[f'{name}_{c}' for c in CLASS_NAMES];table=pq.read_table(target,columns=columns)
        ps=np.column_stack([table[c].to_numpy() for c in columns])
        report['models'][name]=evaluate(y,ps)
        report['models'][name]['roc_auc']={c:float(roc_auc_score(y==j,ps[:,j]))
            if (y==j).any() and (y!=j).any() else None for j,c in enumerate(CLASS_NAMES)}
        del table,ps
    db=connect(config)
    try:
        expressions=[]
        for name in bundle['models']:
            total='('+ '+'.join(f'{name}_{c}' for c in CLASS_NAMES)+')'
            selected='CASE outcome '+ ' '.join(f'WHEN {j} THEN {name}_{c}' for j,c in enumerate(CLASS_NAMES))+' END'
            brier='+'.join(f'power({name}_{c}/{total}-CAST(outcome={j} AS INTEGER),2)' for j,c in enumerate(CLASS_NAMES))
            expressions += [f'sum(-ln(greatest(1e-15,({selected})/{total}))) {name}_loss',f'sum({brier}) {name}_brier']
        weekly=db.execute("SELECT date_trunc('week',flight_date)::DATE week_start,count(*) n,"+','.join(expressions)+
                          ' FROM read_parquet(?) GROUP BY week_start ORDER BY week_start',[str(target)]).fetchdf()
        indices=np.random.default_rng(config['seed']).integers(0,len(weekly),size=(1000,len(weekly)))
        resampled_n=weekly.n.to_numpy()[indices].sum(axis=1);paired={}
        pairs=[('forest_vs_logistic','forest_schedule','logistic_schedule')]
        if 'forest_rotation' in bundle['models']:
            pairs.extend([('forest','forest_rotation','forest_schedule'),('logistic','logistic_rotation','logistic_schedule')])
        if 'forest_rotation_sigmoid' in bundle['models']:
            pairs.append(('forest_sigmoid','forest_rotation_sigmoid','forest_schedule_sigmoid'))
        for kind,rotation_name,schedule_name in pairs:
            paired[kind]={}
            for metric in ['loss','brier']:
                delta=(weekly[f'{rotation_name}_{metric}']-weekly[f'{schedule_name}_{metric}']).to_numpy()
                samples=delta[indices].sum(axis=1)/resampled_n
                paired[kind][metric]={'first_minus_second':float(delta.sum()/weekly.n.sum()),
                                     'first_model':rotation_name,'second_model':schedule_name,
                                     'interval_95':np.quantile(samples,[.025,.975]).tolist()}
        report['paired_comparisons']={'method':'1000 paired week bootstrap replicates','weeks':len(weekly),'results':paired}
        display_forest='forest_rotation' if 'forest_rotation' in bundle['models'] else 'forest_schedule'
        display_logistic='logistic_rotation' if 'logistic_rotation' in bundle['models'] else 'logistic_schedule'
        counts=', '.join(f'sum(outcome={j}) n{j}' for j in range(4))
        report['evening_test_comparison']=db.execute(
            f"SELECT rotation_bucket,count(*) n,{counts},avg(CAST(outcome=1 AS INTEGER)) observed_delay,avg({display_forest}_delayed) forest_delay "
            "FROM read_parquet(?) WHERE time_bucket='evening' GROUP BY rotation_bucket ORDER BY rotation_bucket",[str(target)]).fetchdf().to_dict('records')
        example=db.execute(f"""SELECT carrier_id,route_id,season,month,time_bucket,
            min(origin_code) origin,min(destination_code) destination,min(carrier_code) carrier,
            count(*) n,{counts},avg({display_forest}_delayed) forest_delay,
            avg({display_logistic}_delayed) logistic_delay FROM read_parquet(?)
            GROUP BY ALL ORDER BY n DESC LIMIT 20""",[str(target)]).fetchdf()
        profiles=[]
        for row in example.to_dict('records'):
            description=bundle['pooling'].describe(row)
            p=np.array(description['probabilities']);icons=np.floor(p*100).astype(int)
            for j in np.argsort(p*100-icons)[::-1][:100-int(icons.sum())]:icons[j]+=1
            profiles.append({**row,'empirical_bayes':description,
                'icon_array_100':dict(zip(CLASS_NAMES,icons.tolist())),
                'uncertainty_display_score':float(1-min(1,(description['intervals_95'][1][1]-description['intervals_95'][1][0])/.25))})
        write_json(root/'dashboard-profiles.example.json',{'class_order':CLASS_NAMES,'synthetic':False,
            'population':bundle['population'],'profiles':profiles,
            'note':'Intervals belong to empirical-Bayes leaf profiles, not RF prediction confidence.'})
        rows=[];pool=bundle['pooling'];leaf=pool.levels[-1]
        for key,own in pool.counts[-1].items():
            alpha=own+pool.strengths[-1]*pool.priors[-1][key]
            rows.append({**dict(zip(leaf,key)),'n_fit':int(own.sum()),
                         **{f'alpha_{c}':float(alpha[j]) for j,c in enumerate(CLASS_NAMES)}})
        pd.DataFrame(rows).to_parquet(root/'hierarchical-posterior.parquet',index=False,compression='zstd')
        cells=db.execute(f"SELECT route_id,carrier_id,season,month,time_bucket,count(*) n,{counts} "
            "FROM read_parquet(?) GROUP BY route_id,carrier_id,season,month,time_bucket HAVING count(*)>=100 ORDER BY hash(route_id,carrier_id,season,month,time_bucket) LIMIT 5000",[str(target)]).fetchdf()
        interval_checks={c:[] for c in CLASS_NAMES}
        for row in cells.to_dict('records'):
            info=bundle['pooling'].describe(row)
            if info['hierarchy_level']!=4 or info['matching_fit_rows']<100:continue
            for j,c in enumerate(CLASS_NAMES):
                observed=row[f'n{j}']/row['n'];lower,upper=info['intervals_95'][j]
                interval_checks[c].append(lower<=observed<=upper)
        report['interval_validation']={'kind':'Descriptive fraction of 2025 cell rates inside prior-year fitted posterior intervals; not nominal coverage',
            'minimum_fit_and_test_rows':100,'max_test_cells':5000,
            'by_outcome':{c:{'cells':len(v),'fraction_inside':float(np.mean(v)) if v else None} for c,v in interval_checks.items()}}
    finally:db.close()
    report['evaluation_seconds']=time.monotonic()-started
    write_json(root/'evaluation.json',report)
    return report


def evaluate_all_flight_pool(bundle, root):
    """Evaluate the schedule hierarchy separately on the full flight population."""
    root=Path(root);path=root/'all-flight-hierarchy-evaluation.json'
    if path.exists():return json.loads(path.read_text())
    config=bundle['config'];pool=bundle['all_flight_pooling'];target=root/'all-flight-hierarchy-predictions.parquet'
    db=connect(config);writer=None
    try:
        rows=db.execute("""SELECT route_id,carrier_id,season,month,time_bucket,outcome
            FROM read_parquet(?,hive_partitioning=false) WHERE flight_date BETWEEN ?::DATE AND ?::DATE""",
            [files(config),*config['splits']['test']]).to_arrow_reader(config['score_batch_rows'])
        for batch in rows:
            frame=batch.to_pandas()
            for col in pool.leaf_features:frame[col]=frame[col].astype(str)
            ps=pool.predict_proba(frame)
            for j,c in enumerate(CLASS_NAMES):frame[f'p_{c}']=ps[:,j].astype('float32')
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(target,table.schema,compression='zstd')
            writer.write_table(table)
    finally:
        if writer:writer.close()
        db.close()
    table=pq.read_table(target,columns=['outcome']+[f'p_{c}' for c in CLASS_NAMES])
    y=table['outcome'].to_numpy().astype('int8');ps=np.column_stack([table[f'p_{c}'].to_numpy() for c in CLASS_NAMES])
    results=evaluate(y,ps)
    results['roc_auc']={c:float(roc_auc_score(y==j,ps[:,j])) for j,c in enumerate(CLASS_NAMES)}
    global_ps=np.tile(pool.global_probabilities,(len(y),1))
    report={'population':'All accepted flights, without requiring tail rotation availability',
            'test_period':config['splits']['test'],'hierarchy_history':config.get('pooling_display_period'),
            'test_interpretation':config['test_interpretation'],
            'models':{'hierarchical_schedule':results,'global_recent':evaluate(y,global_ps)}}
    write_json(path,report);return report


def outcome_icons(p):
    p = np.asarray(p, float)
    if p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-6):
        raise ValueError('Expected four coherent probabilities')
    p = p/p.sum()
    counts = np.floor(p*100).astype(int)
    remainder = 100-int(counts.sum())
    counts[np.argsort(-(p*100-counts), kind='stable')[:remainder]] += 1
    return counts.tolist()


def extra_diagnostics(bundle, root):
    db = connect(bundle['config']); target = str(root/'predictions.parquet')
    expressions = []
    for name in bundle['models']:
        selected = 'CASE outcome '+' '.join(f'WHEN {j} THEN {name}_{c}' for j,c in enumerate(CLASS_NAMES))+' END'
        total = '('+'+'.join(f'{name}_{c}' for c in CLASS_NAMES)+')'
        brier = '+'.join(f'power({name}_{c}/{total}-CAST(outcome={j} AS INTEGER),2)' for j,c in enumerate(CLASS_NAMES))
        expressions += [f'avg(-ln(greatest(1e-15,({selected})/{total}))) {name}_log_loss',f'avg({brier}) {name}_brier']
    try:
        monthly = db.execute('SELECT month,count(*) flights,'+','.join(expressions)+
            ' FROM read_parquet(?) GROUP BY month ORDER BY CAST(month AS INTEGER)',[target]).fetchdf()
        counts = ','.join(f'sum(outcome={j}) n{j}' for j in range(4))
        cells = db.execute(f"""SELECT route_id,carrier_id,season,month,time_bucket,count(*) n,{counts}
            FROM read_parquet(?) GROUP BY route_id,carrier_id,season,month,time_bucket HAVING count(*)>=100
            ORDER BY hash(route_id,carrier_id,season,month,time_bucket) LIMIT 5000""",[target]).fetchdf()
    finally: db.close()
    monthly.to_csv(root/'monthly-results.csv',index=False)
    checks = {c:[] for c in CLASS_NAMES}
    for row in cells.to_dict('records'):
        alpha,n_fit,level = bundle['pooling'].posterior(row)
        if level != len(bundle['pooling'].levels) or n_fit < 100: continue
        for j,c in enumerate(CLASS_NAMES):
            lo,hi = betabinom.ppf([.025,.975],int(row['n']),alpha[j],alpha.sum()-alpha[j])
            checks[c].append(bool(lo<=row[f'n{j}']<=hi))
    report = {'monthly_results':monthly.to_dict('records'),'predictive_interval_checks':{
        'method':'Marginal 95% beta-binomial future-count intervals, conditional on fitted priors',
        'minimum_fit_and_test_rows':100,'maximum_checked_cells':5000,
        'caveat':'Exploratory cross-year diagnostic; prior fitting uncertainty and risk drift remain.',
        'by_outcome':{c:{'cells':len(v),'fraction_inside':float(np.mean(v)) if v else None} for c,v in checks.items()}}}
    write_json(root/'diagnostics.json',report)
    return report


def verify_run(bundle,root):
    config=bundle['config']; target=root/'predictions.parquet'; db=connect(config)
    clause="AND rotation_status='matched'" if config['require_matched_rotation'] else ''
    try:
        expected=db.execute(f"SELECT count(*) FROM read_parquet(?,hive_partitioning=false) WHERE flight_date BETWEEN ?::DATE AND ?::DATE {clause}",
            [files(config),*config['splits']['test']]).fetchone()[0]
        differences=db.execute(f"""WITH expected AS (SELECT source_file,source_row FROM read_parquet(?,hive_partitioning=false)
            WHERE flight_date BETWEEN ?::DATE AND ?::DATE {clause}),actual AS (SELECT source_file,source_row FROM read_parquet(?))
            SELECT count(*) FROM ((SELECT * FROM expected EXCEPT SELECT * FROM actual)
            UNION ALL (SELECT * FROM actual EXCEPT SELECT * FROM expected))""",
            [files(config),*config['splits']['test'],str(target)]).fetchone()[0]
        distinct=db.execute('SELECT count(*) FROM (SELECT DISTINCT source_file,source_row FROM read_parquet(?))',[str(target)]).fetchone()[0]
        sample=db.execute('SELECT * FROM read_parquet(?) ORDER BY hash(source_file,source_row) LIMIT 1000',[str(target)]).fetchdf()
    finally: db.close()
    actual=pq.ParquetFile(target).metadata.num_rows
    if actual!=expected or distinct!=actual or differences: raise AssertionError('Prediction source keys do not reconcile')
    errors={}
    for name,estimator in bundle['models'].items():
        errors[name]=float(np.max(np.abs(checked_proba(estimator,sample)-sample[[f'{name}_{c}' for c in CLASS_NAMES]].to_numpy())))
        if errors[name]>1e-6: raise AssertionError('Saved model predictions changed')
    np.testing.assert_allclose(sample[EB_FEATURES],bundle['feature_pool'].predict_proba(sample),atol=1e-7,rtol=0)
    same=len({r['sample_fingerprint'] for r in bundle['fit_reports'].values()})==1
    if not same: raise AssertionError('Fitting rows differ within population')
    report={'expected_rows':expected,'scored_rows':actual,'distinct_source_keys':distinct,'source_key_differences':differences,
        'maximum_reload_error_by_model':errors,'history_features_reproduced':True,'same_fitting_rows_within_population':same}
    write_json(root/'verification.json',report)
    return report


def dashboard_export(bundle,root):
    output=root/'dashboard'; output.mkdir(exist_ok=True)
    forest='forest_rotation' if 'forest_rotation' in bundle['models'] else 'forest_schedule'
    logistic='logistic_rotation' if 'logistic_rotation' in bundle['models'] else 'logistic_schedule'
    averages=','.join(f'avg({model}_{c}) {label}_{c}' for model,label in [(forest,'forest'),(logistic,'logistic')] for c in CLASS_NAMES)
    db=connect(bundle['config'])
    try:
        frame=db.execute(f"""SELECT origin_code,destination_code,carrier_code,route_id,carrier_id,season,month,time_bucket,
            count(*) n_backtest,{','.join(f'sum(outcome={j}) observed_{c}' for j,c in enumerate(CLASS_NAMES))},{averages}
            FROM read_parquet(?) GROUP BY ALL ORDER BY origin_code,route_id,carrier_id,month,time_bucket""",
            [str(root/'predictions.parquet')]).fetchdf()
    finally: db.close()
    keys=list(bundle['pooling'].leaf_features)
    descriptions=[bundle['pooling'].posterior(row) for row in frame[keys].to_dict('records')]
    alpha=np.asarray([item[0] for item in descriptions]); total=alpha.sum(axis=1)
    for j,c in enumerate(CLASS_NAMES):
        frame['eb_'+c]=alpha[:,j]/total
        frame['eb_'+c+'_lower']=beta.ppf(.025,alpha[:,j],total-alpha[:,j])
        frame['eb_'+c+'_upper']=beta.ppf(.975,alpha[:,j],total-alpha[:,j])
    frame['n_fit_group']=[item[1] for item in descriptions]
    frame['hierarchy_level']=[item[2] for item in descriptions]
    pool=bundle['pooling']
    frame['exact_leaf_fit_rows']=[int(pool.counts[-1].get(tuple(str(row[c]) for c in keys),np.zeros(4)).sum()) for row in frame[keys].to_dict('records')]
    frame.to_parquet(root/'dashboard-cells.parquet',index=False,compression='zstd')
    columns=list(frame.columns); origins=[]
    for origin,group in frame.groupby('origin_code',sort=True):
        filename=str(origin)+'.json.gz'
        payload={'origin':str(origin),'columns':columns,'rows':group.to_numpy().tolist()}
        (output/filename).write_bytes(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0))
        origins.append({'origin':str(origin),'file':filename,'cells':len(group),
            'backtest_flights':int(group.n_backtest.sum()),'bytes':(output/filename).stat().st_size})
    report={'schema_version':1,'class_order':CLASS_NAMES,'synthetic':False,'population':bundle['population'],
        'test_period':bundle['config']['splits']['test'],'historical_eb_period':bundle['config']['pooling_display_period'],
        'model_means_scope':'Frozen-model averages over 2025 schedules; historical backtest comparisons',
        'interval_scope':'EB probabilities only; broader fallback groups have hierarchy_level<4; not RF uncertainty',
        'month_filter':'Monthly cells; never average interval endpoints across months',
        'rotation_caveat':'Retrospective tail assignments have not been validated for booking availability',
        'cells':len(frame),'origins':origins}
    write_json(output/'manifest.json',report)
    return {'cells':len(frame),'origins':len(origins),'compressed_bytes':sum(x['bytes'] for x in origins)}


TEAM_UI_AIRPORTS = set('ATL DFW DEN ORD LAX JFK LGA EWR SFO SEA MIA BOS MCO CLT'.split())
TEAM_UI_TIME_BLOCKS = {
    'morning': 'Morning (05-11)', 'afternoon': 'Afternoon (12-16)',
    'evening': 'Evening (17-20)', 'night': 'Night (21-04)',
}
TEAM_UI_CAUSES = {
    'CarrierDelay': 'carrier', 'WeatherDelay': 'weather', 'NASDelay': 'nas',
    'SecurityDelay': 'security', 'LateAircraftDelay': 'late_aircraft',
}
TEAM_UI_ROUTE_KEYS = ['origin_code', 'destination_code']
TEAM_UI_CELL_KEYS = TEAM_UI_ROUTE_KEYS + ['carrier_code', 'time_bucket', 'month']


def ui_posteriors(pool, frame):
    """Compute 2024 posterior summaries for the exact grain used in the UI."""
    records = [pool.posterior(row) for row in frame.to_dict('records')]
    alpha = np.stack([record[0] for record in records])
    total = alpha.sum(axis=1, keepdims=True)
    mean = alpha / total
    lower = beta.ppf(0.025, alpha[:, 1], total[:, 0] - alpha[:, 1])
    upper = beta.ppf(0.975, alpha[:, 1], total[:, 0] - alpha[:, 1])
    return mean, lower, upper, alpha, records


def annual_ui_pool(monthly_pool):
    dimensions = ('route_id', 'carrier_id', 'time_bucket')
    indices = [monthly_pool.leaf_features.index(name) for name in dimensions]
    annual = {}
    for key, counts in monthly_pool.counts[-1].items():
        annual_key = tuple(key[index] for index in indices)
        if annual_key not in annual:
            annual[annual_key] = counts.copy()
        else:
            annual[annual_key] += counts
    grouped = pd.DataFrame(
        [(*key, *counts) for key, counts in annual.items()],
        columns=[*dimensions, 'n0', 'n1', 'n2', 'n3'],
    )
    return FourLevelPool(
        dimensions, grouping_levels=(dimensions[:1], dimensions[:2], dimensions),
    ).fit(grouped)


def team_delay_cause_minutes(config, cohort_path, selected_routes):
    """Sum reported 2025 delay minutes for selected scored flights by source row."""
    root = Path(config['raw_data_dir']) / 'raw'
    db = connect(config)
    try:
        db.register('selected_ui_routes', selected_routes[TEAM_UI_ROUTE_KEYS])
        scored = db.execute("""
            SELECT p.source_file, p.source_row, p.origin_code, p.destination_code,
                   p.carrier_code, p.time_bucket, CAST(p.month AS INTEGER) AS month
            FROM read_parquet(?) p JOIN selected_ui_routes USING(origin_code, destination_code)
            WHERE p.outcome = 1
            ORDER BY p.source_file, p.source_row
        """, [str(cohort_path / 'predictions.parquet')]).fetchdf()
    finally:
        db.close()

    cause_columns = list(TEAM_UI_CAUSES)
    aggregates = []
    missing = []
    for filename_, flights in scored.groupby('source_file', sort=False):
        path = root / filename_
        if not path.is_file():
            missing.append(filename_)
            continue
        with zipfile.ZipFile(path) as archive:
            member = next(n for n in archive.namelist() if n.lower().endswith('.csv'))
            with archive.open(member) as entrance:
                raw = pd.read_csv(entrance, usecols=cause_columns, dtype='float64')
        if flights.source_row.max() >= len(raw):
            raise ValueError(f'Delay cause source rows are out of bounds in {filename_}')
        joined = flights[TEAM_UI_CELL_KEYS].reset_index(drop=True)
        for source, label in TEAM_UI_CAUSES.items():
            joined[label] = raw[source].to_numpy()[flights.source_row.to_numpy()]
        summed = joined.groupby(TEAM_UI_CELL_KEYS, as_index=False)[list(TEAM_UI_CAUSES.values())].sum()
        aggregates.append(summed)

    labels = list(TEAM_UI_CAUSES.values())
    if not aggregates:
        return {}, missing
    monthly = pd.concat(aggregates, ignore_index=True).groupby(
        TEAM_UI_CELL_KEYS, as_index=False,
    )[labels].sum()
    annual = monthly.groupby(TEAM_UI_CELL_KEYS[:-1], as_index=False)[labels].sum().assign(month=0)
    cause_frames = pd.concat([monthly, annual], ignore_index=True)
    cause_lookup = {}
    for row in cause_frames.itertuples(index=False, name=None):
        minutes = np.clip(np.asarray(row[len(TEAM_UI_CELL_KEYS):], dtype=float), 0, None)
        total = float(minutes.sum())
        if total > 0:
            cause_lookup[row[:len(TEAM_UI_CELL_KEYS)]] = (dict(zip(labels, (minutes / total).tolist())), total)
    return cause_lookup, missing


def team_ui_matrix_rows(frame, pool, causes):
    mean, lower, upper, alpha, support = ui_posteriors(pool, frame)
    quantile_positions = (np.arange(20) + 0.5) / 20
    dots = beta.ppf(quantile_positions, alpha[:, 1, None], alpha.sum(axis=1)[:, None] - alpha[:, 1, None])
    rows = []
    icon_keys = ('on_time', 'delayed_15m', 'cancelled', 'diverted')
    for i, row in enumerate(frame.to_dict('records')):
        month = int(row['month'])
        forest = [float(row['forest_' + key]) for key in CLASS_NAMES]
        logistic = [float(row['logistic_' + key]) for key in CLASS_NAMES]
        n = int(row['n_backtest'])
        cause_key = tuple(row[key] for key in TEAM_UI_CELL_KEYS)
        shares, minutes = causes.get(cause_key, (dict.fromkeys(TEAM_UI_CAUSES.values(), 0.0), 0.0))
        width = float(upper[i] - lower[i])
        rows.append({
            'route': f"{row['origin_code']}-{row['destination_code']}",
            'carrier': row['carrier_code'],
            'season': row['season'].title() if month else 'All Seasons',
            'month': month, 'month_name': calendar.month_name[month] if month else 'All Months',
            'dep_time_block': TEAM_UI_TIME_BLOCKS[row['time_bucket']],
            'sample_size_n': n,
            'delay15_eb_prob': round(float(mean[i, 1]), 6),
            'delay15_rf_prob': round(forest[1], 6),
            'delay15_ci_lower': round(float(lower[i]), 6),
            'delay15_ci_upper': round(float(upper[i]), 6),
            'delay15_ci_width': round(width, 6),
            'cancel_eb_prob': round(float(mean[i, 2]), 6),
            'cancel_rf_prob': round(forest[2], 6),
            'divert_eb_prob': round(float(mean[i, 3]), 6),
            'divert_rf_prob': round(forest[3], 6),
            'vsup_confidence': round(1.0 - min(1.0, width / 0.25), 6),
            'icon_array_100': dict(zip(icon_keys, outcome_icons(forest))),
            'delay_causes': {key: round(value, 6) for key, value in shares.items()},
            'reported_cause_minutes': int(round(minutes)),
            'quantile_dots_20': np.round(dots[i], 6).tolist(),
            'rf_probabilities': dict(zip(CLASS_NAMES, [round(p, 8) for p in forest])),
            'logistic_probabilities': dict(zip(CLASS_NAMES, [round(p, 8) for p in logistic])),
            'historical_eb_sample_size': support[i][1],
            'historical_eb_hierarchy_level': support[i][2],
            'observed_delay_rate': round(float(row['observed_delayed']) / n, 6),
        })
    return rows


def write_team_ui_json(path, payload):
    """Keep dashboard files compact enough for ordinary GitHub uploads."""
    temp = path.with_suffix('.json.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    temp.replace(path)


def export_team_dashboard(config, run_root, output, bundle=None):
    """Write the exact filenames and field names read by the team's dashboard."""
    cohort = Path(run_root) / 'all-flights'
    cell_file = cohort / 'dashboard-cells.parquet'
    if not cell_file.is_file() or not (cohort / 'bundle.joblib').is_file():
        raise FileNotFoundError(f'Run the model before exporting the dashboard. Missing artifacts in {cohort}')
    if bundle is None:
        bundle = load_model_bundle(cohort / 'bundle.joblib')
    if bundle['config']['require_matched_rotation']:
        raise ValueError('The default dashboard must use the all-flight schedule models')

    monthly = pd.read_parquet(cell_file)
    monthly['month'] = monthly['month'].astype(int)
    route_keys = TEAM_UI_ROUTE_KEYS + ['route_id']
    cell_keys = TEAM_UI_CELL_KEYS[:-1] + ['route_id', 'carrier_id']
    routes = monthly.groupby(route_keys, as_index=False).agg(
        n=('n_backtest', 'sum'), n_delay=('observed_delayed', 'sum'),
    )
    routes = routes.sort_values(['n', *TEAM_UI_ROUTE_KEYS], ascending=[False, True, True])
    visible = routes.origin_code.isin(TEAM_UI_AIRPORTS) & routes.destination_code.isin(TEAM_UI_AIRPORTS)
    selected = routes[visible] if visible.any() else routes.head(150)
    monthly = monthly.merge(selected[TEAM_UI_ROUTE_KEYS], on=TEAM_UI_ROUTE_KEYS, validate='many_to_one')
    if monthly.duplicated(TEAM_UI_CELL_KEYS).any():
        raise ValueError('Multiple internal IDs share a team dashboard cell')

    probability_fields = [f'{model}_{key}' for model in ('forest', 'logistic') for key in CLASS_NAMES]
    weighted = monthly[cell_keys + ['n_backtest', 'observed_delayed', *probability_fields]].copy()
    weighted[probability_fields] = weighted[probability_fields].multiply(weighted.n_backtest, axis=0)
    annual = weighted.groupby(cell_keys, as_index=False).sum()
    annual[probability_fields] = annual[probability_fields].divide(annual.n_backtest, axis=0)
    annual = annual.assign(month=0, season='All Seasons')

    monthly_pool = bundle['pooling']
    year_pool = annual_ui_pool(monthly_pool)
    routes_pool = FourLevelPool(('route_id',), counts=year_pool.counts[:1],
                                priors=year_pool.priors[:1], strengths=year_pool.strengths[:1],
                                global_counts=year_pool.global_counts, grouping_levels=(('route_id',),))
    p, lower, upper, _, support = ui_posteriors(routes_pool, routes[['route_id']])
    route_records = [{
        'Route': f'{row.origin_code}-{row.destination_code}',
        'Origin': row.origin_code, 'Dest': row.destination_code, 'n': int(row.n),
        'raw_rate': round(float(row.n_delay / row.n), 6), 'eb_prob': round(float(p[i, 1]), 6),
        'ci_lower': round(float(lower[i]), 6), 'ci_upper': round(float(upper[i]), 6),
        'ci_width': round(float(upper[i] - lower[i]), 6), 'historical_eb_sample_size': support[i][1],
    } for i, row in enumerate(routes.itertuples(index=False))]

    causes, missing = team_delay_cause_minutes(config, cohort, selected)
    cells = team_ui_matrix_rows(annual, year_pool, causes) + team_ui_matrix_rows(monthly, monthly_pool, causes)
    source_config = bundle['config']
    metadata = {
        'population': bundle['population'],
        'prediction_period': source_config['splits']['test'],
        'historical_interval_period': source_config['pooling_display_period'],
        'model_training_period': source_config['splits']['final_fit'],
        'test_interpretation': bundle['test_interpretation'],
        'sample_size_definition': 'Number of held-out flights. The historical EB sample size is reported separately.',
        'delay_causes_definition': 'Shares of reported delay minutes among held-out delayed flights.',
        'matrix_scope': 'Route pairs selectable in Visual.html, or top observed routes for a custom dataset without those airports.',
        'missing_cause_archives': missing,
        'prediction_models': {'rf_probabilities': 'forest_schedule', 'logistic_probabilities': 'logistic_schedule'},
        'interval_definition': 'Historical empirical Bayes intervals for delay probability.',
        'github_foundation_commit': '5643905fa4c066ddb45bd9ff2527d6986673001a',
    }
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_team_ui_json(output / 'ui_route_map.json', {'routes': route_records, 'metadata': metadata})
    write_team_ui_json(output / 'ui_carrier_time_matrix.json', {'cells': cells, 'metadata': metadata})
    print(f'Team dashboard export: {len(route_records):,} routes and {len(cells):,} cells in {output}', flush=True)
    return {'routes': len(route_records), 'cells': len(cells), 'output': str(output)}


def configuration(path=None, work_dir=DEFAULT_WORK_DIR):
    if path is None:
        base = Path(work_dir).expanduser().resolve()
        config = copy.deepcopy(DEFAULT_CONFIG)
        for name in ('data_dir', 'raw_data_dir', 'artifacts_dir'):
            config[name] = str(base / config[name])
        validate_splits(config['splits'])
    else:
        path = Path(path).expanduser().resolve()
        config = load_config(path)
        config['raw_data_dir'] = str((path.parent / config['raw_data_dir']).resolve())
    history = config['hierarchy_history_period']
    if history[0] > history[1] or history[1] >= config['splits']['initial_fit'][0]:
        raise ValueError('Historical outcome features must predate all model fitting')
    if config['pooling_display_period'][1] >= config['splits']['test'][0]:
        raise ValueError('Display hierarchy cannot use test outcomes')
    return config


def load_model_bundle(path):
    """Load a trusted bundle, including ones saved by the original v4 package."""
    package = 'flight_risk_v4'
    legacy_module = package + '.hierarchical'
    if importlib.util.find_spec(package) is not None:
        return joblib.load(path)

    legacy_package = types.ModuleType(package)
    legacy_package.__path__ = []
    hierarchy = types.ModuleType(legacy_module)
    hierarchy.FourLevelPool = FourLevelPool
    sys.modules[package] = legacy_package
    sys.modules[legacy_module] = hierarchy
    try:
        return joblib.load(path)
    finally:
        sys.modules.pop(legacy_module, None)
        sys.modules.pop(package, None)


def missing_prepared_months(config):
    prepared = Path(config['data_dir']) / 'clean'
    return [
        (year, month)
        for year in config['years']
        for month in range(1, 13)
        if not (prepared / f'year={year}' / f'month={month:02d}' / 'data.parquet').is_file()
    ]


def data_audit(config, output):
    coverage = require_coverage(config)
    db = connect(config)
    try:
        years = db.execute("""SELECT year,count(*) flights,
            sum(rotation_status='matched') linked_flights,sum(outcome=2) cancellations,
            sum(outcome=2 AND rotation_status='matched') linked_cancellations
            FROM read_parquet(?,hive_partitioning=false) GROUP BY year ORDER BY year""",
            [files(config)]).fetchdf()
        statuses = db.execute("""SELECT rotation_status,count(*) flights,sum(outcome=2) cancellations
            FROM read_parquet(?,hive_partitioning=false) GROUP BY rotation_status ORDER BY flights DESC""",
            [files(config)]).fetchdf()
    finally:
        db.close()
    path = Path(config['raw_data_dir']) / 'ingestion-manifest.json'
    counts = json.loads(path.read_text()).get('totals') if path.exists() else None
    report = {'created_at': utcnow(), 'coverage': coverage,
        'yearly_linkage': years.to_dict('records'), 'rotation_statuses': statuses.to_dict('records'),
        'ingestion_counts': counts, 'synthetic': False,
        'interpretation': 'All-flight schedule models and linked-rotation comparisons use distinct populations.'}
    write_json(output / 'data-audit.json', report)
    return report


def prepare(config, allow_download):
    if allow_download:
        download(config['raw_data_dir'], config['years'])
    ingest(config['raw_data_dir'], config['years'], chunk_rows=config['csv_chunk_rows'])
    source_config = {**config, 'data_dir': config['raw_data_dir']}
    build_rotations(source_config, config['data_dir'], workers=config['threads'])


def predict_profiles(bundle_path, input_path, output_path):
    bundle = load_model_bundle(bundle_path)
    frame = pd.read_csv(input_path, dtype=str)
    validate_profile_inputs(frame, bundle['config']['require_matched_rotation'])
    if bundle['config']['require_matched_rotation']:
        if 'rotation_status' not in frame or not frame.rotation_status.eq('matched').all():
            raise ValueError('Rotation model requires valid matched-rotation status')
    for key, value in [('rotation_minutes', np.nan), ('rotation_bucket', 'unknown')]:
        if key not in frame:
            frame[key] = value
    frame = normalized(frame)
    historical = bundle['feature_pool'].predict_proba(frame)
    for j, feature in enumerate(EB_FEATURES):
        frame[feature] = historical[:, j]
    output = frame.copy()
    for name, estimator in bundle['models'].items():
        p = checked_proba(estimator, frame)
        for j, c in enumerate(CLASS_NAMES):
            output[f'{name}_{c}'] = p[:, j]
        output[name + '_predicted_class'] = np.asarray(CLASS_NAMES)[p.argmax(axis=1)]
    output.to_csv(output_path, index=False)


def validate_profile_inputs(frame, rotation=False):
    """Reject invalid requests rather than silently encoding them as unknown."""
    required={'origin_id','destination_id','route_id','carrier_id','month','departure_hour',
              'weekday','season','time_bucket','distance','scheduled_duration'}
    if missing:=required-set(frame.columns):
        raise ValueError('Missing profile fields: '+', '.join(sorted(missing)))
    if frame.empty:
        raise ValueError('No profiles supplied')
    ranges={'origin_id':(1,9999999),'destination_id':(1,9999999),'carrier_id':(1,9999999),
            'month':(1,12),'departure_hour':(0,23),'weekday':(1,7)}
    for key,(lo,hi) in ranges.items():
        values=pd.to_numeric(frame[key],errors='coerce')
        if not (values.notna() & values.between(lo,hi) & values.mod(1).eq(0)).all():
            raise ValueError('Invalid '+key)
        frame[key]=values.astype(int).astype(str)
    route=frame.origin_id+'_'+frame.destination_id
    if not frame.route_id.eq(route).all():
        raise ValueError('route_id must agree with origin_id and destination_id')
    month=frame.month.astype(int); hour=frame.departure_hour.astype(int)
    expected_season=month.map({12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',
                               6:'summer',7:'summer',8:'summer',9:'fall',10:'fall',11:'fall'})
    expected_block=np.select([hour.between(5,11),hour.between(12,16),hour.between(17,20)],
                              ['morning','afternoon','evening'],default='night')
    if not frame.season.eq(expected_season).all() or not frame.time_bucket.eq(expected_block).all():
        raise ValueError('Season or departure block conflicts with month/hour')
    if rotation:
        if not {'rotation_minutes','rotation_bucket','rotation_status'}.issubset(frame):
            raise ValueError('Rotation fields are required for this model')
        minutes=pd.to_numeric(frame.rotation_minutes,errors='coerce')
        bucket=np.select([minutes.lt(40),minutes.lt(90),minutes.lt(180)],
                          ['lt40','40_89','90_179'],default='180_1440')
        if not (minutes.notna() & minutes.between(0,1440)).all() or not frame.rotation_bucket.eq(bucket).all():
            raise ValueError('Invalid or inconsistent rotation bucket')


def run(config, run_name, ui_dir=TEAM_UI_DIR):
    started = time.monotonic()
    root = Path(config['artifacts_dir']) / run_name
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'config.resolved.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Use a new run name when the configuration changes')
    write_json(path, config)
    audit = data_audit(config, root)
    results = {}
    for matched, label in [(False, 'all-flights'), (True, 'rotation-linked')]:
        cohort = copy.deepcopy(config)
        cohort['require_matched_rotation'] = matched
        output = root / label
        print('\nRunning population:', label, flush=True)
        train_team(cohort, output)
        bundle = load_model_bundle(output / 'bundle.joblib')
        report = evaluate_team(bundle, output)
        print('Checking predictions and exporting dashboard data:', label, flush=True)
        diagnostics = extra_diagnostics(bundle, output)
        verification = verify_run(bundle, output)
        export = dashboard_export(bundle, output)
        if not matched:
            results['team_dashboard'] = export_team_dashboard(config, root, ui_dir, bundle)
        results[label] = {'models': report['models'], 'population': bundle['population'],
            'paired_comparisons': report['paired_comparisons'],
            'evening_comparison': report['evening_test_comparison'],
            'calibration': bundle['calibration'], 'fit_reports': bundle['fit_reports'],
            'diagnostics': diagnostics, 'verification': verification, 'dashboard_export': export}
    results.update(created_at=utcnow(), seconds=time.monotonic()-started,
        data_audit=audit, test_interpretation=config['test_interpretation'],
        source_provenance=SOURCE_PROVENANCE)
    write_json(root / 'findings.json', results)
    print('Completed both populations:', root, flush=True)
    return results


def main(args):
    if args.command == 'predict':
        predict_profiles(args.bundle, args.input, args.output)
        return

    config = configuration(args.config, args.work_dir)
    if args.command == 'export-dashboard':
        export_team_dashboard(config, Path(config['artifacts_dir']) / args.run, args.ui_dir)
    elif args.command == 'prepare':
        prepare(config, args.download)
    elif args.command == 'audit':
        print(json.dumps(data_audit(config, Path(config['artifacts_dir'])), indent=2))
    else:
        run_name = args.run if args.command == 'run' else DEFAULT_RUN
        missing = missing_prepared_months(config)
        if missing and (args.command is None or args.download):
            print(f'Preparing BTS data in {config["raw_data_dir"]}', flush=True)
            prepare(config, allow_download=True)
        elif missing:
            raise SystemExit(
                'Prepared data is missing. Run python model.py prepare --download '
                'or add --download to the run command.'
            )
        run(config, run_name, args.ui_dir if args.command == 'run' else TEAM_UI_DIR)


if __name__ == '__main__':
    main(CLI_ARGS)
