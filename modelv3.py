#!/usr/bin/env python3
"""
CS6242 Group Project — Team 185
Flight Disruption Risk Model & End-to-End BTS Data Pipeline
Authors: Anand Appiya, Benjamin Ahlmark, Rajesh Rajendran, Navraj Grewal, Sumeet Kahlon

Pipeline Overview (matching Figure 1 of Team 185 Proposal + Month-of-Year Granularity):
1. Data Fetching (BTS On-Time Performance 1987–present):
   - Programmatically downloads monthly BTS On-Time Performance ZIP archives directly
     from the Bureau of Transportation Statistics (transtats.bts.gov/PREZIP/).
   - Extracts CSVs, normalizes schema, and stores partitioned Parquet files.
   - Supports both Apache Spark (PySpark) for full 235M+ record scaling and Pandas/PyArrow
     for immediate local prototyping in VSCode.
2. Aircraft Rotation Reconstruction:
   - Tracks daily aircraft rotations using `Tail_Number` and `FlightDate`.
   - Computes turnaround time (`turnaround_mins`), previous flight arrival delay
     (`prev_arr_delay`), leg sequence number, and tight evening turn indicators
     (`<40 min` vs `90–179 min`).
3. Innovation 1 — Hierarchical Empirical-Bayes Beta-Binomial Shrinkage (with Month):
   - Pools sparse `Route x Carrier x Month x DepTimeBlock` cells toward parent
     hierarchies (`Route x Carrier x Season` -> `Route x Carrier` -> `Route` -> Global).
   - Computes posterior disruption probabilities, 95% Beta credible intervals,
     value-suppressing uncertainty scores, 100-unit icon arrays, and delay cause breakdowns.
4. Innovation 2 — Rotation-Aware Calibrated Random Forest:
   - Trains calibrated Random Forest classifiers (`CalibratedClassifierCV` +
     `RandomForestClassifier` via scikit-learn, with an automatic pure-NumPy/SciPy
     calibrated random forest fallback if `scikit-learn` is not yet installed) for
     P(delay 15+), P(cancel), and P(divert).
   - Evaluates models on chronologically held-out months against historical baselines
     and non-rotation Random Forest baselines using ROC-AUC and Brier scores.
5. Innovation 3 Support — D3.js Pre-Aggregated Export (Month-Filterable):
   - Exports lightweight JSON & CSV artifacts with month-by-month (`month: 1..12`) and
     all-months (`month: 0`) cell summaries ready for the D3.js route map and
     carrier x departure-time matrix UI.
"""

from __future__ import annotations

import argparse
import json
import logging
import ssl
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta as beta_dist

# Optional scikit-learn import (uses sklearn when available in VSCode, or built-in
# NumPy/SciPy calibrated tree ensemble if scikit-learn is not installed yet)
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import brier_score_loss as sk_brier_score_loss
    from sklearn.metrics import roc_auc_score as sk_roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CS6242_Team185")


# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

BTS_PREZIP_BASE_URLS = [
    "https://www.transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip",
    "https://www.transtats.bts.gov/PREZIP/On_Time_On_Time_Performance_{year}_{month}.zip",
]

BTS_CORE_COLUMNS = [
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrDel15",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "CRSElapsedTime",
    "ActualElapsedTime",
    "AirTime",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

LEGACY_COLUMN_MAP = {
    "UniqueCarrier": "Reporting_Airline",
    "TailNum": "Tail_Number",
    "FlightNum": "Flight_Number_Reporting_Airline",
}

DELAY_CAUSE_COLS = [
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

MONTH_NAMES = {
    0: "All Months",
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


@dataclass
class PipelineConfig:
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")
    years: List[int] = None
    months: List[int] = None
    rf_n_estimators: int = 120
    rf_max_depth: int = 12
    random_state: int = 42
    use_spark: bool = False
    demo_mode: bool = False
    demo_n_flights: int = 60000

    def __post_init__(self):
        if self.years is None:
            self.years = [2024]
        if self.months is None:
            self.months = list(range(1, 13))
        self.raw_dir = self.data_dir / "raw_bts"
        self.parquet_dir = self.data_dir / "parquet"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. AUTOMATED BTS DATA FETCHER (WITH SSL & PARQUET CACHING)
# ==============================================================================

def _create_permissive_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
    except ssl.SSLError:
        pass
    return ctx


def fetch_bts_month(
    year: int,
    month: int,
    raw_dir: Path,
    parquet_dir: Path,
    force_download: bool = False,
) -> Path:
    """
    Downloads a single month of BTS On-Time Performance data from the official
    BTS PREZIP server, extracts the CSV, normalizes columns, and saves a compressed
    Parquet file for fast reloading.
    """
    parquet_path = parquet_dir / f"bts_ontime_{year}_{month:02d}.parquet"
    if parquet_path.exists() and not force_download:
        logger.info(f"[Cache Hit] Found existing Parquet for {year}-{month:02d}: {parquet_path}")
        return parquet_path

    zip_path = raw_dir / f"bts_ontime_{year}_{month:02d}.zip"
    if not zip_path.exists() or force_download:
        ssl_ctx = _create_permissive_ssl_context()
        downloaded = False
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }

        for url_template in BTS_PREZIP_BASE_URLS:
            url = url_template.format(year=year, month=month)
            logger.info(f"Fetching BTS archive for {year}-{month:02d} from: {url}")
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as resp:
                    total_bytes = int(resp.headers.get("Content-Length", 0))
                    chunk_size = 1024 * 1024
                    bytes_read = 0
                    with open(zip_path, "wb") as out_f:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            out_f.write(chunk)
                            bytes_read += len(chunk)
                            if total_bytes > 0 and (bytes_read // chunk_size) % 5 == 0:
                                pct = (bytes_read / total_bytes) * 100
                                logger.info(
                                    f"  Download progress ({year}-{month:02d}): "
                                    f"{bytes_read / (1024**2):.1f} / {total_bytes / (1024**2):.1f} MB ({pct:.0f}%)"
                                )
                downloaded = True
                logger.info(f"Saved BTS ZIP archive to {zip_path} ({bytes_read / (1024**2):.1f} MB)")
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                logger.warning(f"URL attempt failed ({url}): {exc}")

        if not downloaded:
            raise RuntimeError(
                f"Could not download BTS data for {year}-{month:02d} from TranStats PREZIP. "
                "Check your internet connection, try an earlier month, "
                "or run with `--demo` to test the full pipeline on realistic synthetic BTS data."
            )

    logger.info(f"Extracting and converting {zip_path.name} -> Parquet...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No CSV file found inside {zip_path}")
        target_csv = csv_members[0]
        with zf.open(target_csv) as csv_file:
            header_df = pd.read_csv(csv_file, nrows=0, low_memory=False)
            available_cols = list(header_df.columns)

        use_cols = []
        for col in available_cols:
            mapped = LEGACY_COLUMN_MAP.get(col, col)
            if mapped in BTS_CORE_COLUMNS:
                use_cols.append(col)

        with zf.open(target_csv) as csv_file:
            df = pd.read_csv(
                csv_file,
                usecols=use_cols,
                dtype={
                    "Reporting_Airline": "string",
                    "UniqueCarrier": "string",
                    "Tail_Number": "string",
                    "TailNum": "string",
                    "Origin": "string",
                    "Dest": "string",
                    "CancellationCode": "string",
                },
                low_memory=False,
            )

    df = df.rename(columns=LEGACY_COLUMN_MAP)
    for col in BTS_CORE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df.to_parquet(parquet_path, index=False)
    logger.info(f"Stored cleaned Parquet ({len(df):,} flight records): {parquet_path}")
    return parquet_path


def generate_synthetic_bts_dataset(n_flights: int = 60000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic 12-month BTS On-Time Performance dataset with authentic
    aircraft rotation chains (Tail_Number sequences), monthly weather seasonality (winter
    snowstorms in Jan/Feb/Dec, summer convective storms in Jun/Jul/Aug, calmer autumn/spring),
    and evening turnaround delay propagation matching Team 185's proposal.
    """
    logger.info(f"Generating realistic 12-month synthetic BTS dataset ({n_flights:,} flights)...")
    rng = np.random.default_rng(seed)

    carriers = ["DL", "AA", "UA", "WN", "B6", "AS", "NK", "F9"]
    carrier_base_delay = {
        "DL": 0.12, "AS": 0.13, "UA": 0.17, "WN": 0.19,
        "AA": 0.21, "B6": 0.25, "NK": 0.27, "F9": 0.29,
    }
    carrier_base_cancel = {
        "DL": 0.008, "AS": 0.010, "WN": 0.013, "UA": 0.016,
        "AA": 0.019, "B6": 0.023, "NK": 0.026, "F9": 0.029,
    }

    # Monthly seasonality multiplier (higher disruption in Jun/Jul/Dec/Jan, lower in Sep/Oct/Apr)
    month_risk_multiplier = {
        1: 1.22,   # Jan: Winter storms
        2: 1.15,   # Feb: Winter ice/snow
        3: 1.05,   # Mar: Spring break volume
        4: 0.92,   # Apr: Moderate spring
        5: 0.98,   # May: Late spring
        6: 1.28,   # Jun: Summer thunderstorms & peak load
        7: 1.32,   # Jul: Peak summer storms
        8: 1.18,   # Aug: Late summer
        9: 0.78,   # Sep: Lowest delay month historically
        10: 0.82,  # Oct: Crisp autumn weather
        11: 0.95,  # Nov: Thanksgiving volume
        12: 1.25,  # Dec: Holiday & winter weather
    }

    airports = ["ATL", "DFW", "DEN", "ORD", "LAX", "JFK", "LGA", "SFO", "SEA", "MIA", "BOS", "EWR", "MCO", "CLT"]
    route_distances = {}
    route_congestion = {}
    for o in airports:
        for d in airports:
            if o != d:
                route_distances[(o, d)] = int(rng.integers(250, 2600))
                hub_boost = 0.05 if (o in {"EWR", "JFK", "LGA", "ORD"} or d in {"EWR", "JFK", "LGA", "ORD"}) else 0.0
                route_congestion[(o, d)] = hub_boost

    # Sample dates across all 12 months of 2024
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="3D")
    n_aircraft = max(35, n_flights // (len(dates) * 4))

    records = []
    flight_id = 1000

    for f_date in dates:
        month = int(f_date.month)
        quarter = (month - 1) // 3 + 1
        day_of_month = int(f_date.day)
        day_of_week = int(f_date.dayofweek) + 1
        date_str = f_date.strftime("%Y-%m-%d")

        m_mult = month_risk_multiplier[month]
        daily_weather_factor = m_mult * (1.40 if rng.random() < 0.08 else 1.0)

        for ac_idx in range(n_aircraft):
            if len(records) >= n_flights:
                break
            carrier = carriers[ac_idx % len(carriers)]
            tail_num = f"N{100 + ac_idx:03d}{carrier}"
            n_legs = int(rng.integers(3, 6))

            curr_origin = rng.choice(airports)
            curr_dep_mins = int(rng.integers(330, 540))
            prev_arr_delay_mins = 0.0
            prev_cancelled = 0

            for leg in range(n_legs):
                if len(records) >= n_flights:
                    break
                dest_choices = [a for a in airports if a != curr_origin]
                curr_dest = rng.choice(dest_choices)
                dist = route_distances[(curr_origin, curr_dest)]
                cong = route_congestion[(curr_origin, curr_dest)]
                flight_mins = int(40 + dist * 0.115)

                if leg == 0:
                    turnaround_scheduled = 240
                else:
                    turn_bucket = rng.choice(["tight", "medium", "relaxed"], p=[0.28, 0.47, 0.25])
                    if turn_bucket == "tight":
                        turnaround_scheduled = int(rng.integers(22, 39))
                    elif turn_bucket == "medium":
                        turnaround_scheduled = int(rng.integers(40, 89))
                    else:
                        turnaround_scheduled = int(rng.integers(90, 175))
                    curr_dep_mins += turnaround_scheduled

                if curr_dep_mins >= 1410:
                    break

                crs_dep_hhmm = (curr_dep_mins // 60) * 100 + (curr_dep_mins % 60)
                crs_arr_total = (curr_dep_mins + flight_mins) % 1440
                crs_arr_hhmm = (crs_arr_total // 60) * 100 + (crs_arr_total % 60)

                is_evening = 17 <= (curr_dep_mins // 60) <= 21
                base_p_delay = (carrier_base_delay[carrier] + cong) * daily_weather_factor
                if is_evening and leg > 0:
                    if turnaround_scheduled < 40:
                        p_delay = (0.367 * m_mult) + cong + 0.005 * max(0.0, prev_arr_delay_mins)
                    elif turnaround_scheduled >= 90:
                        p_delay = (0.244 * m_mult) + cong + 0.002 * max(0.0, prev_arr_delay_mins)
                    else:
                        p_delay = (0.295 * m_mult) + cong + 0.003 * max(0.0, prev_arr_delay_mins)
                else:
                    turn_penalty = 0.10 if (leg > 0 and turnaround_scheduled < 40) else 0.0
                    prop_penalty = min(0.45, 0.007 * max(0.0, prev_arr_delay_mins))
                    p_delay = base_p_delay + turn_penalty + prop_penalty

                p_delay = float(np.clip(p_delay, 0.03, 0.90))
                p_cancel = float(
                    np.clip(
                        (carrier_base_cancel[carrier] + 0.2 * cong) * daily_weather_factor
                        + (0.06 if prev_cancelled else 0.0)
                        + (0.03 if prev_arr_delay_mins > 45 else 0.0),
                        0.002,
                        0.28,
                    )
                )
                p_divert = float(
                    np.clip(
                        0.0025 * daily_weather_factor + (0.005 if prev_arr_delay_mins > 60 else 0.0),
                        0.001,
                        0.04,
                    )
                )

                is_cancelled = int(rng.random() < p_cancel)
                is_diverted = int(0 if is_cancelled else (rng.random() < p_divert))

                if is_cancelled or is_diverted:
                    arr_del15 = 0
                    dep_del15 = 0
                    arr_delay = np.nan
                    dep_delay = np.nan
                    c_delay = w_delay = n_delay = s_delay = l_delay = np.nan
                    prev_arr_delay_mins = 50.0 if is_cancelled else 35.0
                    prev_cancelled = is_cancelled
                else:
                    arr_del15 = int(rng.random() < p_delay)
                    if arr_del15:
                        arr_delay = float(rng.integers(15, 140))
                        dep_delay = float(max(0.0, arr_delay - rng.integers(-10, 15)))
                        dep_del15 = int(dep_delay >= 15.0)
                        late_ac_share = 0.50 if prev_arr_delay_mins >= 15 else 0.15
                        l_delay = round(arr_delay * late_ac_share, 1)
                        rem = arr_delay - l_delay
                        c_delay = round(rem * 0.45, 1)
                        n_delay = round(rem * 0.35, 1)
                        w_delay = round(rem * 0.18, 1)
                        s_delay = round(max(0.0, rem - c_delay - n_delay - w_delay), 1)
                    else:
                        arr_delay = float(rng.integers(-18, 14))
                        dep_delay = float(rng.integers(-10, 12))
                        dep_del15 = 0
                        c_delay = w_delay = n_delay = s_delay = l_delay = 0.0

                    prev_arr_delay_mins = max(0.0, arr_delay)
                    prev_cancelled = 0

                records.append(
                    {
                        "Year": f_date.year,
                        "Quarter": quarter,
                        "Month": month,
                        "DayofMonth": day_of_month,
                        "DayOfWeek": day_of_week,
                        "FlightDate": date_str,
                        "Reporting_Airline": carrier,
                        "Tail_Number": tail_num,
                        "Flight_Number_Reporting_Airline": str(flight_id),
                        "Origin": curr_origin,
                        "Dest": curr_dest,
                        "CRSDepTime": crs_dep_hhmm,
                        "DepTime": crs_dep_hhmm,
                        "DepDelay": dep_delay,
                        "DepDelayMinutes": max(0.0, dep_delay) if pd.notna(dep_delay) else np.nan,
                        "DepDel15": dep_del15,
                        "CRSArrTime": crs_arr_hhmm,
                        "ArrTime": crs_arr_hhmm,
                        "ArrDelay": arr_delay,
                        "ArrDelayMinutes": max(0.0, arr_delay) if pd.notna(arr_delay) else np.nan,
                        "ArrDel15": arr_del15,
                        "Cancelled": is_cancelled,
                        "CancellationCode": "B" if is_cancelled and daily_weather_factor > 1.0 else ("A" if is_cancelled else ""),
                        "Diverted": is_diverted,
                        "CRSElapsedTime": float(flight_mins),
                        "ActualElapsedTime": float(flight_mins),
                        "AirTime": float(max(20, flight_mins - 20)),
                        "Distance": float(dist),
                        "CarrierDelay": c_delay,
                        "WeatherDelay": w_delay,
                        "NASDelay": n_delay,
                        "SecurityDelay": s_delay,
                        "LateAircraftDelay": l_delay,
                    }
                )
                flight_id += 1
                curr_origin = curr_dest
                curr_dep_mins = crs_arr_total

    df = pd.DataFrame(records)
    logger.info(
        f"Generated {len(df):,} synthetic flights across {df['Month'].nunique()} months and {df['Tail_Number'].nunique()} aircraft."
    )
    return df


# ==============================================================================
# 3. PREPROCESSING & AIRCRAFT ROTATION RECONSTRUCTION (PANDAS & PYSPARK)
# ==============================================================================

def hhmm_to_minutes(hhmm_series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(hhmm_series, errors="coerce").fillna(0).astype(int)
    hours = (vals // 100).clip(0, 24)
    mins = (vals % 100).clip(0, 59)
    return hours * 60 + mins


def assign_season(month_series: pd.Series) -> pd.Series:
    m = pd.to_numeric(month_series, errors="coerce").fillna(1).astype(int)
    return pd.Series(
        np.select(
            [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])],
            ["Winter", "Spring", "Summer"],
            default="Fall",
        ),
        index=month_series.index,
    )


def assign_dep_time_block(crs_dep_time: pd.Series) -> pd.Series:
    hours = (pd.to_numeric(crs_dep_time, errors="coerce").fillna(0).astype(int) // 100).clip(0, 23)
    return pd.Series(
        np.select(
            [
                hours.between(5, 11),
                hours.between(12, 16),
                hours.between(17, 20),
            ],
            [
                "Morning (05-11)",
                "Afternoon (12-16)",
                "Evening (17-20)",
            ],
            default="Night (21-04)",
        ),
        index=crs_dep_time.index,
    )


def rebuild_rotations_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuilds daily aircraft rotations using `Tail_Number` and `FlightDate` (Innovation 2).
    """
    logger.info("Rebuilding aircraft rotation chains and turnaround features...")
    df = df.copy()

    df["Origin"] = df["Origin"].astype(str).str.strip()
    df["Dest"] = df["Dest"].astype(str).str.strip()
    df["Reporting_Airline"] = df["Reporting_Airline"].astype(str).str.strip()
    df["Route"] = df["Origin"] + "-" + df["Dest"]
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").fillna(1).astype(int)

    df["Cancelled"] = pd.to_numeric(df["Cancelled"], errors="coerce").fillna(0).astype(int)
    df["Diverted"] = pd.to_numeric(df["Diverted"], errors="coerce").fillna(0).astype(int)
    df["ArrDel15"] = pd.to_numeric(df["ArrDel15"], errors="coerce").fillna(0).astype(int)
    df["DepDel15"] = pd.to_numeric(df["DepDel15"], errors="coerce").fillna(0).astype(int)

    df["Season"] = assign_season(df["Month"])
    df["DepTimeBlock"] = assign_dep_time_block(df["CRSDepTime"])
    df["dep_mins_midnight"] = hhmm_to_minutes(df["CRSDepTime"])
    df["arr_mins_midnight"] = hhmm_to_minutes(df["CRSArrTime"])

    df["Tail_Number"] = df["Tail_Number"].fillna("UNKNOWN").astype(str).str.strip()
    valid_tail_mask = ~df["Tail_Number"].isin(["", "UNKNOWN", "nan", "None", "UNKN"])

    df = df.sort_values(["FlightDate", "Tail_Number", "dep_mins_midnight"]).reset_index(drop=True)

    grp = df.groupby(["FlightDate", "Tail_Number"], sort=False)
    df["rotation_leg_num"] = np.where(valid_tail_mask, grp.cumcount() + 1, 1)
    df["prev_arr_mins"] = np.where(valid_tail_mask, grp["arr_mins_midnight"].shift(1), np.nan)
    df["prev_arr_delay"] = np.where(
        valid_tail_mask,
        grp["ArrDelayMinutes"].shift(1).fillna(0.0),
        0.0,
    )
    df["prev_cancelled"] = np.where(
        valid_tail_mask,
        grp["Cancelled"].shift(1).fillna(0).astype(int),
        0,
    )
    df["prev_origin"] = np.where(valid_tail_mask, grp["Origin"].shift(1), "NONE")

    raw_turnaround = df["dep_mins_midnight"] - df["prev_arr_mins"]
    raw_turnaround = np.where(raw_turnaround < -300, raw_turnaround + 1440, raw_turnaround)
    df["turnaround_mins"] = np.where(
        (df["rotation_leg_num"] > 1) & pd.notna(raw_turnaround),
        np.clip(raw_turnaround, 5.0, 600.0),
        np.nan,
    )

    df["turnaround_bucket"] = np.select(
        [
            df["rotation_leg_num"] == 1,
            df["turnaround_mins"] < 40,
            df["turnaround_mins"].between(40, 89.999),
            df["turnaround_mins"].between(90, 179.999),
        ],
        [
            "First_Leg",
            "<40m",
            "40-89m",
            "90-179m",
        ],
        default="180m+",
    )

    is_evening_or_night = df["DepTimeBlock"].isin(["Evening (17-20)", "Night (21-04)"])
    df["is_tight_evening_turn"] = (
        (df["rotation_leg_num"] > 1) & is_evening_or_night & (df["turnaround_mins"] < 40)
    ).astype(int)

    evening_sub = df[is_evening_or_night & (df["rotation_leg_num"] > 1) & (df["Cancelled"] == 0)]
    if len(evening_sub) > 0:
        tight_rate = evening_sub.loc[evening_sub["turnaround_mins"] < 40, "ArrDel15"].mean()
        relaxed_rate = evening_sub.loc[evening_sub["turnaround_mins"].between(90, 179), "ArrDel15"].mean()
        logger.info(
            f"[Rotation Diagnostic] Evening turns <40m late rate: {tight_rate * 100:.1f}% vs. "
            f"90-179m late rate: {relaxed_rate * 100:.1f}%"
        )

    return df


def rebuild_rotations_spark(parquet_paths: List[Path]) -> pd.DataFrame:
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        logger.warning("PySpark not installed in current Python environment; using Pandas engine.")
        dfs = [pd.read_parquet(p) for p in parquet_paths]
        return rebuild_rotations_pandas(pd.concat(dfs, ignore_index=True))

    logger.info("Starting local Apache Spark session for BTS rotation reconstruction...")
    spark = (
        SparkSession.builder.appName("CS6242_Team185_BTS_Pipeline")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    sdf = spark.read.parquet(*[str(p) for p in parquet_paths])
    pdf = sdf.toPandas()
    spark.stop()
    return rebuild_rotations_pandas(pdf)


# ==============================================================================
# 4. INNOVATION 1: HIERARCHICAL EMPIRICAL-BAYES BETA-BINOMIAL SHRINKAGE
# ==============================================================================

class HierarchicalEmpiricalBayesShrinkage:
    """
    Implements Innovation 1: Hierarchical Beta-Binomial Empirical-Bayes Shrinkage
    with Month-of-Year support:
      Level 0 (Global) -> Level 1 (Route) -> Level 2 (Route x Carrier)
      -> Level 3 (Route x Carrier x Season) -> Level 4 (Route x Carrier x Season x Month x DepTimeBlock)
    Also produces an all-months Level 3B (Route x Carrier x DepTimeBlock) table so users can
    toggle between 'All Months' and any specific month (January..December) in the UI.
    """

    HIERARCHY_LEVELS = [
        ["Route"],
        ["Route", "Reporting_Airline"],
        ["Route", "Reporting_Airline", "Season"],
        ["Route", "Reporting_Airline", "Season", "Month", "DepTimeBlock"],
    ]

    def __init__(self, min_prior_strength: float = 12.0, max_prior_strength: float = 120.0):
        self.min_prior_strength = min_prior_strength
        self.max_prior_strength = max_prior_strength
        self.global_means_: Dict[str, float] = {}
        self.level_tables_: Dict[str, Dict[int, pd.DataFrame]] = {}
        self.cell_summary_df_: Optional[pd.DataFrame] = None
        self.annual_cell_summary_df_: Optional[pd.DataFrame] = None

    def _estimate_concentration_m(self, n: np.ndarray, k: np.ndarray, mu_parent: np.ndarray) -> float:
        valid = n >= 5
        if np.sum(valid) < 5:
            return 25.0
        p_obs = k[valid] / n[valid]
        mu = np.clip(mu_parent[valid], 1e-4, 1.0 - 1e-4)
        sample_var = np.var(p_obs - mu, ddof=1)
        expected_binom_var = np.mean(mu * (1.0 - mu) / n[valid])
        excess_var = max(1e-6, sample_var - expected_binom_var)
        avg_pq = float(np.mean(mu * (1.0 - mu)))
        m_est = max(1.0, (avg_pq / excess_var) - 1.0)
        return float(np.clip(m_est, self.min_prior_strength, self.max_prior_strength))

    def _build_cell_table_from_levels(
        self,
        df: pd.DataFrame,
        cell_cols: List[str],
        parent_cols: List[str],
        parent_tables: Dict[str, pd.DataFrame],
        targets: Tuple[str, ...],
    ) -> pd.DataFrame:
        target_dfs = {}
        for target in targets:
            agg = (
                df.groupby(cell_cols, observed=True)[target]
                .agg(n="count", k="sum")
                .reset_index()
            )
            agg["raw_rate"] = agg["k"] / agg["n"].clip(lower=1)
            parent_df = parent_tables[target]
            agg = agg.merge(
                parent_df[parent_cols + ["eb_prob"]].rename(columns={"eb_prob": "parent_prob"}),
                on=parent_cols,
                how="left",
            )
            agg["parent_prob"] = agg["parent_prob"].fillna(self.global_means_[target])
            m_lvl = self._estimate_concentration_m(
                agg["n"].to_numpy(dtype=float),
                agg["k"].to_numpy(dtype=float),
                agg["parent_prob"].to_numpy(dtype=float),
            )
            agg["prior_M"] = m_lvl
            agg["alpha_post"] = agg["k"] + m_lvl * agg["parent_prob"]
            agg["beta_post"] = (agg["n"] - agg["k"]) + m_lvl * (1.0 - agg["parent_prob"])
            agg["eb_prob"] = agg["alpha_post"] / (agg["alpha_post"] + agg["beta_post"])
            agg["ci_lower"] = beta_dist.ppf(0.025, agg["alpha_post"], agg["beta_post"])
            agg["ci_upper"] = beta_dist.ppf(0.975, agg["alpha_post"], agg["beta_post"])
            agg["ci_width"] = agg["ci_upper"] - agg["ci_lower"]
            target_dfs[target] = agg

        base_cells = target_dfs["ArrDel15"][
            cell_cols + ["n", "k", "raw_rate", "eb_prob", "ci_lower", "ci_upper", "ci_width", "alpha_post", "beta_post"]
        ].rename(
            columns={
                "k": "delay15_count",
                "raw_rate": "delay15_raw_prob",
                "eb_prob": "delay15_eb_prob",
                "ci_lower": "delay15_ci_lower",
                "ci_upper": "delay15_ci_upper",
                "ci_width": "delay15_ci_width",
                "alpha_post": "delay15_alpha",
                "beta_post": "delay15_beta",
            }
        )

        for other_target, prefix in [("Cancelled", "cancel"), ("Diverted", "divert")]:
            sub = target_dfs[other_target][
                cell_cols + ["k", "raw_rate", "eb_prob", "ci_lower", "ci_upper", "ci_width"]
            ].rename(
                columns={
                    "k": f"{prefix}_count",
                    "raw_rate": f"{prefix}_raw_prob",
                    "eb_prob": f"{prefix}_eb_prob",
                    "ci_lower": f"{prefix}_ci_lower",
                    "ci_upper": f"{prefix}_ci_upper",
                    "ci_width": f"{prefix}_ci_width",
                }
            )
            base_cells = base_cells.merge(sub, on=cell_cols, how="left")

        cause_agg = (
            df.groupby(cell_cols, observed=True)[DELAY_CAUSE_COLS]
            .sum()
            .reset_index()
        )
        total_cause_mins = cause_agg[DELAY_CAUSE_COLS].sum(axis=1).replace(0, np.nan)
        for c_col in DELAY_CAUSE_COLS:
            share_col = f"share_{c_col}"
            cause_agg[share_col] = (cause_agg[c_col] / total_cause_mins).fillna(0.0).round(4)

        base_cells = base_cells.merge(
            cause_agg[cell_cols + [f"share_{c}" for c in DELAY_CAUSE_COLS]],
            on=cell_cols,
            how="left",
        )

        max_reasonable_width = 0.25
        base_cells["vsup_confidence"] = (
            1.0 - (base_cells["delay15_ci_width"] / max_reasonable_width).clip(0.0, 1.0)
        ).round(4)
        return base_cells

    def fit(self, df: pd.DataFrame, targets: Tuple[str, ...] = ("ArrDel15", "Cancelled", "Diverted")) -> "HierarchicalEmpiricalBayesShrinkage":
        logger.info("Fitting Hierarchical Empirical-Bayes Beta-Binomial Shrinkage (including Month-of-Year level)...")
        self.level_tables_ = {}

        for target in targets:
            global_mean = float(df[target].mean())
            self.global_means_[target] = global_mean
            self.level_tables_[target] = {}

            prev_level_df = None
            prev_cols = None

            for lvl_idx, cols in enumerate(self.HIERARCHY_LEVELS):
                agg = (
                    df.groupby(cols, observed=True)[target]
                    .agg(n="count", k="sum")
                    .reset_index()
                )
                agg["raw_rate"] = agg["k"] / agg["n"].clip(lower=1)

                if lvl_idx == 0:
                    agg["parent_prob"] = global_mean
                else:
                    agg = agg.merge(
                        prev_level_df[prev_cols + ["eb_prob"]].rename(columns={"eb_prob": "parent_prob"}),
                        on=prev_cols,
                        how="left",
                    )
                    agg["parent_prob"] = agg["parent_prob"].fillna(global_mean)

                m_lvl = self._estimate_concentration_m(
                    agg["n"].to_numpy(dtype=float),
                    agg["k"].to_numpy(dtype=float),
                    agg["parent_prob"].to_numpy(dtype=float),
                )
                agg["prior_M"] = m_lvl
                agg["alpha_post"] = agg["k"] + m_lvl * agg["parent_prob"]
                agg["beta_post"] = (agg["n"] - agg["k"]) + m_lvl * (1.0 - agg["parent_prob"])
                agg["eb_prob"] = agg["alpha_post"] / (agg["alpha_post"] + agg["beta_post"])

                agg["ci_lower"] = beta_dist.ppf(0.025, agg["alpha_post"], agg["beta_post"])
                agg["ci_upper"] = beta_dist.ppf(0.975, agg["alpha_post"], agg["beta_post"])
                agg["ci_width"] = agg["ci_upper"] - agg["ci_lower"]

                self.level_tables_[target][lvl_idx] = agg
                prev_level_df = agg
                prev_cols = cols

        # Build Monthly cell table (Level 4: Route x Carrier x Season x Month x DepTimeBlock)
        lvl2_parents = {t: self.level_tables_[t][2] for t in targets}
        self.cell_summary_df_ = self._build_cell_table_from_levels(
            df=df,
            cell_cols=self.HIERARCHY_LEVELS[-1],
            parent_cols=self.HIERARCHY_LEVELS[2],
            parent_tables=lvl2_parents,
            targets=targets,
        )

        # Also build Annual/All-Months cell table (Route x Carrier x DepTimeBlock, with Month=0)
        lvl1_parents = {t: self.level_tables_[t][1] for t in targets}
        annual_df = self._build_cell_table_from_levels(
            df=df,
            cell_cols=["Route", "Reporting_Airline", "DepTimeBlock"],
            parent_cols=["Route", "Reporting_Airline"],
            parent_tables=lvl1_parents,
            targets=targets,
        )
        annual_df["Season"] = "All Seasons"
        annual_df["Month"] = 0
        self.annual_cell_summary_df_ = annual_df

        logger.info(
            f"Built Empirical-Bayes estimates for {len(self.cell_summary_df_):,} monthly cells "
            f"and {len(self.annual_cell_summary_df_):,} all-months summary cells."
        )
        return self

    def transform_flight_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cell_cols = self.HIERARCHY_LEVELS[-1]

        if self.cell_summary_df_ is not None:
            merge_cols = [
                "delay15_eb_prob",
                "delay15_raw_prob",
                "delay15_ci_width",
                "cancel_eb_prob",
                "cancel_raw_prob",
                "divert_eb_prob",
                "divert_raw_prob",
            ]
            out = out.merge(self.cell_summary_df_[cell_cols + merge_cols], on=cell_cols, how="left")

        # Fallback to Annual (Route x Carrier x DepTimeBlock), then Level 2 (Route x Carrier x Season), then Level 1
        if self.annual_cell_summary_df_ is not None:
            ann_cols = ["Route", "Reporting_Airline", "DepTimeBlock"]
            ann_sub = self.annual_cell_summary_df_[
                ann_cols + ["delay15_eb_prob", "delay15_raw_prob", "cancel_eb_prob", "cancel_raw_prob", "divert_eb_prob", "divert_raw_prob"]
            ].rename(
                columns={
                    "delay15_eb_prob": "_ann_delay15_eb",
                    "delay15_raw_prob": "_ann_delay15_raw",
                    "cancel_eb_prob": "_ann_cancel_eb",
                    "cancel_raw_prob": "_ann_cancel_raw",
                    "divert_eb_prob": "_ann_divert_eb",
                    "divert_raw_prob": "_ann_divert_raw",
                }
            )
            out = out.merge(ann_sub, on=ann_cols, how="left")
            for prefix in ["delay15", "cancel", "divert"]:
                out[f"{prefix}_eb_prob"] = out[f"{prefix}_eb_prob"].fillna(out[f"_ann_{prefix}_eb"])
                out[f"{prefix}_raw_prob"] = out[f"{prefix}_raw_prob"].fillna(out[f"_ann_{prefix}_raw"])
            out = out.drop(
                columns=[
                    "_ann_delay15_eb", "_ann_delay15_raw",
                    "_ann_cancel_eb", "_ann_cancel_raw",
                    "_ann_divert_eb", "_ann_divert_raw",
                ]
            )

        for target, prefix in [("ArrDel15", "delay15"), ("Cancelled", "cancel"), ("Diverted", "divert")]:
            lvl1_df = self.level_tables_[target][1][["Route", "Reporting_Airline", "eb_prob", "raw_rate"]].rename(
                columns={"eb_prob": f"_{prefix}_lvl1_eb", "raw_rate": f"_{prefix}_lvl1_raw"}
            )
            out = out.merge(lvl1_df, on=["Route", "Reporting_Airline"], how="left")
            g_mean = self.global_means_[target]
            out[f"{prefix}_eb_prob"] = out[f"{prefix}_eb_prob"].fillna(out[f"_{prefix}_lvl1_eb"]).fillna(g_mean)
            out[f"{prefix}_raw_prob"] = out[f"{prefix}_raw_prob"].fillna(out[f"_{prefix}_lvl1_raw"]).fillna(g_mean)
            out = out.drop(columns=[f"_{prefix}_lvl1_eb", f"_{prefix}_lvl1_raw"])

        out["delay15_ci_width"] = out["delay15_ci_width"].fillna(0.15)
        return out


# ==============================================================================
# 5. INNOVATION 2: ROTATION-AWARE CALIBRATED RANDOM FOREST & BACKTESTING
# ==============================================================================

def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if HAS_SKLEARN:
        return float(sk_roc_auc_score(y_true, y_prob))
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=float)
    sorted_probs = y_prob[order]
    diffs = np.concatenate(([True], sorted_probs[1:] != sorted_probs[:-1], [True]))
    idx = np.flatnonzero(diffs)
    for i in range(len(idx) - 1):
        start, end = idx[i], idx[i + 1]
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
    sum_pos_ranks = float(np.sum(ranks[y_true == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if HAS_SKLEARN:
        return float(sk_brier_score_loss(y_true, y_prob))
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_prob, dtype=float)) ** 2))


class _NumpyCalibratedRandomForest:
    def __init__(self, n_estimators: int = 80, max_depth: int = 6, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.medians_: Optional[np.ndarray] = None
        self.cat_maps_: List[Dict[str, float]] = []
        self.trees_: List[dict] = []
        self.platt_a_: float = 1.0
        self.platt_b_: float = 0.0

    def _encode_matrix(self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], is_train: bool, y: Optional[np.ndarray] = None) -> np.ndarray:
        X_num = df[num_cols].to_numpy(dtype=float)
        if is_train:
            self.medians_ = np.nanmedian(X_num, axis=0)
            self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        inds = np.where(np.isnan(X_num))
        X_num[inds] = np.take(self.medians_, inds[1])

        cat_arrays = []
        if is_train:
            self.cat_maps_ = []
            global_mean = float(np.mean(y)) if y is not None else 0.15
            for c in cat_cols:
                s = pd.Series(y).groupby(df[c].astype(str).to_numpy()).agg(["count", "mean"])
                smoothed = (s["count"] * s["mean"] + 20.0 * global_mean) / (s["count"] + 20.0)
                cmap = smoothed.to_dict()
                cmap["__DEFAULT__"] = global_mean
                self.cat_maps_.append(cmap)

        for idx, c in enumerate(cat_cols):
            cmap = self.cat_maps_[idx]
            default_val = cmap["__DEFAULT__"]
            vals = df[c].astype(str).map(cmap).fillna(default_val).to_numpy(dtype=float)
            cat_arrays.append(vals[:, None])

        return np.hstack([X_num] + cat_arrays)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator, mtry: int) -> dict:
        n_samples, n_features = X.shape
        pos_rate = float(np.mean(y))
        if depth >= self.max_depth or n_samples < 50 or pos_rate <= 0.0005 or pos_rate >= 0.9995:
            return {"leaf": True, "prob": pos_rate}

        feat_indices = rng.choice(n_features, size=min(mtry, n_features), replace=False)
        best_gain = -1.0
        best_feat = -1
        best_thresh = 0.0
        parent_var = pos_rate * (1.0 - pos_rate)

        for f_idx in feat_indices:
            col = X[:, f_idx]
            quantiles = np.quantile(col, [0.2, 0.4, 0.5, 0.6, 0.8])
            for thresh in np.unique(quantiles):
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left < 20 or n_right < 20:
                    continue
                p_left = float(np.mean(y[left_mask]))
                p_right = float(np.mean(y[~left_mask]))
                child_var = (n_left * p_left * (1.0 - p_left) + n_right * p_right * (1.0 - p_right)) / n_samples
                gain = parent_var - child_var
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f_idx)
                    best_thresh = float(thresh)

        if best_feat == -1 or best_gain <= 1e-6:
            return {"leaf": True, "prob": pos_rate}

        left_mask = X[:, best_feat] <= best_thresh
        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1, rng, mtry),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1, rng, mtry),
        }

    def _predict_tree(self, node: dict, X: np.ndarray) -> np.ndarray:
        if node["leaf"]:
            return np.full(len(X), node["prob"], dtype=float)
        mask = X[:, node["feat"]] <= node["thresh"]
        out = np.empty(len(X), dtype=float)
        if np.any(mask):
            out[mask] = self._predict_tree(node["left"], X[mask])
        if np.any(~mask):
            out[~mask] = self._predict_tree(node["right"], X[~mask])
        return out

    def fit(self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], y: np.ndarray) -> "_NumpyCalibratedRandomForest":
        rng = np.random.default_rng(self.random_state)
        X = self._encode_matrix(df, num_cols, cat_cols, is_train=True, y=y)
        n_samples, n_features = X.shape
        mtry = max(2, int(np.sqrt(n_features)) + 1)

        perm = rng.permutation(n_samples)
        cal_size = max(200, int(0.2 * n_samples))
        cal_idx, tr_idx = perm[:cal_size], perm[cal_size:]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]

        self.trees_ = []
        sub_size = min(len(X_tr), 15000)
        for _ in range(self.n_estimators):
            boot_idx = rng.choice(len(X_tr), size=sub_size, replace=True)
            tree = self._build_tree(X_tr[boot_idx], y_tr[boot_idx], 0, rng, mtry)
            self.trees_.append(tree)

        raw_cal_preds = np.mean([self._predict_tree(t, X_cal) for t in self.trees_], axis=0)
        eps = 1e-5
        logits = np.log(np.clip(raw_cal_preds, eps, 1.0 - eps) / (1.0 - np.clip(raw_cal_preds, eps, 1.0 - eps)))

        def nll(params: np.ndarray) -> float:
            a, b = params
            p = expit(a * logits + b)
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            return -float(np.mean(y_cal * np.log(p) + (1.0 - y_cal) * np.log(1.0 - p)))

        res = minimize(nll, np.array([1.0, 0.0]), method="BFGS")
        if res.success:
            self.platt_a_, self.platt_b_ = float(res.x[0]), float(res.x[1])
        return self

    def predict_proba(self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> np.ndarray:
        X = self._encode_matrix(df, num_cols, cat_cols, is_train=False)
        raw_preds = np.mean([self._predict_tree(t, X) for t in self.trees_], axis=0)
        eps = 1e-5
        logits = np.log(np.clip(raw_preds, eps, 1.0 - eps) / (1.0 - np.clip(raw_preds, eps, 1.0 - eps)))
        return expit(self.platt_a_ * logits + self.platt_b_)


class RotationAwareFlightRiskModel:
    CATEGORICAL_FEATURES = [
        "Reporting_Airline",
        "Origin",
        "Dest",
        "Season",
        "DepTimeBlock",
    ]

    SCHEDULE_NUMERIC_FEATURES = [
        "Month",
        "DayOfWeek",
        "dep_mins_midnight",
        "Distance",
        "CRSElapsedTime",
        "delay15_eb_prob",
        "cancel_eb_prob",
        "divert_eb_prob",
    ]

    ROTATION_NUMERIC_FEATURES = [
        "rotation_leg_num",
        "turnaround_mins",
        "prev_arr_delay",
        "prev_cancelled",
        "is_tight_evening_turn",
    ]

    def __init__(self, n_estimators: int = 120, max_depth: int = 12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models_with_rotation_: Dict[str, object] = {}
        self.models_without_rotation_: Dict[str, object] = {}

    def _build_sklearn_calibrated_rf(self, numeric_cols: List[str], cat_cols: List[str]):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numeric_cols),
                (
                    "cat",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    cat_cols,
                ),
            ]
        )
        base_rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=15,
            n_jobs=-1,
            random_state=self.random_state,
        )
        calibrated_clf = CalibratedClassifierCV(
            estimator=base_rf,
            method="sigmoid",
            cv=3,
        )
        return Pipeline([("prep", preprocessor), ("clf", calibrated_clf)])

    def fit(self, train_df: pd.DataFrame, targets: Tuple[str, ...] = ("ArrDel15", "Cancelled", "Diverted")) -> "RotationAwareFlightRiskModel":
        full_num_cols = self.SCHEDULE_NUMERIC_FEATURES + self.ROTATION_NUMERIC_FEATURES
        engine_label = "scikit-learn CalibratedClassifierCV(RandomForest)" if HAS_SKLEARN else "NumPy/SciPy Calibrated Random Forest"
        logger.info(f"Using ML Backend: {engine_label}")

        for target in targets:
            y_train = train_df[target].to_numpy(dtype=int)
            pos_count = int(np.sum(y_train == 1))
            if pos_count < 10:
                logger.warning(f"Target '{target}' has only {pos_count} positive samples; skipping RF fit.")
                continue

            logger.info(f"Training Calibrated Random Forest for {target} (positives={pos_count:,}/{len(y_train):,})...")

            if HAS_SKLEARN:
                pipe_no_rot = self._build_sklearn_calibrated_rf(
                    self.SCHEDULE_NUMERIC_FEATURES, self.CATEGORICAL_FEATURES
                )
                pipe_no_rot.fit(
                    train_df[self.SCHEDULE_NUMERIC_FEATURES + self.CATEGORICAL_FEATURES],
                    y_train,
                )
                self.models_without_rotation_[target] = pipe_no_rot

                pipe_rot = self._build_sklearn_calibrated_rf(
                    full_num_cols, self.CATEGORICAL_FEATURES
                )
                pipe_rot.fit(
                    train_df[full_num_cols + self.CATEGORICAL_FEATURES],
                    y_train,
                )
                self.models_with_rotation_[target] = pipe_rot
            else:
                rf_no_rot = _NumpyCalibratedRandomForest(
                    n_estimators=min(60, self.n_estimators),
                    max_depth=min(6, self.max_depth),
                    random_state=self.random_state,
                )
                rf_no_rot.fit(train_df, self.SCHEDULE_NUMERIC_FEATURES, self.CATEGORICAL_FEATURES, y_train)
                self.models_without_rotation_[target] = rf_no_rot

                rf_rot = _NumpyCalibratedRandomForest(
                    n_estimators=min(60, self.n_estimators),
                    max_depth=min(6, self.max_depth),
                    random_state=self.random_state,
                )
                rf_rot.fit(train_df, full_num_cols, self.CATEGORICAL_FEATURES, y_train)
                self.models_with_rotation_[target] = rf_rot

        return self

    def predict_proba_df(self, df: pd.DataFrame, use_rotation: bool = True) -> pd.DataFrame:
        models = self.models_with_rotation_ if use_rotation else self.models_without_rotation_
        num_cols = (
            self.SCHEDULE_NUMERIC_FEATURES + self.ROTATION_NUMERIC_FEATURES
            if use_rotation
            else self.SCHEDULE_NUMERIC_FEATURES
        )
        feature_cols = num_cols + self.CATEGORICAL_FEATURES

        preds = pd.DataFrame(index=df.index)
        for target, col_name in [
            ("ArrDel15", "prob_delay15"),
            ("Cancelled", "prob_cancel"),
            ("Diverted", "prob_divert"),
        ]:
            if target in models:
                model = models[target]
                if HAS_SKLEARN:
                    preds[col_name] = model.predict_proba(df[feature_cols])[:, 1]
                else:
                    preds[col_name] = model.predict_proba(df, num_cols, self.CATEGORICAL_FEATURES)
            else:
                preds[col_name] = 0.002
        return preds


def evaluate_backtest(
    test_df: pd.DataFrame,
    rf_model: RotationAwareFlightRiskModel,
) -> pd.DataFrame:
    logger.info("Evaluating models on chronologically held-out test set using ROC-AUC and Brier score...")
    preds_no_rot = rf_model.predict_proba_df(test_df, use_rotation=False)
    preds_rot = rf_model.predict_proba_df(test_df, use_rotation=True)

    target_specs = [
        ("ArrDel15", "delay15_raw_prob", "delay15_eb_prob", "prob_delay15"),
        ("Cancelled", "cancel_raw_prob", "cancel_eb_prob", "prob_cancel"),
        ("Diverted", "divert_raw_prob", "divert_eb_prob", "prob_divert"),
    ]

    rows = []
    for target_col, raw_col, eb_col, rf_col in target_specs:
        y_true = test_df[target_col].to_numpy(dtype=int)
        if len(np.unique(y_true)) < 2:
            continue

        candidates = [
            ("1. Historical Unpooled Cell Rate", test_df[raw_col].to_numpy(dtype=float)),
            ("2. Empirical-Bayes Beta-Binomial Shrinkage", test_df[eb_col].to_numpy(dtype=float)),
            ("3. Calibrated RF (Schedule Only, No Rotation)", preds_no_rot[rf_col].to_numpy(dtype=float)),
            ("4. Calibrated RF (With Aircraft Rotation Features)", preds_rot[rf_col].to_numpy(dtype=float)),
        ]

        for model_name, y_prob in candidates:
            y_prob_clipped = np.clip(np.nan_to_num(y_prob, nan=float(np.mean(y_true))), 0.0, 1.0)
            auc = compute_roc_auc(y_true, y_prob_clipped)
            brier = compute_brier_score(y_true, y_prob_clipped)
            rows.append(
                {
                    "Outcome": target_col,
                    "Model": model_name,
                    "ROC_AUC": round(auc, 4),
                    "Brier_Score": round(brier, 5),
                    "Test_Flights": int(len(y_true)),
                    "Base_Rate": round(float(np.mean(y_true)), 4),
                }
            )

    metrics_df = pd.DataFrame(rows)
    return metrics_df


# ==============================================================================
# 6. INNOVATION 3 SUPPORT: D3.JS UNCERTAINTY-AWARE UI ARTIFACT EXPORT (WITH MONTH)
# ==============================================================================

def _rows_to_ui_cells(df_cells: pd.DataFrame) -> List[dict]:
    ui_cells = []
    for _, row in df_cells.iterrows():
        p_del = float(row.get("rf_prob_delay15", row["delay15_eb_prob"]))
        p_can = float(row.get("rf_prob_cancel", row["cancel_eb_prob"]))
        p_div = float(row.get("rf_prob_divert", row["divert_eb_prob"]))

        n_cancel_icons = int(round(p_can * 100))
        n_divert_icons = int(round(p_div * 100))
        n_delay_icons = int(round(p_del * 100))
        n_ontime_icons = max(0, 100 - n_cancel_icons - n_divert_icons - n_delay_icons)

        m_num = int(row.get("Month", 0))
        ui_cells.append(
            {
                "route": row["Route"],
                "carrier": row["Reporting_Airline"],
                "season": row["Season"],
                "month": m_num,
                "month_name": MONTH_NAMES.get(m_num, "All Months"),
                "dep_time_block": row["DepTimeBlock"],
                "sample_size_n": int(row["n"]),
                "delay15_eb_prob": round(float(row["delay15_eb_prob"]), 4),
                "delay15_rf_prob": round(p_del, 4),
                "delay15_ci_lower": round(float(row["delay15_ci_lower"]), 4),
                "delay15_ci_upper": round(float(row["delay15_ci_upper"]), 4),
                "delay15_ci_width": round(float(row["delay15_ci_width"]), 4),
                "cancel_eb_prob": round(float(row["cancel_eb_prob"]), 4),
                "cancel_rf_prob": round(p_can, 4),
                "divert_eb_prob": round(float(row["divert_eb_prob"]), 4),
                "vsup_confidence": round(float(row["vsup_confidence"]), 4),
                "icon_array_100": {
                    "on_time": n_ontime_icons,
                    "delayed_15m": n_delay_icons,
                    "cancelled": n_cancel_icons,
                    "diverted": n_divert_icons,
                },
                "delay_causes": {
                    "carrier": round(float(row.get("share_CarrierDelay", 0.0)), 4),
                    "weather": round(float(row.get("share_WeatherDelay", 0.0)), 4),
                    "nas": round(float(row.get("share_NASDelay", 0.0)), 4),
                    "security": round(float(row.get("share_SecurityDelay", 0.0)), 4),
                    "late_aircraft": round(float(row.get("share_LateAircraftDelay", 0.0)), 4),
                },
            }
        )
    return ui_cells


def export_d3_ui_artifacts(
    eb_estimator: HierarchicalEmpiricalBayesShrinkage,
    full_df_with_eb: pd.DataFrame,
    rf_model: RotationAwareFlightRiskModel,
    metrics_df: pd.DataFrame,
    output_dir: Path,
    max_monthly_cells_json: int = 25000,
) -> Dict[str, Path]:
    """
    Exports pre-aggregated JSON and CSV files for the D3.js interface (including Month-of-Year):
      - `ui_route_map.json`: Route-level summary (overall + monthly breakdown)
      - `ui_carrier_time_matrix.json`: Carrier x Departure-Time matrix cells with `month` (0=All, 1..12=Jan..Dec)
      - `backtest_evaluation_metrics.csv` & `.json`
    """
    logger.info(f"Exporting month-aware D3.js visualization payloads to {output_dir}...")
    monthly_cell_df = eb_estimator.cell_summary_df_.copy()
    annual_cell_df = eb_estimator.annual_cell_summary_df_.copy()

    rf_preds = rf_model.predict_proba_df(full_df_with_eb, use_rotation=True)
    tmp = full_df_with_eb[["Route", "Reporting_Airline", "Season", "Month", "DepTimeBlock"]].copy()
    tmp["rf_prob_delay15"] = rf_preds["prob_delay15"]
    tmp["rf_prob_cancel"] = rf_preds["prob_cancel"]
    tmp["rf_prob_divert"] = rf_preds["prob_divert"]

    # Attach RF predictions to Monthly cells
    cell_rf_monthly = (
        tmp.groupby(["Route", "Reporting_Airline", "Season", "Month", "DepTimeBlock"], observed=True)[
            ["rf_prob_delay15", "rf_prob_cancel", "rf_prob_divert"]
        ]
        .mean()
        .reset_index()
    )
    monthly_cell_df = monthly_cell_df.merge(
        cell_rf_monthly,
        on=["Route", "Reporting_Airline", "Season", "Month", "DepTimeBlock"],
        how="left",
    ).sort_values("n", ascending=False).reset_index(drop=True)

    # Attach RF predictions to Annual (Month=0) cells
    cell_rf_annual = (
        tmp.groupby(["Route", "Reporting_Airline", "DepTimeBlock"], observed=True)[
            ["rf_prob_delay15", "rf_prob_cancel", "rf_prob_divert"]
        ]
        .mean()
        .reset_index()
    )
    annual_cell_df = annual_cell_df.merge(
        cell_rf_annual,
        on=["Route", "Reporting_Airline", "DepTimeBlock"],
        how="left",
    ).sort_values("n", ascending=False).reset_index(drop=True)

    combined_csv_df = pd.concat([annual_cell_df, monthly_cell_df], ignore_index=True)
    cells_csv_path = output_dir / "bts_cell_risk_estimates.csv"
    combined_csv_df.to_csv(cells_csv_path, index=False)

    # Keep all carriers and time blocks for the top 150 routes so no competing carrier is truncated
    top_routes = (
        annual_cell_df.groupby("Route", observed=True)["n"]
        .sum()
        .nlargest(150)
        .index
    )
    annual_top = annual_cell_df[annual_cell_df["Route"].isin(top_routes)].copy()
    monthly_top = monthly_cell_df[monthly_cell_df["Route"].isin(top_routes)].copy()
    if len(monthly_top) > max_monthly_cells_json:
        # Preserve proportional carrier coverage per route across months
        monthly_top = (
            monthly_top.groupby(["Route", "Reporting_Airline"], observed=True, group_keys=False)
            .apply(lambda g: g.nlargest(max(4, max_monthly_cells_json // max(1, len(top_routes) * 4)), "n"))
            .reset_index(drop=True)
        )

    ui_cells = (
        _rows_to_ui_cells(annual_top)
        + _rows_to_ui_cells(monthly_top)
    )

    matrix_json_path = output_dir / "ui_carrier_time_matrix.json"
    with open(matrix_json_path, "w", encoding="utf-8") as f:
        json.dump({"cells": ui_cells}, f, indent=2)

    # Route-level summary for D3 US origin-destination map
    route_df = eb_estimator.level_tables_["ArrDel15"][0].copy()
    route_df[["Origin", "Dest"]] = route_df["Route"].str.split("-", expand=True)
    route_records = route_df[
        ["Route", "Origin", "Dest", "n", "raw_rate", "eb_prob", "ci_lower", "ci_upper", "ci_width"]
    ].to_dict(orient="records")

    route_json_path = output_dir / "ui_route_map.json"
    with open(route_json_path, "w", encoding="utf-8") as f:
        json.dump({"routes": route_records}, f, indent=2)

    metrics_csv_path = output_dir / "backtest_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    metrics_json_path = output_dir / "backtest_evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({"backtest_metrics": metrics_df.to_dict(orient="records")}, f, indent=2)

    return {
        "cells_csv": cells_csv_path,
        "matrix_json": matrix_json_path,
        "route_json": route_json_path,
        "metrics_csv": metrics_csv_path,
        "metrics_json": metrics_json_path,
    }


# ==============================================================================
# 7. MAIN CLI & EXECUTION ENTRYPOINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CS6242 Team 185 — BTS Flight Disruption Risk Model & Data Pipeline"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2024],
        help="BTS Year(s) to download and model (e.g. --years 2024)",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=list(range(1, 13)),
        help="BTS Month(s) (1-12) to download (e.g. --months 1 2 3 ... 12). Final month is held out for backtesting.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to cache downloaded BTS ZIP and Parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to export trained model predictions, backtest metrics, and D3.js JSON payloads",
    )
    parser.add_argument(
        "--use-spark",
        action="store_true",
        help="Use Apache Spark (PySpark) for Parquet ingestion and rotation reconstruction",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run pipeline on realistic 12-month synthetic BTS rotation data (no internet download needed)",
    )
    parser.add_argument(
        "--demo-flights",
        type=int,
        default=60000,
        help="Number of synthetic flights to generate when --demo is enabled",
    )
    return parser.parse_args()


def run_pipeline(cfg: PipelineConfig) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    if cfg.demo_mode:
        raw_df = generate_synthetic_bts_dataset(n_flights=cfg.demo_n_flights, seed=cfg.random_state)
        demo_parquet = cfg.parquet_dir / "bts_synthetic_demo.parquet"
        raw_df.to_parquet(demo_parquet, index=False)
        parquet_files = [demo_parquet]
    else:
        parquet_files = []
        for yr in cfg.years:
            for mo in cfg.months:
                p_path = fetch_bts_month(yr, mo, cfg.raw_dir, cfg.parquet_dir)
                parquet_files.append(p_path)

    if cfg.use_spark:
        df = rebuild_rotations_spark(parquet_files)
    else:
        dfs = [pd.read_parquet(p) for p in parquet_files]
        df = rebuild_rotations_pandas(pd.concat(dfs, ignore_index=True))

    df = df.sort_values("FlightDate").reset_index(drop=True)
    unique_months = sorted(df["Month"].unique())
    if len(unique_months) >= 2:
        holdout_month = unique_months[-1]
        train_mask = df["Month"] < holdout_month
        test_mask = df["Month"] == holdout_month
        logger.info(
            f"Chronological Split: Training on months {[int(m) for m in unique_months[:-1]]} ({train_mask.sum():,} flights), "
            f"Backtesting on held-out month {int(holdout_month)} ({test_mask.sum():,} flights)."
        )
    else:
        split_idx = int(len(df) * 0.8)
        cutoff_date = df.loc[split_idx, "FlightDate"]
        train_mask = df["FlightDate"] < cutoff_date
        test_mask = ~train_mask
        logger.info(
            f"Chronological Split (single month): Training on < {cutoff_date} ({train_mask.sum():,} flights), "
            f"Backtesting on >= {cutoff_date} ({test_mask.sum():,} flights)."
        )

    train_df = df.loc[train_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)

    eb_estimator = HierarchicalEmpiricalBayesShrinkage()
    eb_estimator.fit(train_df)

    train_eb = eb_estimator.transform_flight_features(train_df)
    test_eb = eb_estimator.transform_flight_features(test_df)

    rf_model = RotationAwareFlightRiskModel(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        random_state=cfg.random_state,
    )
    rf_model.fit(train_eb)

    metrics_df = evaluate_backtest(test_eb, rf_model)
    print("\n" + "=" * 88)
    print("CS6242 TEAM 185 — CHRONOLOGICAL BACKTEST EVALUATION (AUC & BRIER SCORE)")
    print("=" * 88)
    print(metrics_df.to_string(index=False))
    print("=" * 88 + "\n")

    # Refit EB on full dataset so all 12 months are exported to the UI JSON
    eb_estimator.fit(df)
    full_eb = eb_estimator.transform_flight_features(df)
    artifacts = export_d3_ui_artifacts(
        eb_estimator=eb_estimator,
        full_df_with_eb=full_eb,
        rf_model=rf_model,
        metrics_df=metrics_df,
        output_dir=cfg.output_dir,
    )
    logger.info("Pipeline completed! Exported artifacts:")
    for name, path in artifacts.items():
        logger.info(f"  - {name}: {path.resolve()}")

    return metrics_df, artifacts


if __name__ == "__main__":
    args = parse_args()
    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        years=args.years,
        months=args.months,
        use_spark=args.use_spark,
        demo_mode=args.demo,
        demo_n_flights=args.demo_flights,
    )
    run_pipeline(config)