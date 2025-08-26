import math
import random
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd


PARQUET_PATH = "/Users/cassidy.hilton/Cursor Projects/unitFocus/data/FCT_PROMO_HOTLIST_FULL.parquet"

# App thresholds
RATIO = 0.45
FLOOR = 0.25
UNDER = 0.98
ON = 1.01

# Coverage window: today-120d .. today+30d
TODAY = pd.Timestamp.today().normalize()
COVERAGE_START = TODAY - pd.Timedelta(days=120)
COVERAGE_END = TODAY + pd.Timedelta(days=30)

# Target distribution
TARGET_SHARE = {
    "Replenish": 0.25,      # low + under
    "Fix": 0.25,            # good + under
    "Maintain": 0.25,       # good + on
    "Investigate": 0.25     # low + near/on
}


def choose_expected_by_category(cat: str, region: str) -> float:
    base_low, base_high = 300, 5000
    # Gentle category-specific ranges
    ranges = {
        "Specialty Beverages": (800, 3000),
        "Beverage": (500, 2000),
        "Health & Wellness": (700, 4000),
        "Meat & Seafood": (1200, 5000),
        "Bakery": (300, 1500),
        "Frozen Foods": (600, 2500),
        "Snacks": (300, 1800),
        "Candy & Confectionery": (300, 1500),
        "Breakfast & Cereal": (400, 1600),
        "Pantry": (400, 2000),
        "Produce": (300, 1200),
        "Dairy & Eggs": (400, 1800),
        "Personal Care": (500, 2500),
        "Household Cleaning": (500, 2200),
        "Pet Care": (600, 2600),
        "Baby Care": (700, 3000),
    }
    lo, hi = ranges.get(cat, (base_low, base_high))
    # mild regional modulation
    reg_factor = {
        "Mountain": 1.0,
        "Northeast": 1.07,
        "South": 0.97,
        "West": 1.03,
    }.get(region, 1.0)
    val = np.random.uniform(lo, hi) * reg_factor
    return float(np.clip(val, base_low, base_high))


def enforce_date_coverage(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure DATE column exists
    if "DATE" not in df.columns:
        df["DATE"] = pd.to_datetime(df["LAST_SNAPSHOT_AT"]).dt.date.astype(str)

    # Ensure daily coverage by setting LAST_SNAPSHOT_AT and DATE round-robin across the window
    date_range = pd.date_range(COVERAGE_START, COVERAGE_END, freq="D")
    need_days = set(date_range.date)
    have_days = set(pd.to_datetime(df["LAST_SNAPSHOT_AT"]).dt.date.unique())
    missing = sorted(list(need_days - have_days))
    if not missing:
        return df

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    take = min(len(missing), len(df))
    ts_16 = pd.to_datetime("16:00:00").time()
    for i in range(take):
        d = pd.Timestamp(missing[i])
        dt = pd.Timestamp.combine(d, ts_16)
        df.at[i, "LAST_SNAPSHOT_AT"] = dt
        df.at[i, "DATE"] = d.date().isoformat()
    return df


def assign_quadrants(n: int) -> np.ndarray:
    counts = {k: int(v * n) for k, v in TARGET_SHARE.items()}
    # fix rounding drift
    rem = n - sum(counts.values())
    keys = list(counts.keys())
    i = 0
    while rem > 0:
        counts[keys[i % len(keys)]] += 1
        rem -= 1
        i += 1
    labels = (
        ["Replenish"] * counts["Replenish"] +
        ["Fix"] * counts["Fix"] +
        ["Maintain"] * counts["Maintain"] +
        ["Investigate"] * counts["Investigate"]
    )
    rng = np.random.default_rng(42)
    rng.shuffle(labels)
    return np.array(labels)


def _classify_row(row: pd.Series) -> str:
    ratio = RATIO
    floor = FLOOR
    under = UNDER
    on = ON
    stock_now = int(row["ONHAND_LATEST"]) + int(row["BACKROOM_START"])
    stock_start = max(1, int(row["ONHAND_START"]) + int(row["BACKROOM_START"]))
    inv_ratio = stock_now / stock_start if stock_start else 0.0
    low = (inv_ratio < ratio) and (stock_now < max(3, int(math.floor(floor * stock_start))))
    pct = float(row.get("PCT_TO_PLAN_SO_FAR", np.nan))
    if not np.isfinite(pct) or pct == 0:
        exp = float(row.get("EXPECTED_REVENUE_SO_FAR_USD", 0.0))
        act = float(row.get("REVENUE_SO_FAR_USD", 0.0))
        pct = (act / exp) if exp else 0.0
    underp = pct <= under
    onplan = pct >= on
    if low and underp:
        return "Replenish"
    if (not low) and underp:
        return "Fix"
    if (not low) and onplan:
        return "Maintain"
    if low and (not underp) and (not onplan):
        return "Investigate"
    return "Other"


def apply_inventory_and_perf(row: pd.Series, quadrant: str) -> pd.Series:
    # Determine inventory regime
    is_low = quadrant in ("Replenish", "Investigate")
    # Inventory bands
    if is_low:
        onhand_start = int(np.random.randint(2, 16))
        backroom_start = int(np.random.randint(0, 11))
        if quadrant == "Replenish":
            inv_ratio = float(np.random.uniform(0.10, 0.35))
        else:  # Investigate
            inv_ratio = float(np.random.uniform(0.15, 0.40))
    else:
        onhand_start = int(np.random.randint(8, 41))
        backroom_start = int(np.random.randint(5, 26))
        if quadrant == "Fix":
            inv_ratio = float(np.random.uniform(0.60, 1.10))
        else:  # Maintain
            inv_ratio = float(np.random.uniform(0.70, 1.20))

    stock_start = max(1, onhand_start + backroom_start)

    # Target stock_now from inv_ratio with constraints
    desired = int(round(inv_ratio * stock_start))
    thresh = int(max(3, math.floor(FLOOR * stock_start)))
    if is_low:
        stock_now = min(desired, max(0, thresh - 1))
    else:
        stock_now = max(desired, thresh)
    stock_now = int(max(0, min(stock_now, stock_start)))

    # Compose stock_now into components honoring ranges
    if is_low:
        # prefer backroom up to 0-10; put remainder on shelf limited by 0-15
        max_br = 10
        backroom_start = int(min(max_br, stock_now))
        onhand_latest = int(max(0, stock_now - backroom_start))
        onhand_latest = int(min(onhand_latest, onhand_start))
        # if still too large to fit onhand_start, cap and push to backroom within max
        if onhand_latest + backroom_start != stock_now:
            extra = stock_now - (onhand_latest + backroom_start)
            add_br = min(max_br - backroom_start, max(0, extra))
            backroom_start += add_br
            onhand_latest = max(0, stock_now - backroom_start)
            onhand_latest = min(onhand_latest, onhand_start)
    else:
        # choose backroom in 5-25 but not exceeding stock_now
        min_br, max_br = 5, 25
        backroom_start = int(min(max_br, max(min_br, stock_now // 3)))
        backroom_start = int(min(backroom_start, stock_now))
        onhand_latest = int(max(0, stock_now - backroom_start))
        if onhand_latest > onhand_start:
            onhand_start = onhand_latest

    # Baseline units ~ onhand_start ± 20%
    bu_mu = onhand_start
    bu_sigma = max(1.0, 0.20 * bu_mu)
    baseline_units = int(np.clip(np.random.normal(bu_mu, bu_sigma), 1, bu_mu * 2))

    # Performance bands
    if quadrant == "Replenish":
        pct = float(np.random.uniform(0.75, 0.93))
    elif quadrant == "Fix":
        pct = float(np.random.uniform(0.85, 0.97))
    elif quadrant == "Maintain":
        pct = float(np.random.uniform(1.02, 1.10))
    else:  # Investigate
        pct = float(np.random.uniform(0.98, 1.04))

    expected = choose_expected_by_category(str(row.get("CATEGORY", "")), str(row.get("REGION", "")))
    actual = pct * expected
    # Add ±10% noise to actual; then clamp pct back into band
    noise = np.random.uniform(-0.10, 0.10)
    actual *= (1.0 + noise)

    # Recompute pct and clamp within bands
    pct = float(np.clip(actual / max(expected, 1e-6),
                        0.75 if quadrant == "Replenish" else 0.85 if quadrant == "Fix" else 1.02 if quadrant == "Maintain" else 0.98,
                        0.93 if quadrant == "Replenish" else 0.97 if quadrant == "Fix" else 1.10 if quadrant == "Maintain" else 1.04))
    actual = pct * expected

    # Write back allowed fields only
    row["ONHAND_START"] = int(onhand_start)
    row["BACKROOM_START"] = int(backroom_start)
    row["ONHAND_LATEST"] = int(onhand_latest)
    row["BASELINE_UNITS"] = int(baseline_units)

    row["EXPECTED_REVENUE_SO_FAR_USD"] = float(round(expected, 2))
    row["REVENUE_SO_FAR_USD"] = float(round(actual, 2))
    row["PCT_TO_PLAN_SO_FAR"] = float(round(pct, 4))

    # Final enforcement to ensure quadrant classification
    # Adjust small deltas if needed
    for _ in range(3):
        label = _classify_row(row)
        stock_now = int(row["ONHAND_LATEST"]) + int(row["BACKROOM_START"]) 
        stock_start = max(1, int(row["ONHAND_START"]) + int(row["BACKROOM_START"]))
        inv_ratio = stock_now / stock_start if stock_start else 0
        thresh = int(max(3, math.floor(FLOOR * stock_start)))
        pct = float(row["PCT_TO_PLAN_SO_FAR"])
        if label == quadrant:
            break
        if quadrant == "Replenish":
            # force low + under
            if not (inv_ratio < RATIO and stock_now < thresh):
                stock_now = max(0, min(thresh - 1, stock_start - 1))
                row["BACKROOM_START"] = min(int(row["BACKROOM_START"]), stock_now)
                row["ONHAND_LATEST"] = max(0, stock_now - int(row["BACKROOM_START"]))
                row["ONHAND_START"] = max(2, min(15, int(row["ONHAND_START"])) )
            if pct > UNDER:
                pct = float(np.random.uniform(0.75, 0.93))
        elif quadrant == "Fix":
            # good + under
            if not (stock_now >= thresh and inv_ratio >= RATIO):
                stock_now = max(thresh, int(round(0.7 * stock_start)))
                br = max(5, min(25, stock_now // 3))
                row["BACKROOM_START"] = min(br, stock_now)
                row["ONHAND_LATEST"] = stock_now - int(row["BACKROOM_START"])
                row["ONHAND_START"] = max(8, int(row["ONHAND_LATEST"]))
            if pct > UNDER:
                pct = float(np.random.uniform(0.85, 0.97))
        elif quadrant == "Maintain":
            # good + on
            if not (stock_now >= thresh and inv_ratio >= RATIO):
                stock_now = max(thresh, int(round(0.9 * stock_start)))
                br = max(5, min(25, stock_now // 3))
                row["BACKROOM_START"] = min(br, stock_now)
                row["ONHAND_LATEST"] = stock_now - int(row["BACKROOM_START"])
                row["ONHAND_START"] = max(8, int(row["ONHAND_LATEST"]))
            if pct < ON:
                pct = float(np.random.uniform(1.02, 1.10))
        else:  # Investigate
            # low + near/on
            if not (inv_ratio < RATIO and stock_now < thresh):
                stock_now = max(0, min(thresh - 1, int(round(0.3 * stock_start))))
                row["BACKROOM_START"] = min(int(row["BACKROOM_START"]), stock_now)
                row["ONHAND_LATEST"] = max(0, stock_now - int(row["BACKROOM_START"]))
                row["ONHAND_START"] = max(2, min(15, int(row["ONHAND_START"])) )
            pct = float(np.random.uniform(0.98, 1.04))
        expected = float(row["EXPECTED_REVENUE_SO_FAR_USD"])
        row["REVENUE_SO_FAR_USD"] = float(round(pct * expected, 2))
        row["PCT_TO_PLAN_SO_FAR"] = float(round(pct, 4))
    return row


def main():
    df = pd.read_parquet(PARQUET_PATH)
    if not np.issubdtype(df["LAST_SNAPSHOT_AT"].dtype, np.datetime64):
        df["LAST_SNAPSHOT_AT"] = pd.to_datetime(df["LAST_SNAPSHOT_AT"])

    # Ensure date coverage
    df = enforce_date_coverage(df)

    # Ensure DATE equals LAST_SNAPSHOT_AT (string YYYY-MM-DD)
    df["DATE"] = pd.to_datetime(df["LAST_SNAPSHOT_AT"]).dt.strftime("%Y-%m-%d")

    n = len(df)
    quadrants = assign_quadrants(n)

    # Apply per-row adjustments with exact quadrant enforcement
    records = []
    for i, row in enumerate(df.itertuples(index=False)):
        s = pd.Series(row._asdict())
        s = apply_inventory_and_perf(s, quadrants[i])
        # Final safety: ensure label matches
        lbl = _classify_row(s)
        if lbl != quadrants[i]:
            s = apply_inventory_and_perf(s, quadrants[i])
        records.append(s)
    out = pd.DataFrame(records)

    # Compute REVENUE_FORECASTED_USD to mimic a time-series prediction with slight up/down deviations
    # Base on paced full-day estimate then apply weekly seasonality, category drift, and small Gaussian noise
    if "DAY_ELAPSED_SHARE" in out.columns:
        share = out["DAY_ELAPSED_SHARE"].astype(float).replace(0, 1.0).clip(lower=0.1, upper=1.0)
    else:
        share = pd.Series(1.0, index=out.index)

    paced = (out["REVENUE_SO_FAR_USD"] / share).fillna(out["REVENUE_SO_FAR_USD"]).astype(float)
    ts = pd.to_datetime(out["LAST_SNAPSHOT_AT"]) if not np.issubdtype(out["LAST_SNAPSHOT_AT"].dtype, np.datetime64) else out["LAST_SNAPSHOT_AT"]
    dow = ts.dt.dayofweek
    season = 1.0 + 0.035 * np.sin(2 * np.pi * dow / 7.0)

    # Stable small drift per category
    rs = np.random.RandomState(42)
    cats = out["CATEGORY"].astype(str).fillna("NA").unique()
    drift_map = {c: rs.uniform(-0.02, 0.02) for c in cats}
    drift = out["CATEGORY"].astype(str).map(drift_map).fillna(0.0).values

    noise = rs.normal(0.0, 0.03, size=len(out))
    factor = (1.0 + drift) * season * (1.0 + noise)
    factor = pd.Series(factor, index=out.index).clip(0.92, 1.12)

    forecast = (paced * factor).clip(lower=0).round(2)
    out["REVENUE_FORECASTED_USD"] = forecast

    # Preserve original column order and types as much as possible
    # If DATE wasn't in the original schema, append it at the end to keep the app working
    cols = df.columns.tolist()
    if "DATE" not in cols:
        cols = cols + ["DATE"]
    out = out[cols]

    # Save back to parquet (overwrite)
    out.to_parquet(PARQUET_PATH)


if __name__ == "__main__":
    np.random.seed(42)
    main()


