import pandas as pd
import numpy as np
import re
from typing import Optional
import os
# ------- Path to your data -------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "cars_data_normalized.csv")
# ---------- CONFIG ----------
#DATA_CSV = "cars_data_normalized.csv"
FUEL_CATEGORIES = ["Petrol", "Diesel", "Hybrid", "CNG"]


def normalize_series(s: pd.Series) -> pd.Series:
    """Normalize numeric series to [0,1], robust to NaNs and constant columns."""
    s = pd.to_numeric(s, errors="coerce")
    if s.isnull().all():
        return pd.Series(0.0, index=s.index)
    s_filled = s.fillna(s.median())
    mn, mx = s_filled.min(), s_filled.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s_filled - mn) / (mx - mn)

def map_fuel_to_category(s: str) -> str:
    """Map manufacturer fuel strings to the 4 chosen categories."""
    if s is None:
        return "Petrol"
    stt = str(s).lower()
    if "cng" in stt:
        return "CNG"
    if "diesel" in stt or "deisel" in stt:
        return "Diesel"
    if "hybrid" in stt or "phev" in stt or "plug-in" in stt or "plug in" in stt or "mild" in stt:
        return "Hybrid"
    if "petrol" in stt or "gasoline" in stt or "turbo" in stt or "petrol " in stt or "petrol(" in stt:
        return "Petrol"
    return "Petrol"

def parse_budget_input(s: str) -> Optional[int]:
    """Parse budget inputs like '10', '10.5', '1.2cr', '100k', '₹12.5' into INR integer."""
    if s is None:
        return None
    s0 = str(s).strip().lower().replace(",", "").replace("₹", "")
    if s0 == "":
        return None
    # crores
    if any(x in s0 for x in ("cr", "crore")):
        m = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s0)
        if not m: return None
        try: return int(round(float(m[0]) * 10000000))
        except: return None
    # lakhs
    if any(x in s0 for x in ("lakh","lakhs")) and not s0.endswith("k"):
        m = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s0)
        if not m: return None
        try: return int(round(float(m[0]) * 100000))
        except: return None
    # trailing 'L' meaning lakhs
    if s0.endswith("l") and not s0.endswith("k"):
        try:
            v = float(s0[:-1]); return int(round(v * 100000))
        except: pass
    # thousands
    if s0.endswith("k"):
        try: return int(round(float(s0[:-1]) * 1000))
        except: return None
    m = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s0)
    if not m: return None
    try:
        val = float(m[0])
    except:
        return None
    if val >= 100000: return int(round(val))
    if 1000 < val < 100000: return int(round(val))
    return int(round(val * 100000))

def format_price(inr):
    """Format INR integer into readable string (Cr / L)."""
    try:
        if pd.isna(inr) or inr is None: return "N/A"
        inr = int(inr)
        if inr >= 10000000: return f"₹{round(inr/10000000,2)} Cr"
        if inr >= 100000: return f"₹{round(inr/100000,1)} L"
        return f"₹{inr}"
    except:
        return str(inr)

#load data
def load_data(DATA_PATH) -> pd.DataFrame:
    """Load dataset and prepare normalized columns (fuel_category, body_type_clean, numeric casts)."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["fuel_raw"] = df.get("fuel_type", df.get("original_fuel_type", pd.Series(dtype=object))).astype(str)
    df["fuel_category"] = df["fuel_raw"].apply(map_fuel_to_category)
    df["body_type_clean"] = df.get("body_type", df.get("original_body_type", pd.Series(dtype=object))).astype(str).str.strip()
    # cast commonly used numeric columns
    for c in ["price_inr","seating_capacity_num","mileage_value","power_bhp","engine_cc_num","airbags_num","adas_level_num","service_cost_per_year_avg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

#calculating axes
def compute_axes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized axes for each car:
    performance, economy, safety, comfort, ownership, price.
    Seating intentionally excluded from scoring (treated as primary filter).
    """
    df = df.copy()
    for col in ["power_bhp","torque_nm","top_speed_kmph","mileage_value","range_km_est",
                "airbags_num","adas_level_num","service_cost_per_year_avg",
                "ground_clearance_mm","sunroof_yes","cruise_control_yes","price_inr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    perf_cols = [c for c in ["power_bhp","torque_nm","top_speed_kmph"] if c in df.columns]
    perf_norm = pd.concat([normalize_series(df[c]) for c in perf_cols], axis=1).mean(axis=1) if perf_cols else pd.Series(0.0, index=df.index)
    eco_cols = [c for c in ["mileage_value","range_km_est"] if c in df.columns]
    eco_norm = pd.concat([normalize_series(df[c]) for c in eco_cols], axis=1).mean(axis=1) if eco_cols else pd.Series(0.0, index=df.index)
    safety_cols = [c for c in ["airbags_num","adas_level_num"] if c in df.columns]
    safety_norm = pd.concat([normalize_series(df[c]) for c in safety_cols], axis=1).mean(axis=1) if safety_cols else pd.Series(0.0, index=df.index)
    comfort_cols = [c for c in ["sunroof_yes","cruise_control_yes","ground_clearance_mm"] if c in df.columns]
    comfort_norm = pd.concat([normalize_series(df[c]) for c in comfort_cols], axis=1).mean(axis=1) if comfort_cols else pd.Series(0.0, index=df.index)
    ownership_norm = (1.0 - normalize_series(df["service_cost_per_year_avg"].fillna(df["service_cost_per_year_avg"].median()))) if "service_cost_per_year_avg" in df.columns else pd.Series(0.0, index=df.index)
    price_axis = (1.0 - normalize_series(df["price_inr"].fillna(df["price_inr"].median()))) if "price_inr" in df.columns else pd.Series(0.0, index=df.index)

    axes = pd.DataFrame({
        "axis_performance": perf_norm,
        "axis_economy": eco_norm,
        "axis_safety": safety_norm,
        "axis_comfort": comfort_norm,
        "axis_ownership": ownership_norm,
        "axis_price": price_axis
    }, index=df.index)

    for col in axes.columns:
        df[col] = axes[col]
    return df

# recommending cars
def recommend(df: pd.DataFrame, weights: dict, top_n: int = 20, filters: Optional[dict] = None,
              soft_fuel: bool = False, soft_body: bool = False,
              selected_fuel: Optional[str] = None, selected_body: Optional[str] = None,
              soft_fuel_weight: float = 0.0, soft_body_weight: float = 0.0) -> pd.DataFrame:

    D = df.copy()
    # ensure axes present
    if not all(ax in D.columns for ax in ["axis_performance","axis_economy","axis_safety","axis_comfort","axis_ownership","axis_price"]):
        D = compute_axes(D)

    extra_axis_names = []
    if soft_fuel and selected_fuel:
        D["axis_fuel_pref"] = (D.get("fuel_category", pd.Series(dtype=object)).fillna("").astype(str) == selected_fuel).astype(float)
        extra_axis_names.append("axis_fuel_pref")
    if soft_body and selected_body:
        D["body_clean"] = D.get("body_type", D.get("original_body_type", pd.Series())).astype(str).str.strip().str.lower()
        sel_body = str(selected_body).strip().lower()
        D["axis_body_pref"] = D["body_clean"].str.contains(re.escape(sel_body), na=False).astype(float)
        extra_axis_names.append("axis_body_pref")

    base_axes = ["axis_performance","axis_economy","axis_safety","axis_comfort","axis_ownership","axis_price"]
    axis_cols = base_axes + extra_axis_names

    weight_keys = ["performance","economy","safety","comfort","ownership","price"]
    if "axis_fuel_pref" in axis_cols:
        weight_keys.append("fuel_pref")
    if "axis_body_pref" in axis_cols:
        weight_keys.append("body_pref")

    w = np.array([weights.get(k, 0.0) for k in weight_keys], dtype=float)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()

    D["final_score"] = 0.0
    for i, col in enumerate(axis_cols):
        D["final_score"] += w[i] * D[col].fillna(0)

    # soft penalty for price above max budget if provided
    if filters and filters.get("max_budget"):
        maxb = filters["max_budget"]
        D["budget_penalty"] = D["price_inr"].apply(lambda p: 0 if pd.isna(p) else max(0, (p - maxb) / (maxb + 1)))
        D["final_score"] = D["final_score"] - 0.5 * D["budget_penalty"]

    D = D.sort_values("final_score", ascending=False).reset_index(drop=True)
    return D.head(top_n)


