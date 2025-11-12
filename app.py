import streamlit as st
import pandas as pd
import numpy as np
import io
import textwrap
import re
from collections import OrderedDict

import expert_system as es  # expert_system.py

st.set_page_config(page_title="Car Recommender — Polished UI", layout="wide", initial_sidebar_state="expanded")

DATA_CSV = es.DATA_CSV
VALID_IMAGE_KEYS = ("image_url", "photo_url", "img", "thumbnail")
FUEL_CATEGORIES = es.FUEL_CATEGORIES

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .header { display:flex; align-items:center; gap:16px; }
    .hero { padding:18px; border-radius:12px; background: linear-gradient(90deg, rgba(37,99,235,0.06), rgba(124,58,237,0.03)); margin-bottom: 12px; }
    .muted { color: #6b7280; font-size:13px; }
    .card { padding:12px; border-radius:12px; background: linear-gradient(180deg,#0f1724, #071018); color: #e6eef8; box-shadow: 0 6px 18px rgba(15,23,42,0.06); border:1px solid rgba(255,255,255,0.02); transition: transform .12s ease, box-shadow .12s ease; }
    .card:hover { transform: translateY(-6px); box-shadow: 0 20px 40px rgba(15,23,42,0.10); }
    .card-title { font-weight:600; font-size:16px; margin:0; color: #e6eef8; }
    .spec { font-size:13px; color:#c7d2e0; }
    .pill { display:inline-block; padding:6px 10px; border-radius:999px; font-size:12px; background:rgba(255,255,255,0.04); color:#9be7b5; font-weight:600; }
    .score-bar { height:8px; border-radius:6px; background:linear-gradient(90deg,#ff8a00,#ff5e62); }
    .result-grid { gap: 16px; }
    .btn-flat { background:transparent; border:1px solid rgba(255,255,255,0.06); padding:8px 12px; border-radius:8px; color:#e6eef8; }
    </style>
    """,
    unsafe_allow_html=True
)

# cache the dataset load in Streamlit
@st.cache_data
def cached_load_data():
    return es.load_data()


df_raw = None
with st.spinner("Loading data..."):
    try:
        df_raw = cached_load_data()
    except Exception as e:
        st.error(f"Failed to load '{DATA_CSV}': {e}")
        st.stop()

# ---------- SESSION STATE ----------
if "shortlist" not in st.session_state:
    st.session_state["shortlist"] = []
if "compare" not in st.session_state:
    st.session_state["compare"] = []

def add_to_shortlist(uid):
    if uid not in st.session_state["shortlist"]:
        st.session_state["shortlist"].append(uid)

def remove_from_shortlist(uid):
    if uid in st.session_state["shortlist"]:
        st.session_state["shortlist"].remove(uid)

def toggle_compare(uid):
    if uid in st.session_state["compare"]:
        st.session_state["compare"].remove(uid)
    else:
        if len(st.session_state["compare"]) >= 3:
            st.warning("Compare limited to 3 cars.")
        else:
            st.session_state["compare"].append(uid)

def reset_filters():
    for k in ["min_budget_input","max_budget_input","fuel_choice","body_choice","seating_choice","fuel_mode","body_mode"]:
        if k in st.session_state:
            del st.session_state[k]

# UI content 
st.markdown('<div class="header"><h1 style="margin:0">🚗Expert System for Personalized Car Recommendation</h1><div class="muted" style="margin-left:8px">Budget-first • Soft/hard fuel & body preferences • Shortlist & compare</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div style="display:flex;justify-content:space-between;align-items:center"><div><strong style="font-size:18px">Refined UI — quicker decisions</strong><div class="muted">Use primary filters to reduce corpus, then tune priorities to rank cars.</div></div><div style="text-align:right"><small class="muted">Tip: try soft preference first to avoid overfiltering.</small></div></div></div>', unsafe_allow_html=True)

# SIDEBAR: Primary filters
st.sidebar.header("Primary filters")
st.sidebar.markdown("Enter budgets like: `10` (→ 10 L), `10.5`, `100k`, `1.2cr`, `₹12.5`")
min_budget_input = st.sidebar.text_input("Min budget (optional)", value="", key="min_budget_input")
max_budget_input = st.sidebar.text_input("Max budget (optional)", value="", key="max_budget_input")
fuel_choice = st.sidebar.selectbox("Fuel type (choose or leave blank)", options=[""] + FUEL_CATEGORIES, index=0, key="fuel_choice")
body_options = [""] + sorted(df_raw.get("body_type_clean", pd.Series(dtype=object)).dropna().unique().tolist())
body_choice = st.sidebar.selectbox("Body type (choose or leave blank)", options=body_options, index=0, key="body_choice")

# seating dropdown (exact match) — build options from data with counts
seat_col = "seating_capacity_num" if "seating_capacity_num" in df_raw.columns else "original_seating_capacity" if "original_seating_capacity" in df_raw.columns else None

seat_options = ["Any"]
seat_label_to_int = {}  # map label -> int seat count for parsing later

if seat_col:
    # convert to numeric, drop NaN
    seat_series = pd.to_numeric(df_raw[seat_col], errors="coerce").dropna().astype(int)
    if not seat_series.empty:
        counts = seat_series.value_counts().to_dict()  # {seats: count}
        # sort seat counts ascending
        for seats in sorted(counts.keys()):
            label = f"{seats} seats ({counts[seats]})"
            seat_options.append(label)
            seat_label_to_int[label] = int(seats)
    else:
        # fallback defaults (common seat values)
        fallback = [2,4,5,7]
        for s in fallback:
            label = f"{s} seats (0)"
            seat_options.append(label)
            seat_label_to_int[label] = s
else:
    # no seating column: show sensible defaults
    fallback = [2,4,5,7]
    for s in fallback:
        label = f"{s} seats (0)"
        seat_options.append(label)
        seat_label_to_int[label] = s

seating_choice = st.sidebar.selectbox("Seating (exact match)", options=seat_options, index=0, key="seating_choice")

st.sidebar.markdown("---")
st.sidebar.header("Fuel & Body handling")
fuel_mode = st.sidebar.radio("Fuel mode", options=["Hard filter", "Soft preference"], index=0, key="fuel_mode")
body_mode = st.sidebar.radio("Body mode", options=["Hard filter", "Soft preference"], index=0, key="body_mode")

soft_fuel = (fuel_mode == "Soft preference")
soft_body = (body_mode == "Soft preference")

fuel_pref_weight = st.sidebar.slider("Fuel soft-preference importance", 1, 10, 5) if soft_fuel else 0
body_pref_weight = st.sidebar.slider("Body soft-preference importance", 1, 10, 5) if soft_body else 0

st.sidebar.markdown("---")
st.sidebar.header("Priorities (1 low — 10 high)")
col1, col2 = st.sidebar.columns(2)
with col1:
    performance_w = st.slider("Performance", 1, 10, 5)
    economy_w = st.slider("Economy", 1, 10, 6)
    safety_w = st.slider("Safety", 1, 10, 8)
with col2:
    comfort_w = st.slider("Comfort", 1, 10, 5)
    ownership_w = st.slider("Ownership", 1, 10, 5)
    price_w = st.slider("Price sensitivity", 1, 10, 6)

st.sidebar.markdown("---")
top_n = st.sidebar.slider("Show top N results", 5, 200, 20)
if st.sidebar.button("Reset primary filters"):
    reset_filters()
    st.experimental_rerun()

# ---------- BUILD PRIMARY FILTERS ----------
min_budget = es.parse_budget_input(min_budget_input)
max_budget = es.parse_budget_input(max_budget_input)

df_filtered = df_raw.copy()
price_col_guess = "price_inr" if "price_inr" in df_filtered.columns else None
if min_budget is not None and price_col_guess:
    df_filtered = df_filtered[df_filtered[price_col_guess].fillna(0) >= min_budget]
if max_budget is not None and price_col_guess:
    df_filtered = df_filtered[df_filtered[price_col_guess].fillna(np.inf) <= max_budget]

# seating exact-match filter
if seat_col and seating_choice and seating_choice != "Any":
    # parse number from label using our map (preferred) else regex fallback
    sel_seats = None
    if seating_choice in seat_label_to_int:
        sel_seats = seat_label_to_int[seating_choice]
    else:
        m = re.match(r"(\d+)", seating_choice)
        if m:
            sel_seats = int(m.group(1))
    if sel_seats is not None:
        df_filtered = df_filtered[pd.to_numeric(df_filtered[seat_col], errors="coerce").fillna(-1).astype(int) == sel_seats]

# fuel hard filter
if fuel_choice and fuel_mode == "Hard filter":
    df_filtered = df_filtered[df_filtered["fuel_category"] == fuel_choice]

# body hard filter
if body_choice and body_mode == "Hard filter":
    df_filtered = df_filtered[df_filtered.get("body_type_clean", df_filtered.get("original_body_type", pd.Series())).str.lower().str.contains(body_choice.lower(), na=False)]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Cars remaining:** {len(df_filtered)}  (from {len(df_raw)})")

if df_filtered.empty:
    st.warning("No cars matched the primary filters. Widen budget, clear hard filters, or choose a different seating option.")
    st.stop()

# ---------- BUILD WEIGHTS & RUN RECOMMENDER ----------
weights = {
    "performance": float(performance_w),
    "economy": float(economy_w),
    "safety": float(safety_w),
    "comfort": float(comfort_w),
    "ownership": float(ownership_w),
    "price": float(price_w)
}
if soft_fuel:
    weights["fuel_pref"] = float(fuel_pref_weight)
if soft_body:
    weights["body_pref"] = float(body_pref_weight)

filters = {}
if min_budget: filters["min_budget"] = min_budget
if max_budget: filters["max_budget"] = max_budget
if seat_col and seating_choice and seating_choice != "Any":
    try:
        # store exact integer for reference
        if seating_choice in seat_label_to_int:
            filters["seating_choice"] = int(seat_label_to_int[seating_choice])
        else:
            m = re.match(r"(\d+)", seating_choice)
            if m:
                filters["seating_choice"] = int(m.group(1))
    except:
        pass

with st.spinner("Computing recommendations..."):
    df_with_axes = es.compute_axes(df_filtered)
    results = es.recommend(df_with_axes, weights, top_n=top_n, filters=filters,
                        soft_fuel=soft_fuel, soft_body=soft_body,
                        selected_fuel=(fuel_choice if soft_fuel else None),
                        selected_body=(body_choice if soft_body else None),
                        soft_fuel_weight=(fuel_pref_weight if soft_fuel else 0.0),
                        soft_body_weight=(body_pref_weight if soft_body else 0.0))

# ---------- METRICS & DISPLAY ----------
col_a, col_b, col_c, col_d = st.columns([2,1,1,1])
with col_a:
    st.markdown("#### Active filters")
    st.write(f"Budget: {es.format_price(min_budget) if min_budget else 'None'} — {es.format_price(max_budget) if max_budget else 'None'}")
    if seat_col and seating_choice and seating_choice != "Any":
        # show the integer seating if available
        sc = seat_label_to_int.get(seating_choice)
        st.write(f"Seating: {sc if sc is not None else seating_choice}")
    else:
        st.write("Seating: Any")
    st.write(f"Fuel: {fuel_choice or 'Any'} ({fuel_mode}) — Body: {body_choice or 'Any'} ({body_mode})")
with col_b:
    st.metric(label="Candidates", value=len(df_filtered), delta=f"{len(results)} shown")
with col_c:
    avg_price = results["price_inr"].median() if ("price_inr" in results.columns and not results.empty) else None
    st.metric(label="Median price (shown)", value=(es.format_price(int(avg_price)) if avg_price and not np.isnan(avg_price) else "N/A"))
with col_d:
    st.metric(label="Shortlist", value=len(st.session_state["shortlist"]))

st.markdown("---")

if results.empty:
    st.warning("No cars matched after scoring.")
else:
    st.markdown(f"### Top {len(results)} Cars")
    view_mode = st.radio("View", options=["Grid (cards)", "Table"], index=0, horizontal=True)

    def row_id(r):
        if "id" in r and not pd.isna(r["id"]):
            return str(r["id"])
        return f"{r.name}-{r.get('brand','')}-{r.get('model_name','')}-{r.get('variant','')}"

#cards view
    if view_mode == "Grid (cards)":
        cards_per_row = 3
        n = len(results)
        rows = (n + cards_per_row - 1) // cards_per_row

        def get_image_url_or_none(row):
            for key in VALID_IMAGE_KEYS:
                if key in row.index and row.get(key):
                    val = row.get(key)
                    if isinstance(val, str) and val.strip():
                        if len(val.strip()) > 5 and ("/" in val or val.startswith("http")):
                            return val.strip()
            return None

        def s(v):
            return "" if (pd.isna(v) or v is None) else str(v)

        for r in range(rows):
            cols = st.columns(cards_per_row, gap="large")
            for c in range(cards_per_row):
                idx = r*cards_per_row + c
                if idx >= n:
                    cols[c].empty()
                    continue

                row = results.iloc[idx]
                car_uid = row_id(row)
                img_url = get_image_url_or_none(row)

                with cols[c]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)

                    # only render image if present
                    if img_url:
                        try:
                            st.image(img_url, use_container_width=True)
                        except:
                            pass

                    brand = s(row.get("brand", row.get("original_brand", "")))
                    model = s(row.get("model_name", row.get("original_model_name", "")))
                    variant = s(row.get("variant", row.get("original_variant", "")))
                    title = (brand + " " + model + (" - " + variant if variant else "")).strip()
                    fuel = s(row.get("fuel_category", row.get("fuel_raw", "")))

                    st.markdown(f'''
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px">
                          <div style="font-weight:700;color:#e6eef8">{title}</div>
                          <div class="pill">{fuel}</div>
                        </div>''', unsafe_allow_html=True)

                    price_text = es.format_price(row.get("price_inr")) if "price_inr" in row.index else "N/A"
                    mileage = s(row.get("mileage_value", row.get("mileage")))
                    power = s(row.get("power_bhp", row.get("power")))
                    seats_display = s(row.get(seat_col)) if seat_col else "N/A"

                    specs_html = f'<div class="spec" style="margin-top:6px">Price: <strong style="color:#fff;">{price_text}</strong> • Mil: {mileage or "N/A"} kmpl • Pwr: {power or "N/A"} bhp • Seats: {seats_display or "N/A"}</div>'
                    st.markdown(specs_html, unsafe_allow_html=True)

                    score_val = float(row.get("final_score", 0.0) or 0.0)
                    max_score = results["final_score"].max() if ("final_score" in results.columns and not results.empty) else 1.0
                    score_pct = 0.0
                    if max_score and max_score > 0:
                        score_pct = min(max(score_val / max_score, 0.0), 1.0)
                    st.markdown(f'''
                        <div style="display:flex;align-items:center;margin-top:10px">
                          <div style="flex:1;height:10px;background:#0b1220;border-radius:999px;margin-right:10px;overflow:hidden">
                            <div style="width:{int(score_pct*100)}%;height:100%;background:linear-gradient(90deg,#ff8a00,#ff5e62);border-radius:999px"></div>
                          </div>
                          <div style="min-width:54px;text-align:right;font-weight:700;color:#e6eef8">{score_val:.3f}</div>
                        </div>
                        ''', unsafe_allow_html=True)

                    st.write("")

                    b1, b2, b3 = st.columns([1,1,1])
                    with b1:
                        if st.button("Details", key=f"det-{car_uid}"):
                            st.session_state["_open_details"] = car_uid
                    with b2:
                        if car_uid in st.session_state["shortlist"]:
                            if st.button("Remove", key=f"rm-{car_uid}"):
                                remove_from_shortlist(car_uid)
                        else:
                            if st.button("Shortlist", key=f"sl-{car_uid}"):
                                add_to_shortlist(car_uid)
                    with b3:
                        if car_uid in st.session_state["compare"]:
                            if st.button("Uncompare", key=f"ucmp-{car_uid}"):
                                toggle_compare(car_uid)
                        else:
                            if st.button("Compare", key=f"cmp-{car_uid}"):
                                toggle_compare(car_uid)

                    st.markdown("</div>", unsafe_allow_html=True)

#table view
    else:
        display_cols = [
            "brand","model_name","variant",
            "body_type_clean","fuel_category",
            "price_inr","seating_capacity_num","mileage_value","engine_cc_num","power_bhp","torque_nm",
            "airbags_num","adas_level_num","sunroof_yes","cruise_control_yes","ground_clearance_mm","final_score"
        ]
        display_cols = [c for c in display_cols if c in results.columns]
        df_display = results[display_cols].copy()
        if "price_inr" in df_display.columns:
            df_display["price"] = df_display["price_inr"].apply(es.format_price)
            df_display = df_display.drop(columns=["price_inr"])
        df_display = df_display.rename(columns=lambda x: x.replace("_"," ").title())
        st.dataframe(df_display.style.format({"Final Score":"{:.3f}"}), use_container_width=True)

    # details expander logic
    if st.session_state.get("_open_details"):
        target_uid = st.session_state.get("_open_details")
        found = None
        for i, r in results.reset_index().iterrows():
            if row_id(r) == target_uid:
                found = r
                break
        if found is not None:
            st.markdown("---")
            st.markdown(f"### Details — {found.get('brand','')} {found.get('model_name','')} {found.get('variant','') or ''}")
            cols1, cols2 = st.columns([2,3])
            with cols1:
                img_url = None
                for k in VALID_IMAGE_KEYS:
                    if k in found.index and found.get(k):
                        v = found.get(k)
                        if isinstance(v, str) and len(v.strip())>5 and ("/" in v or v.startswith("http")):
                            img_url = v.strip()
                            break
                if img_url:
                    try:
                        st.image(img_url, use_container_width=True)
                    except:
                        pass
            with cols2:
                st.write("**Key specs**")
                keys = ["price_inr","seating_capacity_num","mileage_value","engine_cc_num","power_bhp","top_speed_kmph","airbags_num","adas_level_num","fuel_category","body_type_clean"]
                for k in keys:
                    if k in found.index:
                        if k=="price_inr":
                            st.write("Price:", es.format_price(found[k]))
                        else:
                            st.write(f"{k}: {found[k]}")
                st.write("")
                st.write("**Full row (raw)**")
                raw = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in found.to_dict().items()}
                st.json(raw)
        del st.session_state["_open_details"]

    # Shortlist & Compare panels
    st.markdown("---")
    s1, s2 = st.columns([1,1])
    with s1:
        st.subheader("Shortlist")
        if st.session_state["shortlist"]:
            rows_short = []
            for uid in st.session_state["shortlist"]:
                match = None
                for i, r in results.reset_index().iterrows():
                    if row_id(r) == uid:
                        match = r
                        break
                if match is None:
                    for i, r in df_filtered.reset_index().iterrows():
                        if row_id(r) == uid:
                            match = r
                            break
                if match is not None:
                    rows_short.append(match)
            if rows_short:
                for r in rows_short:
                    st.markdown(f"**{r.get('brand','')} {r.get('model_name','')} {r.get('variant','') or ''}** — {es.format_price(r.get('price_inr'))}")
                    if st.button("Remove", key=f"short-remove-{row_id(r)}"):
                        remove_from_shortlist(row_id(r))
                        st.experimental_rerun()
                buf = io.StringIO()
                pd.DataFrame([r.to_dict() for r in rows_short]).to_csv(buf, index=False)
                buf.seek(0)
                st.download_button("Download shortlist (CSV)", data=buf.getvalue(), file_name="shortlist_cars.csv", mime="text/csv")
            else:
                st.info("Shortlisted cars are not in current filtered results.")
        else:
            st.info("No cars shortlisted yet. Click 'Shortlist' on any card.")
    with s2:
        st.subheader("Compare (max 3)")
        if st.session_state["compare"]:
            compare_rows = []
            for uid in st.session_state["compare"]:
                found = None
                for i, r in results.reset_index().iterrows():
                    if row_id(r) == uid:
                        found = r
                        break
                if found is None:
                    for i, r in df_filtered.reset_index().iterrows():
                        if row_id(r) == uid:
                            found = r
                            break
                if found is not None:
                    compare_rows.append(found)
            if compare_rows:
                comp_cols = ["brand","model_name","variant","fuel_category","body_type_clean","price_inr","seating_capacity_num","mileage_value","engine_cc_num","power_bhp","airbags_num","adas_level_num","final_score"]
                comp_cols = [c for c in comp_cols if c in compare_rows[0].index]
                comp_df = pd.DataFrame([{c: r.get(c) for c in comp_cols} for r in compare_rows])
                if "price_inr" in comp_df.columns:
                    comp_df["price_display"] = comp_df["price_inr"].apply(es.format_price)
                    comp_df = comp_df.drop(columns=["price_inr"])
                st.dataframe(comp_df.T, use_container_width=True)
                if st.button("Clear compare"):
                    st.session_state["compare"] = []
            else:
                st.info("Selected compare cars are not in current view.")
        else:
            st.info("Add cars to compare from cards (max 3).")

st.markdown("---")
st.caption(textwrap.dedent("""
    Notes:
    • Seating dropdown now shows actual seat counts present in the dataset along with counts (e.g., "5 seats (42)").
    • Selecting a seat option filters for cars with exactly that number of seats. Choose "Any" to disable.  
"""))
