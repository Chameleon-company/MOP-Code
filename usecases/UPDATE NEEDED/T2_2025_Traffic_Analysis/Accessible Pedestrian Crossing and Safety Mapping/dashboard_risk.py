import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from pandas.api.types import is_integer_dtype, is_numeric_dtype

st.set_page_config(page_title="Pedestrian Crash Risk Dashboard", layout="wide")

# =========================
# Load data
# =========================
@st.cache_data(show_spinner=False)
def load_df(path="features_df.csv"):
    try:
        return pd.read_csv(path, encoding="latin1")
    except FileNotFoundError:
        st.error("Couldn't find 'features_df.csv' next to dashboard_risk.py.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to read 'features_df.csv': {e}")
        st.stop()

df = load_df()
st.title("Pedestrian Crash Risk Dashboard")
st.caption(
    f"Loaded rows: {len(df):,}. Columns: "
    + ", ".join(map(str, df.columns))
)

# =========================
# Helpers
# =========================
def coerce_hour(series):
    s = pd.to_numeric(series, errors="coerce").round()
    s = s.clip(lower=0, upper=23)
    return s.astype("Int64")

def to_weekday(s):
    """Map DAY_OF_WEEK in almost any format to full weekday names."""
    ser = pd.Series(s)

    # 1) String labels like "Mon", "monday", "SUN"
    if ser.dtype == object:
        txt = ser.astype(str).str.strip().str.lower()
        key = txt.str[:3]
        map3 = {
            "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
            "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday"
        }
        wk = key.map(map3)
        if wk.notna().sum() >= max(1, int(0.6 * len(ser.dropna()))):
            return wk
        # else fall through to numeric

    # 2) Numeric-looking values
    num = pd.to_numeric(ser, errors="coerce")
    base_mon = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    base_sun = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

    # 0..6 (Mon=0)
    if num.dropna().between(0,6).all():
        return num.map({i: base_mon[i] for i in range(7)})

    # 1..7 (Mon=1) or (Sun=1)
    if num.dropna().between(1,7).all():
        mapped_mon1 = num.map({i+1: base_mon[i] for i in range(7)})
        mapped_sun1 = num.map({i+1: base_sun[i] for i in range(7)})
        return mapped_mon1.where(mapped_mon1.notna(), mapped_sun1)

    # 3) Datetime / epoch timestamps
    dt = pd.to_datetime(ser, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if dt.notna().any():
        return dt.dt.day_name()

    # 4) Give up
    return pd.Series(pd.NA, index=ser.index)

def label_from_risk(vals):
    ser = pd.Series(vals)
    if is_numeric_dtype(ser):
        uniq = np.sort(ser.dropna().unique())
        if len(uniq) == 2 and set(uniq).issubset({0,1}):
            return ser.map({0:"Low", 1:"High"})
        if len(uniq) == 3 and set(uniq).issubset({0,1,2}):
            return ser.map({0:"Low", 1:"Medium", 2:"High"})
    return ser.astype(str)

def percent_high(group):
    if len(group) == 0: return 0.0
    return (group["risk_label"].eq("High").mean() * 100.0)

# =========================
# Expect these columns (from your list)
# =========================
expected = [
    "LATITUDE","LONGITUDE","ACCIDENT_HOUR","DAY_OF_WEEK","SPEED_ZONE",
    "LIGHT_CONDITION","ROAD_GEOMETRY","is_intersection","NO_OF_VEHICLES",
    "INJ_OR_FATAL","FATALITY","SERIOUSINJURY","vulnerable_total",
    "DEG_URBAN_NAME","DTP_REGION","risk_category"
]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing expected columns: {missing}")
    st.stop()

# =========================
# Prepare canonical fields
# =========================
df["risk_label"]    = label_from_risk(df["risk_category"])
df["ACCIDENT_HOUR"] = coerce_hour(df["ACCIDENT_HOUR"])
df = df.dropna(subset=["ACCIDENT_HOUR"])
df["ACCIDENT_HOUR"] = df["ACCIDENT_HOUR"].astype(int)
df["WEEKDAY"]       = to_weekday(df["DAY_OF_WEEK"])

# =========================
# Sidebar filters
# =========================
risk_options = sorted(df["risk_label"].dropna().unique().tolist())
risk_filter = st.sidebar.multiselect("Select Risk Category", options=risk_options, default=risk_options)
df = df[df["risk_label"].isin(risk_filter)]

# Hour slider
if len(df):
    hmin, hmax = int(df["ACCIDENT_HOUR"].min()), int(df["ACCIDENT_HOUR"].max())
    hr_range = st.sidebar.slider("Hour filter", 0, 23, (hmin, hmax))
    df = df[(df["ACCIDENT_HOUR"] >= hr_range[0]) & (df["ACCIDENT_HOUR"] <= hr_range[1])]

#  extra filters
geo_opts = sorted(df["ROAD_GEOMETRY"].dropna().unique().tolist())
sel_geos = st.sidebar.multiselect("Road geometry", geo_opts, default=geo_opts[:10] if len(geo_opts)>10 else geo_opts)
if sel_geos: df = df[df["ROAD_GEOMETRY"].isin(sel_geos)]

spd_opts = sorted(df["SPEED_ZONE"].dropna().unique().tolist())
sel_spds = st.sidebar.multiselect("Speed zone", spd_opts, default=spd_opts[:10] if len(spd_opts)>10 else spd_opts)
if sel_spds: df = df[df["SPEED_ZONE"].isin(sel_spds)]

# =========================
# Tabs
# =========================
tab_over, tab_time, tab_loc, tab_design, tab_vuln = st.tabs(
    ["Overview", "Patterns Over Time", "Hotspots & Locations", "Road Design & Speed", "Vulnerable Users"]
)

# =========================
# Tab: Overview
# =========================
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    total_crashes = len(df)
    pct_high = df["risk_label"].eq("High").mean()*100 if total_crashes else 0
    peak_hour = int(df["ACCIDENT_HOUR"].mode().iloc[0]) if len(df) else "—"
    if len(df):
        hotspot = (
            df.assign(lat_r=df["LATITUDE"].round(3), lon_r=df["LONGITUDE"].round(3))
              .groupby(["lat_r","lon_r"]).size().sort_values(ascending=False).head(1)
        )
        top_hotspot = (f"{hotspot.index[0][0]}, {hotspot.index[0][1]} "
                       f"({int(hotspot.iloc[0])})") if len(hotspot) else "—"
    else:
        top_hotspot = "—"

    c1.metric("Total crashes (filtered)", f"{total_crashes:,}")
    c2.metric("% High risk", f"{pct_high:.1f}%")
    c3.metric("Peak crash hour", f"{peak_hour}")
    c4.metric("Top hotspot (rounded)", top_hotspot)

    st.markdown("### Crashes by hour of day, split by risk category")
    work = df.dropna(subset=["ACCIDENT_HOUR", "risk_label"]).copy()
    hour_cat = (
        work.groupby(["ACCIDENT_HOUR","risk_label"]).size().reset_index(name="count")
    )
    all_hours = pd.Index(range(24), name="ACCIDENT_HOUR")
    all_cats  = pd.Index(sorted(work["risk_label"].unique()), name="risk_label")
    hour_cat = (
        hour_cat.set_index(["ACCIDENT_HOUR","risk_label"])
                .reindex(pd.MultiIndex.from_product([all_hours, all_cats]), fill_value=0)
                .reset_index()
    )
    chart_hours = (
        alt.Chart(hour_cat)
           .mark_bar()
           .encode(
               x=alt.X("ACCIDENT_HOUR:O", title="Hour of day (0–23)"),
               y=alt.Y("count:Q", title="Crashes"),
               color=alt.Color("risk_label:N", title="Risk"),
               tooltip=["ACCIDENT_HOUR","risk_label","count"]
           ).properties(height=320)
    )
    st.altair_chart(chart_hours, use_container_width=True)

    st.markdown("### Crash locations map")
    map_df = df[["LATITUDE","LONGITUDE"]].dropna().rename(columns={"LATITUDE":"lat","LONGITUDE":"lon"})
    st.map(map_df)

# =========================
# Tab: Patterns Over Time
# =========================
with tab_time:
    cols = st.columns(2)

    with cols[0]:
        st.markdown("#### Risk mix by hour (100%)")
        mix = (
            df.groupby(["ACCIDENT_HOUR","risk_label"])
              .size().reset_index(name="count")
        )
        tot = mix.groupby("ACCIDENT_HOUR")["count"].transform("sum")
        mix["pct"] = (mix["count"] / tot) * 100
        area = (
            alt.Chart(mix)
               .mark_area()
               .encode(
                   x=alt.X("ACCIDENT_HOUR:O", title="Hour"),
                   y=alt.Y("pct:Q", stack="normalize", title="Share of crashes"),
                   color=alt.Color("risk_label:N", title="Risk"),
                   tooltip=["ACCIDENT_HOUR","risk_label",alt.Tooltip("pct:Q",format=".1f")]
               )
        )
        st.altair_chart(area, use_container_width=True)

    with cols[1]:
        st.markdown("#### % High risk by weekday & hour")
        have_weekday = df["WEEKDAY"].notna().any()
        if have_weekday:
            tmp = (
                df.groupby(["WEEKDAY","ACCIDENT_HOUR"])
                  .apply(lambda g: (g["risk_label"].eq("High").mean()*100) if len(g) else 0)
                  .reset_index(name="pct_high")
            )
            week_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            if set(tmp["WEEKDAY"].dropna().unique()).issubset(set(week_order)):
                tmp["WEEKDAY"] = pd.Categorical(tmp["WEEKDAY"], categories=week_order, ordered=True)
            heat = (
                alt.Chart(tmp)
                   .mark_rect()
                   .encode(
                       x=alt.X("ACCIDENT_HOUR:O", title="Hour"),
                       y=alt.Y("WEEKDAY:N", title="Weekday"),
                       color=alt.Color("pct_high:Q", title="% High", scale=alt.Scale(scheme="reds")),
                       tooltip=["WEEKDAY","ACCIDENT_HOUR",alt.Tooltip("pct_high:Q", format=".1f")]
                   ).properties(height=320)
            )
            st.altair_chart(heat, use_container_width=True)
        else:
            st.info("DAY_OF_WEEK could not be mapped to weekday names. "
                    "Ensure values are numbers (0–6 or 1–7) or strings like 'Mon', 'Tuesday', etc.")

# =========================
# Tab: Hotspots & Locations
# =========================
with tab_loc:
    st.markdown("#### Top location cells (rounded) by crashes and % High")
    g = (
        df.assign(lat_r=df["LATITUDE"].round(3), lon_r=df["LONGITUDE"].round(3))
          .groupby(["lat_r","lon_r"])
          .agg(
              crashes=("risk_label","size"),
              pct_high=("risk_label", lambda s: (s.eq("High").mean()*100) if len(s) else 0)
          )
          .reset_index()
          .sort_values(["crashes","pct_high"], ascending=[False, False])
          .head(15)
    )
    grid = (
        alt.Chart(g)
           .mark_bar()
           .encode(
               x=alt.X("crashes:Q", title="Crashes"),
               y=alt.Y("lat_r:N", sort="-x", title="Latitude (rounded)"),
               color=alt.Color("pct_high:Q", title="% High", scale=alt.Scale(scheme="reds")),
               tooltip=["lat_r","lon_r","crashes", alt.Tooltip("pct_high:Q",format=".1f")]
           ).properties(height=360)
    )
    st.altair_chart(grid, use_container_width=True)

# =========================
# Tab: Road Design & Speed
# =========================
with tab_design:
    cols = st.columns(2)

    with cols[0]:
        st.markdown("#### % High by road geometry")
        geo_stats = (
            df.groupby("ROAD_GEOMETRY")
              .apply(percent_high)
              .reset_index(name="pct_high")
              .sort_values("pct_high", ascending=False)
        )
        geo_chart = (
            alt.Chart(geo_stats)
               .mark_bar()
               .encode(
                   x=alt.X("pct_high:Q", title="% High"),
                   y=alt.Y("ROAD_GEOMETRY:N", sort="-x", title="Road geometry"),
                   tooltip=["ROAD_GEOMETRY", alt.Tooltip("pct_high:Q", format=".1f")]
               ).properties(height=360)
        )
        st.altair_chart(geo_chart, use_container_width=True)

    with cols[1]:
        st.markdown("#### % High by speed zone")
        spd_stats = (
            df.groupby("SPEED_ZONE")
              .apply(percent_high)
              .reset_index(name="pct_high")
              .sort_values("pct_high", ascending=False)
        )
        spd_chart = (
            alt.Chart(spd_stats)
               .mark_bar()
               .encode(
                   x=alt.X("pct_high:Q", title="% High"),
                   y=alt.Y("SPEED_ZONE:N", sort="-x", title="Speed zone"),
                   tooltip=["SPEED_ZONE", alt.Tooltip("pct_high:Q", format=".1f")]
               ).properties(height=360)
        )
        st.altair_chart(spd_chart, use_container_width=True)

# =========================
# Tab: Vulnerable Users
# =========================
with tab_vuln:
    st.markdown("#### Vulnerable users vs risk")
    agg = (
        df.groupby("risk_label")["vulnerable_total"]
          .agg(["count","mean"])
          .reset_index()
          .rename(columns={"mean":"avg_vulnerable"})
    )
    c = (
        alt.Chart(agg)
           .mark_bar()
           .encode(
               x=alt.X("risk_label:N", title="Risk"),
               y=alt.Y("avg_vulnerable:Q", title="Avg. vulnerable count"),
               tooltip=["risk_label","count", alt.Tooltip("avg_vulnerable:Q", format=".2f")],
               color="risk_label:N"
           ).properties(height=320)
    )
    st.altair_chart(c, use_container_width=True)

    share = (
        df.assign(has_vuln=(pd.to_numeric(df["vulnerable_total"], errors="coerce").fillna(0) > 0).astype(int))
          .groupby("risk_label")["has_vuln"].mean().mul(100)
          .rename("pct_with_vulnerable").reset_index()
    )
    c2 = (
        alt.Chart(share)
           .mark_bar()
           .encode(
               x=alt.X("risk_label:N", title="Risk"),
               y=alt.Y("pct_with_vulnerable:Q", title="% crashes with vulnerable person"),
               tooltip=["risk_label", alt.Tooltip("pct_with_vulnerable:Q", format=".1f")],
               color="risk_label:N"
           ).properties(height=320)
    )
    st.altair_chart(c2, use_container_width=True)
