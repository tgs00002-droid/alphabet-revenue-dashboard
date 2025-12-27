import re
import time
import io
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# =========================
# Config
# =========================
SEGMENT_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
FINANCIALS_URL = "https://stockanalysis.com/stocks/goog/financials/"

LOGO_URL = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# =========================
# Helpers: parsing / detection
# =========================
def parse_money_to_float(x) -> float:
    """Parse values like 56.57B, 344.0M, -207.0M, 12,345, 123.4 into float dollars."""
    if x is None:
        return np.nan
    s = str(x).strip()
    if s in ("", "—", "-", "N/A", "NA", "None", "nan", "NaN"):
        return np.nan

    # remove $ and commas
    s = s.replace("$", "").replace(",", "")
    neg = False

    # handle (123) negative style
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # allow leading negative sign
    if s.startswith("-"):
        neg = True
        s = s[1:].strip()

    m = re.match(r"^(\d+(\.\d+)?)([KMBT])?$", s, flags=re.IGNORECASE)
    if not m:
        try:
            val = float(s)
            return -val if neg else val
        except Exception:
            return np.nan

    num = float(m.group(1))
    suf = (m.group(3) or "").upper()
    mult = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suf]
    val = num * mult
    return -val if neg else val


def is_quarter_end(dt: pd.Timestamp) -> bool:
    if pd.isna(dt):
        return False
    # StockAnalysis uses quarter-end dates like Mar 31, Jun 30, Sep 30, Dec 31
    return (dt.month, dt.day) in [(3, 31), (6, 30), (9, 30), (12, 31)]


def wide_table_to_long(df_wide: pd.DataFrame, label_name: str) -> pd.DataFrame:
    """
    Turn a 'wide' table where first col is labels and the rest are dates into long:
    date | label_name | value
    """
    df = df_wide.copy()
    df.columns = [str(c).strip() for c in df.columns]

    label_col = df.columns[0]
    # Identify date-like columns
    date_cols = []
    for c in df.columns[1:]:
        try:
            pd.to_datetime(str(c))
            date_cols.append(c)
        except Exception:
            pass

    if not date_cols:
        return pd.DataFrame()

    long = df.melt(id_vars=[label_col], value_vars=date_cols, var_name="date", value_name="value_raw")
    long = long.rename(columns={label_col: label_name})
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long[label_name] = long[label_name].astype(str).str.strip()
    long["value"] = long["value_raw"].apply(parse_money_to_float)
    long = long.dropna(subset=["date"]).dropna(subset=["value"])
    long = long[[ "date", label_name, "value" ]].sort_values([label_name, "date"]).reset_index(drop=True)
    return long


def detect_ttm_vs_quarterly(long_df: pd.DataFrame, label_col: str) -> str:
    """
    Heuristic:
    - If values are monotonic-ish and changes are small, could be quarterly.
    - If values represent trailing totals, then (TTM - prior TTM) yields plausible quarterly.
    We do a simple check: compute diffs by label and see how many diffs are negative/huge.
    """
    df = long_df.sort_values([label_col, "date"]).copy()
    diffs = df.groupby(label_col)["value"].diff()

    # If many diffs are negative across segments, it might be quarterly already (since quarterly can go up/down),
    # but TTM should rarely drop massively unless business changes.
    neg_rate = (diffs < 0).mean()
    # If diffs are mostly positive and smooth, could be TTM.
    # If neg_rate is very low, suspect TTM.
    if neg_rate < 0.15:
        return "TTM"
    return "Quarterly"


def ttm_to_quarterly(long_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = long_df.sort_values([label_col, "date"]).copy()
    df["q"] = df.groupby(label_col)["value"].diff()
    df = df.dropna(subset=["q"]).copy()
    df["value"] = df["q"]
    df = df.drop(columns=["q"])
    return df.sort_values([label_col, "date"]).reset_index(drop=True)


def pick_best_table(tables: list[pd.DataFrame], mode: str) -> pd.DataFrame | None:
    """
    mode='segment' or 'financials'
    Choose table with date columns + expected keywords.
    """
    best = None
    best_score = -1

    segment_keywords = ["search", "youtube", "cloud", "network", "other bets", "subscriptions", "platforms", "devices", "hedging"]
    fin_keywords = ["revenue", "gross", "operating", "net income", "eps", "shares", "tax", "cost of revenue"]

    for t in tables:
        # must have at least 2 cols
        if t.shape[1] < 2:
            continue

        # must have date columns
        long_try = wide_table_to_long(t, "label")
        if long_try.empty:
            continue

        labels_text = " ".join(long_try["label"].astype(str).str.lower().unique().tolist())

        hits = 0
        if mode == "segment":
            hits = sum(k in labels_text for k in segment_keywords)
        else:
            hits = sum(k in labels_text for k in fin_keywords)

        # score: many rows + keyword hits
        score = long_try.shape[0] + hits * 300
        if score > best_score:
            best = t
            best_score = score

    return best


# =========================
# Fetch + parse (cached)
# =========================
def http_get(url: str) -> str:
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


@st.cache_data(ttl=60 * 60 * 6)  # 6 hours
def load_segment_data_live() -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      seg_q_long: date | segment | value  (QUARTERLY dollars)
      meta: info about what happened (ttm detected, etc.)
    """
    meta = {"source": SEGMENT_URL, "detected": None, "converted_from_ttm": False}

    html = http_get(SEGMENT_URL)
    tables = pd.read_html(io.StringIO(html))
    best = pick_best_table(tables, mode="segment")
    if best is None:
        raise ValueError("Could not find segment table on the page (page layout or blocking).")

    seg_long = wide_table_to_long(best, "segment")

    # sanity: quarter-end dates?
    if seg_long["date"].map(is_quarter_end).mean() < 0.7:
        # still might be okay, but warn
        meta["date_warning"] = "Many dates are not quarter-end. StockAnalysis may have changed formatting."

    detected = detect_ttm_vs_quarterly(seg_long, "segment")
    meta["detected"] = detected

    if detected == "TTM":
        seg_q = ttm_to_quarterly(seg_long, "segment")
        meta["converted_from_ttm"] = True
    else:
        seg_q = seg_long

    # final sanity checks
    seg_q = seg_q.dropna(subset=["value"]).copy()
    seg_q = seg_q.sort_values(["segment", "date"])

    # flag extremely weird quarter values (optional)
    # (hedging gains can be negative; others usually non-negative but we won't force it)
    meta["rows"] = int(seg_q.shape[0])
    meta["segments"] = int(seg_q["segment"].nunique())
    meta["min_date"] = str(seg_q["date"].min().date())
    meta["max_date"] = str(seg_q["date"].max().date())

    return seg_q.reset_index(drop=True), meta


@st.cache_data(ttl=60 * 60 * 6)  # 6 hours
def load_income_statement_live() -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      inc_long: date | metric | value  (as reported on the page; typically quarterly)
    """
    meta = {"source": FINANCIALS_URL}

    html = http_get(FINANCIALS_URL)
    tables = pd.read_html(io.StringIO(html))
    best = pick_best_table(tables, mode="financials")
    if best is None:
        raise ValueError("Could not find income statement table on the page (page layout or blocking).")

    inc_long = wide_table_to_long(best, "metric")

    meta["rows"] = int(inc_long.shape[0])
    meta["metrics"] = int(inc_long["metric"].nunique())
    meta["min_date"] = str(inc_long["date"].min().date())
    meta["max_date"] = str(inc_long["date"].max().date())

    return inc_long.reset_index(drop=True), meta


# =========================
# Forecast helper
# =========================
def forecast_series_quarterly(hist_df: pd.DataFrame, years_out: int, extra_cagr: float) -> pd.DataFrame:
    """
    Simple scenario forecast:
    - infer baseline quarterly growth from last 8 quarters
    - add extra uplift (annual) -> convert to quarterly
    """
    df = hist_df.sort_values("date").copy()
    if df.shape[0] < 6:
        last = df["value"].iloc[-1] if df.shape[0] else 0
        future_dates = pd.date_range(df["date"].max() + pd.offsets.QuarterEnd(), periods=years_out * 4, freq="Q")
        return pd.DataFrame({"date": future_dates, "forecast": last})

    tail = df.tail(8)
    start_val = tail["value"].iloc[0]
    end_val = tail["value"].iloc[-1]
    n = max(1, tail.shape[0] - 1)

    if start_val <= 0 or end_val <= 0:
        q_growth = 0.0
    else:
        q_growth = (end_val / start_val) ** (1 / n) - 1

    extra_q = (1 + extra_cagr) ** (1 / 4) - 1
    q_growth_scn = q_growth + extra_q

    last_date = df["date"].max()
    v = df["value"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(), periods=years_out * 4, freq="Q")

    vals = []
    for _ in range(len(future_dates)):
        v = v * (1 + q_growth_scn)
        vals.append(v)

    return pd.DataFrame({"date": future_dates, "forecast": vals})


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Alphabet Revenue Forecast", layout="wide")

c1, c2 = st.columns([1, 10])
with c1:
    st.image(LOGO_URL, width=120)
with c2:
    st.markdown("## Alphabet (Google) Revenue Forecast Dashboard")
    st.caption("Live-scraped from StockAnalysis (segment revenue + income statement) with auto TTM→Quarterly conversion when needed.")

st.divider()

st.sidebar.header("Controls")

refresh = st.sidebar.button("🔄 Force Refresh (ignore cache)")

if refresh:
    load_segment_data_live.clear()
    load_income_statement_live.clear()

years = st.sidebar.slider("Forecast years", 1, 10, 5, 1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario)", 0.00, 0.30, 0.00, 0.01)

st.sidebar.divider()
st.sidebar.caption("If StockAnalysis blocks Streamlit Cloud, run locally or use scheduled caching (GitHub Action).")

# Load data
try:
    seg_q, seg_meta = load_segment_data_live()
except Exception as e:
    st.error("Segment scrape failed.")
    st.write("Reason:", str(e))
    st.stop()

try:
    inc, inc_meta = load_income_statement_live()
except Exception as e:
    st.error("Income statement scrape failed.")
    st.write("Reason:", str(e))
    st.stop()

# Sidebar: segment select
segments = sorted(seg_q["segment"].unique().tolist())
segment_choice = st.sidebar.selectbox("Segment", segments, index=0)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Segment Forecast", "Total (Stacked)", "Income Statement", "Data Quality"])

with tab1:
    st.subheader(f"{segment_choice}: Quarterly history + scenario forecast")

    hist = seg_q[seg_q["segment"] == segment_choice].sort_values("date")
    fc = forecast_series_quarterly(hist[["date", "value"]], years, uplift)

    hist_plot = hist.copy()
    hist_plot["y"] = hist_plot["value"] / 1e9
    hist_plot["type"] = "Historical"

    fc_plot = fc.copy()
    fc_plot["y"] = fc_plot["forecast"] / 1e9
    fc_plot["type"] = "Forecast"
    fc_plot = fc_plot[["date", "y", "type"]]

    plot_df = pd.concat([hist_plot[["date", "y", "type"]], fc_plot], ignore_index=True)

    fig = px.line(
        plot_df, x="date", y="y", color="type",
        labels={"y": "Revenue (USD Billions)", "date": "Quarter"},
        title=f"{segment_choice} | {years}y forecast | uplift {uplift:.0%}"
    )
    st.plotly_chart(fig, use_container_width=True)

    last_hist = float(hist["value"].iloc[-1]) if not hist.empty else 0.0
    end_fc = float(fc["forecast"].iloc[-1]) if not fc.empty else last_hist
    delta = end_fc - last_hist

    k1, k2, k3 = st.columns(3)
    k1.metric("Last Quarter", f"${last_hist/1e9:,.2f}B")
    k2.metric("End Forecast", f"${end_fc/1e9:,.2f}B")
    k3.metric("Δ vs Last Quarter", f"${delta/1e9:,.2f}B")

with tab2:
    st.subheader("Quarterly segment mix (stacked)")

    wide = seg_q.pivot_table(index="date", columns="segment", values="value", aggfunc="sum").sort_index()
    wide_b = wide / 1e9
    long = wide_b.reset_index().melt(id_vars=["date"], var_name="segment", value_name="revenue_b")

    fig2 = px.bar(
        long, x="date", y="revenue_b", color="segment",
        labels={"revenue_b": "Revenue (USD Billions)", "date": "Quarter"},
        title="Segment Revenue (Quarterly) — Stacked"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Income statement (as reported)")

    metrics = sorted(inc["metric"].unique().tolist())
    metric_choice = st.selectbox("Metric", metrics, index=0)

    mdf = inc[inc["metric"] == metric_choice].sort_values("date").copy()
    mdf["value_b"] = mdf["value"] / 1e9

    fig3 = px.line(
        mdf, x="date", y="value_b",
        labels={"value_b": "USD (Billions)", "date": "Quarter"},
        title=f"{metric_choice} (USD Billions)"
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(mdf[["date", "metric", "value"]], use_container_width=True)

with tab4:
    st.subheader("Data Quality / Validation")

    st.write("### Segment page scrape")
    st.json(seg_meta)

    # Validate quarter-end coverage
    q_end_rate = seg_q["date"].map(is_quarter_end).mean()
    st.write(f"- Quarter-end date rate: **{q_end_rate:.0%}** (should be high)")
    st.write(f"- Detected format: **{seg_meta.get('detected')}**")
    if seg_meta.get("converted_from_ttm"):
        st.success("TTM detected → converted to quarterly using diff(TTM).")
    else:
        st.info("Quarterly detected → using as-is.")

    # Show sample vs tooltip style
    st.write("### Latest quarter (segment breakdown)")
    latest_q = seg_q["date"].max()
    snap = seg_q[seg_q["date"] == latest_q].sort_values("value", ascending=False).copy()
    snap["value_b"] = snap["value"] / 1e9
    st.write(f"Quarter: **{latest_q.date()}**")
    st.dataframe(snap[["segment", "value", "value_b"]], use_container_width=True)

    st.write("### Income statement scrape")
    st.json(inc_meta)

# Downloads
st.divider()
c_dl1, c_dl2 = st.columns(2)
with c_dl1:
    st.download_button(
        "Download segment quarterly CSV",
        seg_q.to_csv(index=False).encode("utf-8"),
        "alphabet_segment_quarterly.csv",
        "text/csv",
    )
with c_dl2:
    st.download_button(
        "Download income statement CSV",
        inc.to_csv(index=False).encode("utf-8"),
        "alphabet_income_statement.csv",
        "text/csv",
    )
