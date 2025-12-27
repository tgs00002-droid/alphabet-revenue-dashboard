import os
import io
import re
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Alphabet Revenue Dashboard", layout="wide")

SEGMENT_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
FINANCIALS_URL = "https://stockanalysis.com/stocks/goog/financials/"

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Cache files (commit these to GitHub)
SEGMENT_CACHE = "data/alphabet_segment_revenue.csv"
INCOME_CACHE = "data/alphabet_income_statement_quarterly.csv"


# =========================
# HELPERS
# =========================
def is_blocked_html(html: str) -> bool:
    h = (html or "").lower()
    blocked_signals = [
        "cloudflare",
        "attention required",
        "captcha",
        "verify you are human",
        "access denied",
        "please enable cookies",
    ]
    return any(s in h for s in blocked_signals)


def parse_money_to_float(x):
    """
    Convert strings like '56.57B', '344.00M', '-207.00M', '1.2T', '123,456'
    into numeric USD units.
    """
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "").replace("$", "")
    if s in ["", "—", "-", "None", "nan", "NaN"]:
        return np.nan

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    m = re.match(r"^(-?\d+(\.\d+)?)([KMBT])?$", s, re.IGNORECASE)
    if not m:
        try:
            val = float(s)
            return -val if neg else val
        except:
            return np.nan

    num = float(m.group(1))
    suf = (m.group(3) or "").upper()
    mult = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suf]
    val = num * mult
    return -val if (neg or (str(m.group(1)).startswith("-"))) else val


def safe_read_html_tables(url: str):
    """Return tables (list of DataFrames) from a URL, or raise."""
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    if is_blocked_html(r.text):
        raise RuntimeError("Blocked / bot-protection HTML returned.")
    tables = pd.read_html(io.StringIO(r.text))
    if not tables:
        raise RuntimeError("No tables found on page.")
    return tables


def pick_best_segment_table(tables):
    """
    StockAnalysis pages can include multiple tables.
    We pick the one that looks like segment revenue over time.
    """
    keywords = [
        "google search", "youtube", "google cloud", "network",
        "subscriptions", "platforms", "devices", "other bets", "hedging"
    ]

    best = None
    best_score = -1

    for t in tables:
        if t.shape[1] < 4:
            continue

        cols = [str(c).strip() for c in t.columns]
        first_col = cols[0].lower()

        # must be segment name-ish
        if "segment" not in first_col and "revenue" not in first_col and "category" not in first_col:
            # still allow if rows contain segment keywords
            pass

        # try to melt to long format if columns contain dates
        long = wide_to_long_time_table(t)
        if long.empty:
            continue

        seg_text = " ".join(long["name"].astype(str).str.lower().unique().tolist())
        hits = sum(k in seg_text for k in keywords)
        score = long.shape[0] + hits * 200

        if score > best_score:
            best = t
            best_score = score

    if best is None:
        raise RuntimeError("Could not detect a segment revenue table.")
    return best


def wide_to_long_time_table(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Converts wide tables like:
      Segment | Sep 30 2025 | Jun 30 2025 | ...
    into long:
      date | name | value
    """
    df = df_wide.copy()
    df.columns = [str(c).strip() for c in df.columns]
    label_col = df.columns[0]

    date_cols = []
    for c in df.columns[1:]:
        try:
            pd.to_datetime(str(c))
            date_cols.append(c)
        except:
            pass

    if not date_cols:
        return pd.DataFrame()

    long = df.melt(id_vars=[label_col], value_vars=date_cols, var_name="date", value_name="value")
    long = long.rename(columns={label_col: "name"})
    long["date"] = pd.to_datetime(long["date"])
    long["value"] = long["value"].apply(parse_money_to_float)
    long = long.dropna(subset=["value"])
    long["name"] = long["name"].astype(str).str.strip()
    long = long.sort_values(["name", "date"]).reset_index(drop=True)
    return long


def ttm_to_quarterly(long_ttm: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TTM series into quarterly using:
      Q_t = TTM_t - TTM_(t-1)
    Per segment or metric name.
    """
    df = long_ttm.sort_values(["name", "date"]).copy()
    df["quarter_value"] = df.groupby("name")["value"].diff(1)
    df = df.dropna(subset=["quarter_value"]).copy()
    df["value"] = df["quarter_value"]
    df = df.drop(columns=["quarter_value"])
    return df.sort_values(["name", "date"]).reset_index(drop=True)


def looks_like_ttm(df_long: pd.DataFrame) -> bool:
    """
    Heuristic:
    - If values are very smooth and always increasing by small amounts, maybe TTM
    - Or if the source label includes 'TTM' (not guaranteed)
    - Or if there are too few points and only yearly cadence (not typical here)
    We'll mainly rely on user checkbox + fallback heuristic.
    """
    # if there are only 1-2 points per year per name, could be TTM/annual
    df = df_long.copy()
    df["year"] = df["date"].dt.year
    per_year = df.groupby(["name", "year"]).size().reset_index(name="n")
    # if most names have <=1 point/year
    return (per_year["n"].median() <= 1)


# =========================
# LOADERS (live -> cache)
# =========================
@st.cache_data(ttl=60 * 60)
def load_segment_data_live_or_cache() -> pd.DataFrame:
    """
    Returns long dataframe: date, name(segment), value
    """
    # Try live
    try:
        tables = safe_read_html_tables(SEGMENT_URL)
        seg_wide = pick_best_segment_table(tables)
        seg_long = wide_to_long_time_table(seg_wide)

        if seg_long.empty:
            raise RuntimeError("Segment table parsed but produced empty long dataset.")

        return seg_long

    except Exception as e:
        # fallback
        if os.path.exists(SEGMENT_CACHE):
            st.warning(f"Live segment scrape blocked. Using cached file: {SEGMENT_CACHE}\nReason: {e}")
            df = pd.read_csv(SEGMENT_CACHE)
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values(["name", "date"]).reset_index(drop=True)

        raise RuntimeError(
            f"Segment scrape failed and no cache found at {SEGMENT_CACHE}.\n"
            f"Create cache CSV and commit it. Root cause: {e}"
        )


@st.cache_data(ttl=60 * 60)
def load_income_statement_live_or_cache() -> pd.DataFrame:
    """
    Returns long dataframe: date, name(metric), value
    from the Income Statement quarterly page.
    """
    try:
        tables = safe_read_html_tables(FINANCIALS_URL)

        # StockAnalysis financials page has multiple tables:
        # We want the one that includes "Revenue", "Gross Profit", etc.
        keyword_metrics = ["revenue", "gross profit", "operating income", "net income", "eps"]

        best = None
        best_score = -1

        for t in tables:
            if t.shape[1] < 4:
                continue
            # first column should look like "Metric"
            first_col = str(t.columns[0]).lower()
            if "metric" not in first_col and "item" not in first_col:
                # still allow
                pass

            long = wide_to_long_time_table(t)
            if long.empty:
                continue

            text = " ".join(long["name"].astype(str).str.lower().unique().tolist())
            hits = sum(k in text for k in keyword_metrics)
            score = long.shape[0] + hits * 300
            if score > best_score:
                best = t
                best_score = score

        if best is None:
            raise RuntimeError("Could not detect income statement table.")

        inc_long = wide_to_long_time_table(best)
        if inc_long.empty:
            raise RuntimeError("Income statement parsed but empty.")

        return inc_long

    except Exception as e:
        if os.path.exists(INCOME_CACHE):
            st.warning(f"Live income statement scrape blocked. Using cached file: {INCOME_CACHE}\nReason: {e}")
            df = pd.read_csv(INCOME_CACHE)
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values(["name", "date"]).reset_index(drop=True)

        raise RuntimeError(
            f"Income statement scrape failed and no cache found at {INCOME_CACHE}.\n"
            f"Create cache CSV and commit it. Root cause: {e}"
        )


# =========================
# UI HEADER (Google logo)
# =========================
LOGO_URL = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"

col_a, col_b = st.columns([1, 12])
with col_a:
    st.image(LOGO_URL, width=120)
with col_b:
    st.markdown(
        """
        # Alphabet (Google) Revenue Forecast Dashboard
        Interactive product-level **quarterly** revenue analysis + scenario forecasting.
        """.strip()
    )

st.caption(
    "Data source: StockAnalysis (segment revenue + income statement). "
    "If scraping is blocked in the cloud, this app automatically uses cached CSVs committed to the repo."
)

st.divider()


# =========================
# LOAD DATA
# =========================
seg_long = load_segment_data_live_or_cache()
inc_long = load_income_statement_live_or_cache()

# Sidebar controls
st.sidebar.header("Controls")

all_segments = sorted(seg_long["name"].unique().tolist())
segment_choice = st.sidebar.selectbox("Segment", all_segments, index=0)

years = st.sidebar.slider("Forecast years", 1, 10, 3, 1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario)", 0.00, 0.20, 0.00, 0.01)

force_ttm_mode = st.sidebar.checkbox(
    "Treat scraped segment data as TTM and convert to quarterly",
    value=False,
    help="Turn this on if the segment dataset you loaded appears to be TTM.",
)

st.sidebar.divider()
st.sidebar.markdown(
    """
**Notes**
- If live scraping fails on Streamlit Cloud, the app will use cached CSVs.
- Best practice: generate caches locally and commit them into `data/`.
"""
)

# If TTM, convert
if force_ttm_mode or looks_like_ttm(seg_long):
    seg_q = ttm_to_quarterly(seg_long)
else:
    seg_q = seg_long.copy()

# =========================
# CORE DATASETS
# =========================
seg_q = seg_q.rename(columns={"name": "segment"})
inc_q = inc_long.rename(columns={"name": "metric"})

seg_q["value_b"] = seg_q["value"] / 1e9
inc_q["value_b"] = inc_q["value"] / 1e9


# =========================
# FORECAST (simple CAGR-based scenario)
# =========================
def forecast_series(hist_df: pd.DataFrame, years_out: int, extra_cagr: float):
    """
    A robust, simple forecast:
    - Compute CAGR from last 8 quarters (2 years) if possible
    - Apply extra CAGR uplift to scenario
    - Forecast quarterly forward
    """
    df = hist_df.sort_values("date").copy()
    df = df.dropna(subset=["value"])
    if df.shape[0] < 6:
        # fallback: flat forecast
        last = df["value"].iloc[-1] if df.shape[0] else 0
        future_dates = pd.date_range(df["date"].max() + pd.offsets.QuarterEnd(), periods=years_out * 4, freq="Q")
        f = pd.DataFrame({"date": future_dates, "forecast": last})
        return f

    # Use last 8 quarters if available
    tail = df.tail(8)
    start_val = tail["value"].iloc[0]
    end_val = tail["value"].iloc[-1]
    n_quarters = max(1, tail.shape[0] - 1)

    # Quarterly growth rate from CAGR approximation
    if start_val <= 0 or end_val <= 0:
        q_growth = 0.0
    else:
        q_growth = (end_val / start_val) ** (1 / n_quarters) - 1

    # Convert extra annual cagr uplift to quarterly
    extra_q = (1 + extra_cagr) ** (1 / 4) - 1
    q_growth_scn = q_growth + extra_q

    last_date = df["date"].max()
    last_val = df["value"].iloc[-1]

    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(), periods=years_out * 4, freq="Q")
    vals = []
    v = last_val
    for _ in range(len(future_dates)):
        v = v * (1 + q_growth_scn)
        vals.append(v)

    f = pd.DataFrame({"date": future_dates, "forecast": vals})
    return f


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Product Forecast", "Total Forecast", "Income Statement", "Download"])

# ---- TAB 1: Segment / Product Forecast
with tab1:
    st.subheader(f"{segment_choice}: Historical + Scenario Forecast")

    hist_seg = seg_q[seg_q["segment"] == segment_choice].sort_values("date")

    # forecast
    fc = forecast_series(hist_seg[["date", "value"]], years, uplift)

    # merge for chart
    chart_hist = hist_seg[["date", "value"]].copy()
    chart_hist["type"] = "Historical"
    chart_hist["value_b"] = chart_hist["value"] / 1e9

    chart_fc = fc.copy()
    chart_fc["type"] = "Forecast"
    chart_fc["value_b"] = chart_fc["forecast"] / 1e9
    chart_fc = chart_fc[["date", "type", "value_b"]]

    chart_all = pd.concat([chart_hist[["date","type","value_b"]], chart_fc], ignore_index=True)

    fig = px.line(
        chart_all,
        x="date",
        y="value_b",
        color="type",
        title=f"{segment_choice} Revenue (Quarterly, USD Billions) | Forecast {years}y, uplift {uplift:.0%}",
        labels={"value_b": "Revenue (USD, Billions)", "date": "Quarter"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # KPIs
    if not hist_seg.empty:
        last_hist = hist_seg["value"].iloc[-1]
    else:
        last_hist = 0

    end_forecast = fc["forecast"].iloc[-1] if not fc.empty else last_hist
    delta = end_forecast - last_hist

    k1, k2, k3 = st.columns(3)
    k1.metric("Last Historical Quarter", f"${last_hist/1e9:,.2f}B")
    k2.metric("End of Forecast (Scenario)", f"${end_forecast/1e9:,.2f}B")
    k3.metric("Δ vs Last Historical", f"${delta/1e9:,.2f}B")

    st.caption("Forecast method: simple quarterly growth from last ~2 years + optional extra CAGR uplift.")


# ---- TAB 2: Total Forecast (sum of segments)
with tab2:
    st.subheader("Total Alphabet Segment Revenue (Scenario)")

    # Build total historical by date
    total_hist = seg_q.groupby("date", as_index=False)["value"].sum()
    total_hist = total_hist.sort_values("date")
    total_hist["value_b"] = total_hist["value"] / 1e9

    # Build scenario forecast by summing per segment forecasts
    dates_future = None
    total_future_vals = None

    for seg in seg_q["segment"].unique():
        h = seg_q[seg_q["segment"] == seg].sort_values("date")
        f = forecast_series(h[["date","value"]], years, uplift)
        if f.empty:
            continue
        if dates_future is None:
            dates_future = f["date"].values
            total_future_vals = f["forecast"].values
        else:
            # align by date (should match)
            total_future_vals = total_future_vals + f["forecast"].values

    if dates_future is None:
        st.error("Not enough data to build total forecast.")
    else:
        total_fc = pd.DataFrame({"date": pd.to_datetime(dates_future), "forecast": total_future_vals})
        total_fc["value_b"] = total_fc["forecast"] / 1e9

        hist_plot = total_hist.copy()
        hist_plot["type"] = "Historical"
        hist_plot.rename(columns={"value_b":"y"}, inplace=True)

        fc_plot = total_fc[["date","value_b"]].copy()
        fc_plot["type"] = "Forecast"
        fc_plot.rename(columns={"value_b":"y"}, inplace=True)

        plot_df = pd.concat([hist_plot[["date","type","y"]], fc_plot[["date","type","y"]]], ignore_index=True)

        fig2 = px.line(
            plot_df,
            x="date",
            y="y",
            color="type",
            title=f"Total Segment Revenue (USD Billions) | Forecast {years}y, uplift {uplift:.0%}",
            labels={"y": "Revenue (USD, Billions)", "date": "Quarter"},
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.caption("Total forecast = sum of per-segment scenario forecasts (not a separate fitted model).")


# ---- TAB 3: Income Statement
with tab3:
    st.subheader("Income Statement (Quarterly)")

    metrics = sorted(inc_q["metric"].unique().tolist())
    metric_choice = st.selectbox("Metric", metrics, index=0)

    hist_inc = inc_q[inc_q["metric"] == metric_choice].sort_values("date")

    fig3 = px.line(
        hist_inc,
        x="date",
        y="value_b",
        title=f"{metric_choice} (USD Billions)",
        labels={"value_b": "USD (Billions)", "date": "Quarter"},
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(hist_inc[["date","metric","value"]].rename(columns={"value": "value_usd"}), use_container_width=True)


# ---- TAB 4: Download
with tab4:
    st.subheader("Download Data")

    st.write("Segment revenue (quarterly) dataset:")
    st.download_button(
        "Download segment revenue CSV",
        data=seg_q.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_segment_revenue_quarterly.csv",
        mime="text/csv",
    )

    st.write("Income statement (quarterly) dataset:")
    st.download_button(
        "Download income statement CSV",
        data=inc_q.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_income_statement_quarterly.csv",
        mime="text/csv",
    )

    st.caption(
        "If the app is running on Streamlit Cloud and live scraping is blocked, "
        "these downloads may come from cached repo files."
    )
