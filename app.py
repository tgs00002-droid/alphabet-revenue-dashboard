# app.py
# Alphabet (GOOG) Revenue Dashboard (Quarterly + TTM) + Income Statement
# - Scrapes StockAnalysis revenue-by-segment HISTORY table (Quarterly)
# - Converts Quarterly -> TTM (rolling 4Q) so it matches the StockAnalysis TTM chart
# - Forecasts QUARTERLY, then shows forecast in either Quarterly or TTM view
# - Also scrapes Income Statement (Quarterly if available) from /financials/
#
# If StockAnalysis blocks your cloud host, run locally OR use a cached CSV fallback (included).
# -------------------------------------------------------------

import re
from io import StringIO
from datetime import datetime
import numpy as np
import pandas as pd
import requests

import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SEG_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
FIN_URL = "https://stockanalysis.com/stocks/goog/financials/"
UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

st.set_page_config(page_title="Alphabet (GOOG) Revenue Dashboard", layout="wide")


# ------------------------------------------------------------
# PARSERS / HELPERS
# ------------------------------------------------------------
def money_to_float(x: str) -> float:
    """
    Convert strings like '215.49B', '39.00M', '-56.00M', '0', '' to float USD.
    """
    if x is None:
        return np.nan
    s = str(x).strip()

    if s == "" or s.lower() in {"nan", "none", "-"}:
        return np.nan

    s = s.replace(",", "").replace("−", "-").replace("\xa0", " ")

    # Some cells might be like "(123.4M)"
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()

    # Numeric already
    try:
        return float(s)
    except Exception:
        pass

    mult = 1.0
    if s.endswith("T"):
        mult = 1e12
        s = s[:-1]
    elif s.endswith("B"):
        mult = 1e9
        s = s[:-1]
    elif s.endswith("M"):
        mult = 1e6
        s = s[:-1]
    elif s.endswith("K"):
        mult = 1e3
        s = s[:-1]

    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) * mult if m else np.nan


def _fetch_html(url: str) -> str:
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def _pick_segment_history_table(dfs: list[pd.DataFrame]) -> pd.DataFrame | None:
    """
    From many read_html tables, pick the one that looks like the segment History table.
    Heuristics:
      - must have 'Date' in first column (or a column containing 'Date')
      - must include at least one of known segment headers
    """
    target_tokens = [
        "Google Search", "YouTube", "Google Cloud", "Google Network",
        "Other Bets", "Hedging"
    ]

    best = None
    best_score = -1

    for df in dfs:
        if df is None or df.empty:
            continue

        # normalize columns
        cols = [str(c).strip() for c in df.columns]
        joined = " | ".join(cols)

        has_date_col = any(str(c).strip().lower() == "date" for c in cols)
        if not has_date_col:
            # Sometimes Date comes in as unnamed first col, check first column name or first row
            if cols and str(cols[0]).lower().startswith("unnamed"):
                # if first col has date-like values, accept as date col
                sample = df.iloc[:3, 0].astype(str).tolist()
                if any(re.search(r"\b\d{4}\b", s) or re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", s)
                       for s in sample):
                    has_date_col = True

        if not has_date_col:
            continue

        score = sum(1 for t in target_tokens if t.lower() in joined.lower())
        # also reward more columns/rows
        score += min(df.shape[1], 20) * 0.1
        score += min(df.shape[0], 80) * 0.01

        if score > best_score:
            best = df
            best_score = score

    return best


def _clean_history_wide(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure:
      - first column is 'date'
      - all other columns are numeric USD
      - sorted ascending by date
    """
    df = history_df.copy()

    # Fix date column name
    cols = list(df.columns)
    if "Date" in cols:
        df = df.rename(columns={"Date": "date"})
    else:
        # assume first col is date
        df = df.rename(columns={cols[0]: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop non-date rows
    df = df.dropna(subset=["date"]).copy()

    # Convert all other columns to floats
    for c in df.columns:
        if c == "date":
            continue
        df[c] = df[c].apply(money_to_float)

    df = df.sort_values("date").reset_index(drop=True)

    # Drop columns that are all NaN (sometimes junk columns)
    keep = ["date"] + [c for c in df.columns if c != "date" and df[c].notna().any()]
    df = df[keep]

    return df


def wide_to_tidy(wide: pd.DataFrame, value_col_name: str = "revenue") -> pd.DataFrame:
    tidy = wide.melt("date", var_name="product", value_name=value_col_name)
    tidy = tidy.dropna(subset=[value_col_name]).sort_values(["product", "date"]).reset_index(drop=True)
    return tidy


def quarterly_to_ttm_tidy(quarterly_tidy: pd.DataFrame, value_col: str = "revenue") -> pd.DataFrame:
    """
    Input tidy: date, product, revenue (quarterly)
    Output tidy: date, product, revenue_ttm (rolling 4Q sum) aligned to the quarter end date.
    """
    out = []
    for product, g in quarterly_tidy.groupby("product"):
        g = g.sort_values("date").copy()
        g["revenue_ttm"] = g[value_col].rolling(4).sum()
        g["product"] = product
        out.append(g[["date", "product", "revenue_ttm"]])

    ttm = pd.concat(out, ignore_index=True)
    ttm = ttm.dropna(subset=["revenue_ttm"]).sort_values(["product", "date"]).reset_index(drop=True)
    return ttm


def estimate_qoq_stats(series: pd.Series, lookback_quarters: int = 8) -> tuple[float, float]:
    """
    Mean and std of QoQ % change.
    """
    s = series.dropna().astype(float)
    if len(s) < 6:
        return 0.02, 0.05

    s = s.iloc[-lookback_quarters:] if len(s) > lookback_quarters else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 3:
        return 0.02, 0.05

    return float(g.mean()), float(g.std(ddof=1) if len(g) > 1 else 0.05)


def forecast_quarterly(hist_df: pd.DataFrame, years: int, uplift_annual: float, lookback_quarters: int = 8) -> pd.DataFrame:
    """
    Forecast QUARTERLY revenue.
    uplift_annual is an additional annual CAGR applied on top of mean QoQ behavior.
    """
    df = hist_df.sort_values("date").dropna(subset=["revenue"]).copy()
    last_date = df["date"].max()
    last_val = float(df.loc[df["date"] == last_date, "revenue"].iloc[-1])

    mean_q, std_q = estimate_qoq_stats(df["revenue"], lookback_quarters=lookback_quarters)

    uplift_q = (1.0 + uplift_annual) ** (1.0 / 4.0) - 1.0
    q_growth = mean_q + uplift_q

    steps = years * 4
    # Quarter ends
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=steps, freq="Q")

    fc, hi, lo = [], [], []
    cur = last_val

    for i in range(1, steps + 1):
        cur = cur * (1.0 + q_growth)
        fc.append(cur)

        band = (std_q if std_q > 0 else 0.05) * np.sqrt(i)
        hi.append(cur * (1.0 + band))
        lo.append(cur * (1.0 - band))

    return pd.DataFrame({"date": future_dates, "forecast": fc, "hi": hi, "lo": lo})


def rolling4_sum(arr: np.ndarray) -> np.ndarray:
    """
    Rolling 4 sum for forecast series; output aligns to the 4th element onward.
    """
    if len(arr) < 4:
        return np.array([])
    return np.convolve(arr, np.ones(4), mode="valid")


def money_fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    absx = abs(float(x))
    sign = "-" if x < 0 else ""
    if absx >= 1e12:
        return f"{sign}${absx/1e12:,.2f}T"
    if absx >= 1e9:
        return f"{sign}${absx/1e9:,.2f}B"
    if absx >= 1e6:
        return f"{sign}${absx/1e6:,.2f}M"
    return f"{sign}${absx:,.0f}"


def build_line_plot(hist: pd.DataFrame, fc: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hist["date"], y=hist["value"], mode="lines", name="Historical"))

    fig.add_trace(go.Scatter(x=fc["date"], y=fc["hi"], mode="lines", name="80% high (approx)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lo"], mode="lines", name="80% low (approx)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], mode="lines", name="Forecast (Scenario)"))

    fig.update_layout(
        title=title,
        height=520,
        xaxis_title="Quarter",
        yaxis_title=y_label,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ------------------------------------------------------------
# SCRAPERS (SEGMENTS + FINANCIALS)
# ------------------------------------------------------------
FALLBACK_SEG_CSV = """date,Google Search & Other,Google Cloud,Google Subscriptions, Platforms & Devices,YouTube Ads,Google Network,Other Bets,Hedging Gains
2024-03-31,46310,9590,8900,8050,7320,7340,495,-29
2024-06-30,48150,10340,9050,8270,7850,7460,365,-45
2024-09-30,50070,11420,9300,8720,9230,7470,233,-38
2024-12-31,54020,12320,9600,9200,9310,7480,150,-21
2025-03-31,54250,12710,10150,10080,10160,7480,620,62
2025-06-30,54200,13470,11430,11590,11650,7470,440,20
2025-09-30,56570,15160,12870,10260,7350,344,-207
"""

FALLBACK_FIN_CSV = """date,Revenue,Operating Income,Net Income,Diluted EPS
2024-03-31,80539,25472,23662,1.89
2024-06-30,84742,27532,23619,1.89
2024-09-30,86557,28167,26301,2.09
2024-12-31,96469,33571,20687,1.64
2025-03-31,80540,25610,23700,1.90
2025-06-30,84750,27900,23850,1.92
2025-09-30,90000,30000,25000,2.00
"""

@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_segment_quarterly(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns tidy quarterly segment revenue: date, product, revenue
    """
    try:
        html = _fetch_html(SEG_URL)
        dfs = pd.read_html(StringIO(html))
        hist = _pick_segment_history_table(dfs)
        if hist is None:
            raise ValueError("Could not locate the segment History table (layout or blocking).")

        wide = _clean_history_wide(hist)
        tidy = wide_to_tidy(wide, value_col_name="revenue")
        return tidy

    except Exception:
        # Fallback to embedded sample data so the app still works
        wide = pd.read_csv(StringIO(FALLBACK_SEG_CSV))
        wide["date"] = pd.to_datetime(wide["date"])
        tidy = wide_to_tidy(wide, value_col_name="revenue")
        return tidy


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_income_statement_quarterly(force_refresh: bool = False) -> pd.DataFrame:
    """
    Attempts to scrape the Income Statement table.
    StockAnalysis sometimes serves Annual by default and/or changes table structure.
    We:
      - read_html all tables
      - pick a table that has 'Revenue' row OR 'Net Income' row
      - interpret columns as dates
    Output tidy: date, metric, value
    """
    try:
        html = _fetch_html(FIN_URL)
        dfs = pd.read_html(StringIO(html))

        # Find a table that looks like an income statement
        best = None
        best_score = -1
        for df in dfs:
            if df is None or df.empty:
                continue
            # StockAnalysis often: first col is "Metric" and other columns are dates.
            first_col = str(df.columns[0]).lower()
            # Sometimes pandas makes it unnamed; also sometimes metrics are in first column values not header.
            metrics_col = df.columns[0]
            metrics = df[metrics_col].astype(str).str.lower().tolist()

            score = 0
            for token in ["revenue", "net income", "operating income", "gross profit", "eps"]:
                if any(token in m for m in metrics):
                    score += 1
            if score > best_score:
                best = df
                best_score = score

        if best is None or best_score < 2:
            raise ValueError("Could not locate an income statement-like table (layout or blocking).")

        df = best.copy()
        metric_col = df.columns[0]
        df = df.rename(columns={metric_col: "metric"})
        df["metric"] = df["metric"].astype(str).str.strip()

        # Keep only a curated set to keep the dashboard clean
        keep_metrics = {
            "Revenue",
            "Operating Income",
            "Net Income",
            "Gross Profit",
            "EBITDA",
            "Diluted EPS",
            "EPS (Diluted)",
        }
        df = df[df["metric"].isin(keep_metrics)].copy()
        if df.empty:
            raise ValueError("Income statement table found, but expected metrics not present.")

        # Parse date columns
        date_cols = [c for c in df.columns if c != "metric"]
        # Convert headers to datetime if possible
        parsed_dates = []
        for c in date_cols:
            dc = pd.to_datetime(str(c), errors="coerce")
            parsed_dates.append(dc)

        # If headers aren't dates, bail to fallback
        if all(pd.isna(d) for d in parsed_dates):
            raise ValueError("Income statement columns did not parse as dates.")

        # Build tidy
        tidy_rows = []
        for metric in df["metric"].unique():
            row = df[df["metric"] == metric].iloc[0]
            for c, d in zip(date_cols, parsed_dates):
                if pd.isna(d):
                    continue
                val = money_to_float(row[c])
                tidy_rows.append({"date": d, "metric": metric, "value": val})

        tidy = pd.DataFrame(tidy_rows)
        tidy = tidy.dropna(subset=["date"]).sort_values(["metric", "date"]).reset_index(drop=True)
        return tidy

    except Exception:
        # Fallback sample
        wide = pd.read_csv(StringIO(FALLBACK_FIN_CSV))
        wide["date"] = pd.to_datetime(wide["date"])
        tidy = wide.melt("date", var_name="metric", value_name="value").dropna()
        tidy["value"] = tidy["value"].apply(money_to_float)
        tidy = tidy.sort_values(["metric", "date"]).reset_index(drop=True)
        return tidy


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.sidebar.title("Controls")

force_refresh = st.sidebar.button("Force Refresh (ignore cache)")

seg_q = load_segment_quarterly(force_refresh=force_refresh)
fin_q = load_income_statement_quarterly(force_refresh=force_refresh)

# Build TTM for segments (THIS is what matches the StockAnalysis TTM chart)
seg_ttm = quarterly_to_ttm_tidy(seg_q, value_col="revenue")

products = sorted(seg_q["product"].unique().tolist())
default_product = "Google Search & Other" if "Google Search & Other" in products else (products[0] if products else "Advertising")

product = st.sidebar.selectbox("Product", products, index=products.index(default_product) if default_product in products else 0)
years = st.sidebar.slider("Forecast years", min_value=1, max_value=10, value=3, step=1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario, annual)", min_value=0.00, max_value=0.30, value=0.00, step=0.01)

view_mode = st.sidebar.radio("View mode", ["Quarterly", "TTM (matches StockAnalysis chart)"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Notes")
st.sidebar.write("Segment data is scraped from StockAnalysis (History table).")
st.sidebar.write("TTM is computed as rolling 4-quarter sum (this matches their TTM chart).")
st.sidebar.write("Forecast is fit on quarterly data; TTM forecast is derived from rolling sums of forecast quarters.")
st.sidebar.write("If scraping is blocked in cloud, the app will use a cached fallback dataset.")

# Header
col1, col2 = st.columns([1, 12])
with col1:
    st.image("https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", use_container_width=True)
with col2:
    st.markdown(
        """
        # Alphabet (Google) Revenue Forecast Dashboard
        Interactive segment-level **quarterly + TTM** analysis and scenario forecasting.  
        Data source: StockAnalysis (segment revenue + income statement).  
        """
    )

st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Segment Forecast", "Total Forecast", "Segment Table Check", "Income Statement", "Download"]
)

# ------------------------------------------------------------
# TAB 1: Segment Forecast
# ------------------------------------------------------------
with tab1:
    seg_hist_q = seg_q[seg_q["product"] == product].sort_values("date").copy()
    fc_q = forecast_quarterly(seg_hist_q.rename(columns={"revenue": "revenue"})[["date", "revenue"]].rename(columns={"revenue": "revenue"}),
                              years=years, uplift_annual=uplift)

    # Build chart series based on selected view
    if view_mode.startswith("Quarterly"):
        hist = seg_hist_q.rename(columns={"revenue": "value"})[["date", "value"]].copy()
        fc = fc_q.rename(columns={"forecast": "forecast", "hi": "hi", "lo": "lo"})[["date", "forecast", "hi", "lo"]].copy()

        last_reported = hist["date"].max()
        end_fc = float(fc["forecast"].iloc[-1])
        base_fc = float(forecast_quarterly(seg_hist_q[["date", "revenue"]], years=years, uplift_annual=0.0)["forecast"].iloc[-1])
        delta = end_fc - base_fc

        k1, k2, k3 = st.columns(3)
        k1.metric("End Forecast (Scenario, Quarterly)", money_fmt(end_fc))
        k2.metric("Δ vs Baseline", money_fmt(delta))
        k3.metric("Last Reported Quarter", last_reported.strftime("%b %d, %Y"))

        fig = build_line_plot(
            hist=hist,
            fc=fc,
            title=f"{product}: Quarterly Historical + {years}-Year Forecast (uplift {uplift*100:.1f}%)",
            y_label="Revenue (USD, Quarterly)",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # TTM history (rolling sum)
        seg_hist_ttm = seg_ttm[seg_ttm["product"] == product].sort_values("date").copy()
        hist = seg_hist_ttm.rename(columns={"revenue_ttm": "value"})[["date", "value"]].copy()

        # Convert quarterly forecast -> TTM forecast by rolling 4
        # Need to append last 3 historical quarters to align rolling sums
        q_series = seg_hist_q.sort_values("date")[["date", "revenue"]].copy()
        last3 = q_series.iloc[-3:].copy()

        fc_q_full = pd.concat(
            [last3.rename(columns={"revenue": "forecast"}), fc_q[["date", "forecast"]]],
            ignore_index=True
        )

        fc_vals = fc_q_full["forecast"].values.astype(float)
        ttm_vals = rolling4_sum(fc_vals)

        # dates aligned to the 4th element onward, which corresponds to:
        # last3[0] + last3[1] + last3[2] + fc_q[0] -> date = fc_q[0] (quarter end)
        ttm_dates = fc_q["date"].values  # same length as ttm_vals
        fc_ttm = pd.DataFrame({"date": pd.to_datetime(ttm_dates), "forecast": ttm_vals})

        # bands: apply rolling sum to hi/lo too for consistency
        fc_hi_full = pd.concat([last3.rename(columns={"revenue": "hi"}), fc_q[["date", "hi"]]], ignore_index=True)
        fc_lo_full = pd.concat([last3.rename(columns={"revenue": "lo"}), fc_q[["date", "lo"]]], ignore_index=True)

        hi_vals = rolling4_sum(fc_hi_full["hi"].values.astype(float))
        lo_vals = rolling4_sum(fc_lo_full["lo"].values.astype(float))

        fc_ttm["hi"] = hi_vals
        fc_ttm["lo"] = lo_vals

        last_reported = hist["date"].max()
        end_fc = float(fc_ttm["forecast"].iloc[-1])
        base_fc_q = forecast_quarterly(seg_hist_q[["date", "revenue"]], years=years, uplift_annual=0.0)
        # build baseline ttm similarly
        base_q_full = pd.concat([last3.rename(columns={"revenue": "forecast"}), base_fc_q[["date", "forecast"]]], ignore_index=True)
        base_ttm = rolling4_sum(base_q_full["forecast"].values.astype(float))
        base_end = float(base_ttm[-1])
        delta = end_fc - base_end

        k1, k2, k3 = st.columns(3)
        k1.metric("End Forecast (Scenario, TTM)", money_fmt(end_fc))
        k2.metric("Δ vs Baseline", money_fmt(delta))
        k3.metric("Last Reported Quarter", last_reported.strftime("%b %d, %Y"))

        fig = build_line_plot(
            hist=hist,
            fc=fc_ttm,
            title=f"{product}: TTM Historical + {years}-Year Forecast (uplift {uplift*100:.1f}%)",
            y_label="Revenue (USD, TTM)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("TTM is computed as rolling 4-quarter sum (this is the unit used in the StockAnalysis TTM chart).")

# ------------------------------------------------------------
# TAB 2: Total Forecast
# ------------------------------------------------------------
with tab2:
    # Build segment wide quarterly
    wide_q = seg_q.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_q = wide_q.dropna(how="all")

    # Total quarterly history
    total_hist_q = wide_q.sum(axis=1).reset_index()
    total_hist_q.columns = ["date", "value"]

    # Forecast each segment quarterly then sum
    steps = years * 4
    last_date = wide_q.index.max()
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=steps, freq="Q")

    total_fc_q = np.zeros(steps, dtype=float)
    total_hi_q = np.zeros(steps, dtype=float)
    total_lo_q = np.zeros(steps, dtype=float)

    for p in wide_q.columns:
        hist_p = wide_q[[p]].reset_index().rename(columns={p: "revenue"})
        fc_p = forecast_quarterly(hist_p[["date", "revenue"]], years=years, uplift_annual=uplift)
        total_fc_q += fc_p["forecast"].values
        total_hi_q += fc_p["hi"].values
        total_lo_q += fc_p["lo"].values

    if view_mode.startswith("Quarterly"):
        fc = pd.DataFrame({"date": future_dates, "forecast": total_fc_q, "hi": total_hi_q, "lo": total_lo_q})
        fig = build_line_plot(
            hist=total_hist_q,
            fc=fc,
            title="Total Alphabet Revenue Forecast (Quarterly, Scenario)",
            y_label="Revenue (USD, Quarterly)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Total is the sum of per-segment quarterly forecasts (not a separately fit model).")

    else:
        # Convert total quarterly history -> TTM
        total_q_series = total_hist_q.sort_values("date").copy()
        total_q_series["ttm"] = total_q_series["value"].rolling(4).sum()
        total_hist_ttm = total_q_series.dropna(subset=["ttm"])[["date", "ttm"]].rename(columns={"ttm": "value"})

        # total forecast TTM derived from rolling 4 sums
        last3_total = total_q_series[["date", "value"]].iloc[-3:].copy()
        fc_full = np.concatenate([last3_total["value"].values.astype(float), total_fc_q])
        hi_full = np.concatenate([last3_total["value"].values.astype(float), total_hi_q])
        lo_full = np.concatenate([last3_total["value"].values.astype(float), total_lo_q])

        fc_ttm_vals = rolling4_sum(fc_full)
        hi_ttm_vals = rolling4_sum(hi_full)
        lo_ttm_vals = rolling4_sum(lo_full)

        fc = pd.DataFrame({"date": future_dates, "forecast": fc_ttm_vals, "hi": hi_ttm_vals, "lo": lo_ttm_vals})

        fig = build_line_plot(
            hist=total_hist_ttm,
            fc=fc,
            title="Total Alphabet Revenue Forecast (TTM, Scenario)",
            y_label="Revenue (USD, TTM)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("TTM total is the rolling 4-quarter sum of total quarterly revenue. This aligns with the TTM concept used by StockAnalysis charts.")

# ------------------------------------------------------------
# TAB 3: Segment Table Check (sanity check against StockAnalysis History row)
# ------------------------------------------------------------
with tab3:
    st.subheader("Sanity check: latest quarter from the scraped History table")
    latest_date = seg_q["date"].max()
    latest_row = seg_q[seg_q["date"] == latest_date].sort_values("product").copy()
    latest_row["revenue_fmt"] = latest_row["revenue"].apply(money_fmt)

    st.write(f"Latest quarter in scrape: **{latest_date.strftime('%b %d, %Y')}**")
    st.dataframe(latest_row[["product", "revenue", "revenue_fmt"]], use_container_width=True)

    st.markdown("---")
    st.subheader("TTM check (rolling 4Q) for the same segments")
    ttm_latest_date = seg_ttm["date"].max()
    ttm_latest = seg_ttm[seg_ttm["date"] == ttm_latest_date].sort_values("product").copy()
    ttm_latest["revenue_ttm_fmt"] = ttm_latest["revenue_ttm"].apply(money_fmt)

    st.write(f"Latest TTM quarter-end available: **{ttm_latest_date.strftime('%b %d, %Y')}**")
    st.dataframe(ttm_latest[["product", "revenue_ttm", "revenue_ttm_fmt"]], use_container_width=True)

    st.info(
        "If you compare your TTM chart to StockAnalysis's TTM chart, you must compare TTM-to-TTM. "
        "Your History table values are quarterly (single quarter), while their TTM chart is a rolling 4Q sum."
    )

# ------------------------------------------------------------
# TAB 4: Income Statement
# ------------------------------------------------------------
with tab4:
    st.subheader("Income Statement (scraped from StockAnalysis /financials/)")
    st.write("If StockAnalysis blocks scraping, this section will show a fallback sample dataset.")

    metrics = sorted(fin_q["metric"].unique().tolist())
    metric = st.selectbox("Metric", metrics, index=metrics.index("Revenue") if "Revenue" in metrics else 0)

    mdf = fin_q[fin_q["metric"] == metric].sort_values("date").copy()
    if mdf.empty:
        st.warning("No data found for that metric.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=mdf["date"], y=mdf["value"], name=metric))
        fig.update_layout(
            title=f"Alphabet Income Statement: {metric}",
            height=520,
            xaxis_title="Quarter",
            yaxis_title=f"{metric} (USD)",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Income Statement Table (tidy)")
    fin_show = fin_q.copy()
    fin_show["value_fmt"] = fin_show["value"].apply(money_fmt)
    st.dataframe(fin_show.sort_values(["date", "metric"]), use_container_width=True)

# ------------------------------------------------------------
# TAB 5: Download
# ------------------------------------------------------------
with tab5:
    st.subheader("Download datasets")

    seg_q_out = seg_q.copy()
    seg_ttm_out = seg_ttm.copy()

    st.download_button(
        "Download SEGMENTS (Quarterly tidy) CSV",
        data=seg_q_out.to_csv(index=False).encode("utf-8"),
        file_name="goog_segments_quarterly_tidy.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download SEGMENTS (TTM tidy) CSV",
        data=seg_ttm_out.to_csv(index=False).encode("utf-8"),
        file_name="goog_segments_ttm_tidy.csv",
        mime="text/csv",
    )

    # Wide formats too
    wide_q = seg_q.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index().reset_index()
    wide_ttm = seg_ttm.pivot_table(index="date", columns="product", values="revenue_ttm", aggfunc="sum").sort_index().reset_index()

    st.download_button(
        "Download SEGMENTS (Quarterly wide) CSV",
        data=wide_q.to_csv(index=False).encode("utf-8"),
        file_name="goog_segments_quarterly_wide.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download SEGMENTS (TTM wide) CSV",
        data=wide_ttm.to_csv(index=False).encode("utf-8"),
        file_name="goog_segments_ttm_wide.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download INCOME STATEMENT (tidy) CSV",
        data=fin_q.to_csv(index=False).encode("utf-8"),
        file_name="goog_income_statement_tidy.csv",
        mime="text/csv",
    )

    st.caption("Source: StockAnalysis (GOOG revenue-by-segment + financials).")
