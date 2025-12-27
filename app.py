# app.py
# Alphabet (Google) Revenue Forecast Dashboard (Streamlit)
# - Scrapes StockAnalysis segment revenue (revenue-by-segment) + income statement (financials)
# - Robust table detection (does NOT rely on "History" heading)
# - Uses quarterly data; if page returns TTM-only / blocked, app will show a clear error message
# - Forecast tabs: Product Forecast + Total Forecast + Income Statement + Download

import re
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
SEGMENT_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
INCOME_URL  = "https://stockanalysis.com/stocks/goog/financials/"
UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://stockanalysis.com/",
}

st.set_page_config(page_title="Alphabet (Google) Revenue Forecast Dashboard", layout="wide")


# -----------------------------
# HELPERS
# -----------------------------
def money_to_float(x) -> float:
    """Convert strings like '56.57B', '344.00M', '-207.00M', '0' to float USD."""
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan

    s = s.replace(",", "").replace("−", "-").replace("\xa0", " ").strip()

    if s in {"-", "—"}:
        return np.nan

    # already numeric
    try:
        return float(s)
    except Exception:
        pass

    mult = 1.0
    if s.endswith("T"):
        mult, s = 1e12, s[:-1]
    elif s.endswith("B"):
        mult, s = 1e9, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("K"):
        mult, s = 1e3, s[:-1]

    try:
        return float(s) * mult
    except Exception:
        m = re.search(r"-?\d+(\.\d+)?", s)
        return float(m.group(0)) * mult if m else np.nan


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


def _fetch_html(url: str) -> str:
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def _read_all_tables(html: str) -> list[pd.DataFrame]:
    # pandas read_html is usually best for StockAnalysis tables
    return pd.read_html(html)


def _looks_like_segment_table(df: pd.DataFrame) -> bool:
    """Heuristics to find the segment revenue table."""
    if df is None or df.empty:
        return False

    cols = [str(c).lower() for c in df.columns]
    # We want a Date column + at least 3 segment columns
    has_date = any("date" == c or c.startswith("date") for c in cols)
    if not has_date or df.shape[1] < 4:
        return False

    # At least one known segment name appears in headers
    header_text = " ".join(cols)
    must_have_any = [
        "google search", "youtube", "google cloud", "other bets", "network", "subscriptions"
    ]
    if not any(k in header_text for k in must_have_any):
        return False

    # Must contain dates that parse
    date_col = df.columns[0]
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    return parsed.notna().sum() >= max(3, int(0.5 * len(df)))


def _looks_like_income_table(df: pd.DataFrame) -> bool:
    """Heuristics to find an income statement table."""
    if df is None or df.empty:
        return False

    cols = [str(c).lower() for c in df.columns]
    # first col usually has line items
    first_col_name = str(df.columns[0]).lower()
    if "revenue" in " ".join(df.iloc[:10, 0].astype(str).str.lower().tolist()):
        return True
    if "income statement" in first_col_name:
        return True

    # If table contains common income statement rows
    sample_text = " ".join(df.iloc[:25, 0].astype(str).str.lower().tolist())
    keys = ["revenue", "gross profit", "operating income", "net income"]
    return sum(k in sample_text for k in keys) >= 2


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_segment_quarterly_tidy() -> pd.DataFrame:
    """Return tidy df: date, product, revenue (quarterly)."""
    html = _fetch_html(SEGMENT_URL)
    tables = _read_all_tables(html)

    seg = None
    for t in tables:
        # Ensure first column is treated as Date if it is
        if _looks_like_segment_table(t):
            seg = t.copy()
            break

    if seg is None:
        # If blocked or layout changed: try BeautifulSoup fallback: pick largest table and retry
        soup = BeautifulSoup(html, "lxml")
        all_tables = soup.find_all("table")
        if not all_tables:
            raise ValueError("No tables found on the segment page (blocked or layout changed).")

        # Use pandas on the largest table html snippet
        largest = max(all_tables, key=lambda x: len(x.find_all("tr")))
        seg = pd.read_html(str(largest))[0]
        if not _looks_like_segment_table(seg):
            raise ValueError("Could not find the quarterly segment revenue table on the page.")

    # Normalize columns: first col should be Date
    seg = seg.copy()
    seg.rename(columns={seg.columns[0]: "date"}, inplace=True)

    # Drop any TTM columns if present
    drop_cols = [c for c in seg.columns if str(c).strip().upper() == "TTM"]
    if drop_cols:
        seg = seg.drop(columns=drop_cols)

    seg["date"] = pd.to_datetime(seg["date"], errors="coerce")
    seg = seg.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in seg.columns:
        if c != "date":
            seg[c] = seg[c].apply(money_to_float)

    tidy = seg.melt("date", var_name="product", value_name="revenue").dropna()
    tidy = tidy.sort_values(["product", "date"]).reset_index(drop=True)

    # Safety: ensure quarterly-ish frequency (at least 4 points)
    if tidy["date"].nunique() < 4:
        raise ValueError("Segment data parsed but looks incomplete. (Too few quarters)")

    return tidy


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_income_statement_wide() -> pd.DataFrame:
    """Return wide income statement table (as shown on StockAnalysis)."""
    html = _fetch_html(INCOME_URL)
    tables = _read_all_tables(html)

    inc = None
    for t in tables:
        if _looks_like_income_table(t):
            inc = t.copy()
            break

    if inc is None:
        raise ValueError("Could not find income statement table on the financials page.")

    # Normalize first column name
    inc = inc.copy()
    inc.rename(columns={inc.columns[0]: "Line Item"}, inplace=True)
    return inc


def estimate_q_growth(series: pd.Series, lookback_quarters: int = 8) -> tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) < 6:
        return 0.02, 0.05
    s = s.iloc[-lookback_quarters:] if len(s) > lookback_quarters else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 3:
        return 0.02, 0.05
    return float(g.mean()), float(g.std(ddof=1) if len(g) > 1 else 0.05)


def forecast_series(hist_df: pd.DataFrame, years: int, uplift: float, lookback_quarters: int = 8) -> pd.DataFrame:
    """Forecast next years*4 quarters from a history df with columns [date, revenue]."""
    df = hist_df.sort_values("date").dropna(subset=["revenue"]).copy()
    last_date = df["date"].max()
    last_val = float(df.loc[df["date"] == last_date, "revenue"].iloc[-1])

    mean_q, std_q = estimate_q_growth(df["revenue"], lookback_quarters=lookback_quarters)

    uplift_q = (1.0 + uplift) ** (1.0 / 4.0) - 1.0
    q_growth = mean_q + uplift_q

    steps = years * 4
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=steps, freq="Q")

    fc, hi, lo = [], [], []
    cur = last_val
    base_std = std_q if (std_q and std_q > 0) else 0.05

    for i in range(1, steps + 1):
        cur = cur * (1.0 + q_growth)
        fc.append(cur)
        band = base_std * np.sqrt(i)  # widening band
        hi.append(cur * (1.0 + band))
        lo.append(cur * (1.0 - band))

    return pd.DataFrame({"date": future_dates, "forecast": fc, "hi": hi, "lo": lo})


def build_plot(hist: pd.DataFrame, fc: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["revenue"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["hi"], mode="lines", name="80% high (approx)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lo"], mode="lines", name="80% low (approx)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], mode="lines", name="Forecast (Scenario)"))
    fig.update_layout(
        title=title,
        height=520,
        xaxis_title="Quarter",
        yaxis_title="Revenue (USD)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# -----------------------------
# UI
# -----------------------------
# Sidebar
st.sidebar.title("Controls")
force_refresh = st.sidebar.button("Force Refresh (ignore cache)")

if force_refresh:
    st.cache_data.clear()

# Load data (segment + income)
try:
    tidy = load_segment_quarterly_tidy()
except Exception as e:
    st.error("Segment scrape failed.")
    st.write(f"Reason: {e}")
    st.stop()

try:
    income_wide = load_income_statement_wide()
    income_ok = True
except Exception as e:
    income_ok = False
    income_error = str(e)

products = sorted(tidy["product"].unique().tolist())
default_product = "Advertising" if "Advertising" in products else products[0]

product = st.sidebar.selectbox("Product", products, index=products.index(default_product))
years = st.sidebar.slider("Forecast years", 1, 10, 3, 1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario)", 0.00, 0.30, 0.00, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Notes")
st.sidebar.write("Live data is scraped from StockAnalysis.")
st.sidebar.write("Forecast uses recent quarterly growth with an optional annual uplift.")

# Header
c1, c2 = st.columns([1, 12])
with c1:
    st.image(
        "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png",
        use_container_width=True,
    )
with c2:
    st.markdown(
        """
# Alphabet (Google) Revenue Forecast Dashboard
Interactive product-level revenue forecasting with a scenario-based CAGR uplift. Use the sliders to stress-test assumptions and export results.
"""
    )

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Product Forecast", "Total Forecast", "Income Statement", "Assumptions & Notes", "Download"]
)

# -----------------------------
# TAB 1: Product Forecast
# -----------------------------
with tab1:
    seg = tidy[tidy["product"] == product].copy().sort_values("date")
    fc = forecast_series(seg[["date", "revenue"]], years=years, uplift=uplift)

    end_fc = float(fc["forecast"].iloc[-1])
    base_end = float(forecast_series(seg[["date", "revenue"]], years=years, uplift=0.0)["forecast"].iloc[-1])
    delta = end_fc - base_end

    k1, k2, k3 = st.columns(3)
    k1.metric("End Forecast (Scenario)", money_fmt(end_fc))
    k2.metric("Δ vs Baseline", money_fmt(delta))
    k3.metric("Last Reported Quarter", seg["date"].max().strftime("%b %d, %Y"))

    fig = build_plot(
        seg[["date", "revenue"]],
        fc,
        title=f"{product}: Historical + {years}-Year Forecast (uplift {uplift*100:.1f}%)",
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 2: Total Forecast
# -----------------------------
with tab2:
    wide = tidy.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide = wide.dropna(how="all")
    wide["TOTAL"] = wide.sum(axis=1)

    future_steps = years * 4
    last_date = wide.index.max()
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=future_steps, freq="Q")

    total_fc_vals = np.zeros(future_steps, dtype=float)
    total_hi_vals = np.zeros(future_steps, dtype=float)
    total_lo_vals = np.zeros(future_steps, dtype=float)

    for p in [c for c in wide.columns if c != "TOTAL"]:
        hist_p = wide[[p]].reset_index().rename(columns={p: "revenue"})
        fcp = forecast_series(hist_p[["date", "revenue"]], years=years, uplift=uplift)
        total_fc_vals += fcp["forecast"].values
        total_hi_vals += fcp["hi"].values
        total_lo_vals += fcp["lo"].values

    total_hist = wide["TOTAL"].reset_index().rename(columns={"TOTAL": "revenue"})
    total_fc = pd.DataFrame({"date": future_dates, "forecast": total_fc_vals, "hi": total_hi_vals, "lo": total_lo_vals})

    fig_total = build_plot(total_hist[["date", "revenue"]], total_fc, title="Total Alphabet Revenue Forecast (Scenario)")
    st.plotly_chart(fig_total, use_container_width=True)
    st.caption("This total forecast is the sum of per-segment scenario forecasts (not a separately fit model).")

# -----------------------------
# TAB 3: Income Statement
# -----------------------------
with tab3:
    st.subheader("Income Statement (from StockAnalysis financials)")
    if not income_ok:
        st.warning("Income statement scrape failed.")
        st.write(f"Reason: {income_error}")
    else:
        st.dataframe(income_wide, use_container_width=True)

# -----------------------------
# TAB 4: Assumptions
# -----------------------------
with tab4:
    st.subheader("What this dashboard is")
    st.write(
        """
This dashboard pulls Alphabet segment revenue history and produces a simple scenario forecast.
You choose:
- A product/segment (left sidebar)
- Forecast horizon (years)
- Extra CAGR uplift (scenario stress test)
"""
    )

    st.subheader("Forecast method (simple + explainable)")
    st.write(
        """
- Computes recent **quarter-over-quarter growth** from the last few quarters
- Uses that average growth as the baseline forward growth rate
- Applies your **extra CAGR uplift** (annual) as an additional quarterly growth component
- Adds an approximate uncertainty band that widens over time (based on recent growth volatility)
"""
    )

# -----------------------------
# TAB 5: Download
# -----------------------------
with tab5:
    st.subheader("Download data")
    st.write("Use these downloads to share your results or reuse the dataset.")

    tidy_out = tidy.copy()
    tidy_out["revenue"] = tidy_out["revenue"].astype(float)

    st.download_button(
        "Download tidy segment dataset (date, product, revenue) as CSV",
        data=tidy_out.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_revenue_by_segment_tidy.csv",
        mime="text/csv",
    )

    wide_out = tidy.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_out = wide_out.reset_index()

    st.download_button(
        "Download wide segment dataset (date + all segments) as CSV",
        data=wide_out.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_revenue_by_segment_wide.csv",
        mime="text/csv",
    )

    if income_ok:
        st.download_button(
            "Download income statement table as CSV",
            data=income_wide.to_csv(index=False).encode("utf-8"),
            file_name="alphabet_income_statement.csv",
            mime="text/csv",
        )

    st.caption("Sources: StockAnalysis → GOOG → Revenue by Segment, Financials")
