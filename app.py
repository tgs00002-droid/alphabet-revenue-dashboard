import re
from io import StringIO
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
SOURCE_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

st.set_page_config(
    page_title="Alphabet (Google) Revenue Forecast Dashboard",
    layout="wide",
)

# -----------------------------
# HELPERS
# -----------------------------
def money_to_float(x: str) -> float:
    """
    Convert strings like '215.49B', '39.00M', '-56.00M', '0', '' to float USD.
    """
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    # Remove commas
    s = s.replace(",", "")

    # Sometimes weird unicode minus or NBSP
    s = s.replace("−", "-").replace("\xa0", " ")

    # If it's already numeric
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

    # Final parse
    try:
        return float(s) * mult
    except Exception:
        # last resort: grab first number in string
        m = re.search(r"-?\d+(\.\d+)?", s)
        return float(m.group(0)) * mult if m else np.nan


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_revenue_by_segment() -> pd.DataFrame:
    """
    Scrape the 'History' table from StockAnalysis and return tidy dataframe:
      columns: date, product, revenue
    """
    r = requests.get(SOURCE_URL, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    # Find the History section header, then the first table after it
    # Page structure can change; this approach is resilient.
    history_h2 = None
    for h in soup.find_all(["h2", "h3"]):
        if h.get_text(strip=True).lower() == "history":
            history_h2 = h
            break

    table = None
    if history_h2:
        nxt = history_h2.find_next("table")
        table = nxt

    # Fallback: just pick the largest table on the page
    if table is None:
        tables = soup.find_all("table")
        if not tables:
            raise ValueError("No tables found on the page.")
        table = max(tables, key=lambda t: len(t.find_all("tr")))

    # Extract headers
    thead = table.find("thead")
    if thead:
        header_cells = thead.find_all(["th", "td"])
        headers = [c.get_text(" ", strip=True) for c in header_cells]
    else:
        # sometimes header row is in first tbody row
        first_row = table.find("tr")
        headers = [c.get_text(" ", strip=True) for c in first_row.find_all(["th", "td"])]

    headers = [h.replace("\xa0", " ").strip() for h in headers]
    # Ensure first col is Date
    if headers and headers[0].lower() != "date":
        headers[0] = "Date"

    # Extract rows
    rows = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        vals = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in cells]

        # skip header-like row
        if vals and vals[0].lower() == "date":
            continue

        # Align to headers length
        if len(vals) < len(headers):
            vals = vals + [""] * (len(headers) - len(vals))
        elif len(vals) > len(headers):
            vals = vals[: len(headers)]

        rows.append(vals)

    wide = pd.DataFrame(rows, columns=headers)

    # Clean + parse date
    wide.rename(columns={"Date": "date"}, inplace=True)
    wide["date"] = pd.to_datetime(wide["date"], errors="coerce")

    # Parse all numeric columns
    for c in wide.columns:
        if c != "date":
            wide[c] = wide[c].apply(money_to_float)

    wide = wide.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Tidy
    tidy = wide.melt("date", var_name="product", value_name="revenue").dropna()
    tidy = tidy.sort_values(["product", "date"]).reset_index(drop=True)

    return tidy


def estimate_cagr_quarterly(series: pd.Series, lookback_quarters: int = 8) -> tuple[float, float]:
    """
    Estimate quarterly growth mean and std from last N quarters of history.
    Returns (mean_q_growth, std_q_growth) in decimal, e.g. 0.03 = 3% per quarter.
    """
    s = series.dropna().astype(float)
    if len(s) < 6:
        return 0.02, 0.05  # conservative fallback

    s = s.iloc[-lookback_quarters:] if len(s) > lookback_quarters else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 3:
        return 0.02, 0.05

    return float(g.mean()), float(g.std(ddof=1) if len(g) > 1 else 0.05)


def forecast_series(hist_df: pd.DataFrame, years: int, uplift: float, lookback_quarters: int = 8) -> pd.DataFrame:
    """
    Given a history df with columns [date, revenue], forecast next years*4 quarters.
    Uses average quarterly growth + optional uplift (annual, converted to quarterly).
    """
    df = hist_df.sort_values("date").copy()
    df = df.dropna(subset=["revenue"])
    last_date = df["date"].max()
    last_val = float(df.loc[df["date"] == last_date, "revenue"].iloc[-1])

    mean_q, std_q = estimate_cagr_quarterly(df["revenue"], lookback_quarters=lookback_quarters)

    # Convert annual uplift to quarterly approx:
    # uplift is "extra annual CAGR", so per quarter: (1+uplift)^(1/4)-1
    uplift_q = (1.0 + uplift) ** (1.0 / 4.0) - 1.0
    q_growth = mean_q + uplift_q

    steps = years * 4
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=steps, freq="Q")

    fc = []
    hi = []
    lo = []
    cur = last_val

    for i in range(1, steps + 1):
        cur = cur * (1.0 + q_growth)
        fc.append(cur)

        # Simple uncertainty band expanding with horizon
        band = (std_q if std_q > 0 else 0.05) * np.sqrt(i)
        hi.append(cur * (1.0 + band))
        lo.append(cur * (1.0 - band))

    out = pd.DataFrame({"date": future_dates, "forecast": fc, "hi": hi, "lo": lo})
    return out


def money_fmt(x: float) -> str:
    if x is None or np.isnan(x):
        return "—"
    absx = abs(x)
    sign = "-" if x < 0 else ""
    if absx >= 1e12:
        return f"{sign}${absx/1e12:,.2f}T"
    if absx >= 1e9:
        return f"{sign}${absx/1e9:,.2f}B"
    if absx >= 1e6:
        return f"{sign}${absx/1e6:,.2f}M"
    return f"{sign}${absx:,.0f}"


def build_plot(hist: pd.DataFrame, fc: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["revenue"],
            mode="lines",
            name="Historical",
        )
    )

    # Band
    fig.add_trace(
        go.Scatter(
            x=fc["date"],
            y=fc["hi"],
            mode="lines",
            name="80% high (approx)",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["date"],
            y=fc["lo"],
            mode="lines",
            name="80% low (approx)",
            line=dict(dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fc["date"],
            y=fc["forecast"],
            mode="lines",
            name="Forecast (Scenario)",
        )
    )

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
# Sidebar controls
st.sidebar.title("Controls")

try:
    tidy = load_revenue_by_segment()
    data_ok = True
except Exception as e:
    data_ok = False
    st.error(f"Could not load live data from StockAnalysis. Error: {e}")
    st.stop()

products = sorted(tidy["product"].unique().tolist())
default_product = "Advertising" if "Advertising" in products else products[0]

product = st.sidebar.selectbox("Product", products, index=products.index(default_product))
years = st.sidebar.slider("Forecast years", min_value=1, max_value=10, value=3, step=1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario)", min_value=0.00, max_value=0.30, value=0.00, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Notes")
st.sidebar.write("Live data is scraped from StockAnalysis (History table).")
st.sidebar.write("Forecast uses recent quarterly growth with an optional annual uplift.")

# Header with Google logo
col1, col2 = st.columns([1, 12])
with col1:
    st.image(
        "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png",
        use_container_width=True,
    )
with col2:
    st.markdown(
        """
        # Alphabet (Google) Revenue Forecast Dashboard
        Interactive product-level revenue forecasting with a scenario-based CAGR uplift. Use the sliders to stress-test assumptions and export results.
        """
    )

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Product Forecast", "Total Forecast", "Assumptions & Notes", "Download"])

# -----------------------------
# TAB 1: Product Forecast
# -----------------------------
with tab1:
    seg = tidy[tidy["product"] == product].copy()
    seg = seg.sort_values("date")
    fc = forecast_series(seg[["date", "revenue"]], years=years, uplift=uplift)

    end_fc = float(fc["forecast"].iloc[-1])
    base_fc = forecast_series(seg[["date", "revenue"]], years=years, uplift=0.0)["forecast"].iloc[-1]
    delta = float(end_fc - base_fc)

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
    # Build wide frame from tidy
    wide = tidy.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide = wide.dropna(how="all")
    wide["TOTAL"] = wide.sum(axis=1)

    # Forecast each segment, then sum forecasts
    future_steps = years * 4
    last_date = wide.index.max()
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=future_steps, freq="Q")

    total_fc_vals = np.zeros(future_steps, dtype=float)
    total_hi_vals = np.zeros(future_steps, dtype=float)
    total_lo_vals = np.zeros(future_steps, dtype=float)

    for p in wide.columns:
        if p == "TOTAL":
            continue
        hist_p = wide[[p]].reset_index().rename(columns={p: "revenue"})
        fcp = forecast_series(hist_p[["date", "revenue"]], years=years, uplift=uplift)
        total_fc_vals += fcp["forecast"].values
        total_hi_vals += fcp["hi"].values
        total_lo_vals += fcp["lo"].values

    total_hist = wide["TOTAL"].reset_index().rename(columns={"TOTAL": "revenue"})
    total_fc = pd.DataFrame({"date": future_dates, "forecast": total_fc_vals, "hi": total_hi_vals, "lo": total_lo_vals})

    fig_total = build_plot(
        total_hist[["date", "revenue"]],
        total_fc,
        title="Total Alphabet Revenue Forecast (Scenario)",
    )

    st.plotly_chart(fig_total, use_container_width=True)
    st.caption("This total forecast is the sum of per-segment scenario forecasts (not a separately fit model).")

# -----------------------------
# TAB 3: Assumptions
# -----------------------------
with tab3:
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
        - Computes recent **quarter-over-quarter growth** from the last few quarters of the selected series
        - Uses that average growth as the baseline forward growth rate
        - Applies your **extra CAGR uplift** (annual) as an additional quarterly growth component
        - Adds an approximate uncertainty band that widens over time (based on recent growth volatility)
        """
    )

    st.info(
        "If you want, we can upgrade the model to something more advanced (ETS/Prophet/SARIMAX) once you confirm the target behavior and speed you want on Streamlit Cloud."
    )

# -----------------------------
# TAB 4: Download
# -----------------------------
with tab4:
    st.subheader("Download data")
    st.write("Use these downloads to share your results or reuse the dataset.")

    tidy_csv = tidy.copy()
    tidy_csv["revenue"] = tidy_csv["revenue"].astype(float)

    st.download_button(
        label="Download tidy dataset (date, product, revenue) as CSV",
        data=tidy_csv.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_revenue_by_segment_tidy.csv",
        mime="text/csv",
    )

    # Also export a wide table
    wide_out = tidy.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_out.reset_index(inplace=True)

    st.download_button(
        label="Download wide dataset (date + all segments) as CSV",
        data=wide_out.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_revenue_by_segment_wide.csv",
        mime="text/csv",
    )

    st.caption("Source page: StockAnalysis → GOOG → Metrics → Revenue by Segment")
