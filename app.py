# app.py
# Alphabet (Google) Revenue Forecast Dashboard
# Segment revenue scraped from StockAnalysis (History table)
# Income Statement pulled from Yahoo Finance (via yfinance)
#
# requirements.txt (minimum):
# streamlit
# pandas
# numpy
# requests
# beautifulsoup4
# lxml
# plotly
# yfinance

import re
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# -----------------------------
# CONFIG
# -----------------------------
SEGMENT_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

st.set_page_config(
    page_title="Alphabet (Google) Revenue Forecast Dashboard",
    layout="wide",
)

# -----------------------------
# HELPERS
# -----------------------------
def money_to_float(x) -> float:
    """
    Convert strings like '215.49B', '39.00M', '-56.00M', '0', '' to float USD.
    """
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    s = s.replace(",", "").replace("−", "-").replace("\xa0", " ").strip()

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
        return ""
    absx = abs(float(x))
    sign = "-" if x < 0 else ""
    if absx >= 1e12:
        return f"{sign}${absx/1e12:,.2f}T"
    if absx >= 1e9:
        return f"{sign}${absx/1e9:,.2f}B"
    if absx >= 1e6:
        return f"{sign}${absx/1e6:,.2f}M"
    return f"{sign}${absx:,.0f}"


def quarter_end_dates_from_any(dt_series: pd.Series) -> pd.Series:
    """
    Make sure dates line up on QuarterEnd for clean plotting/rolling.
    """
    d = pd.to_datetime(dt_series, errors="coerce")
    d = d.dropna()
    # Force to quarter end
    return (d + pd.offsets.QuarterEnd(0)).astype("datetime64[ns]")


def compute_ttm_from_quarterly(tidy_q: pd.DataFrame) -> pd.DataFrame:
    """
    Input tidy_q columns: date, product, revenue (quarterly)
    Output tidy_ttm: date, product, revenue (rolling 4Q sum)
    """
    tidy_q = tidy_q.copy()
    tidy_q["date"] = quarter_end_dates_from_any(tidy_q["date"])
    tidy_q = tidy_q.dropna(subset=["date"])
    tidy_q = tidy_q.sort_values(["product", "date"])

    out = []
    for p, g in tidy_q.groupby("product", sort=False):
        gg = g.sort_values("date").copy()
        gg["revenue"] = pd.to_numeric(gg["revenue"], errors="coerce")
        gg["ttm"] = gg["revenue"].rolling(4, min_periods=4).sum()
        gg = gg.dropna(subset=["ttm"])
        out.append(gg[["date", "product", "ttm"]].rename(columns={"ttm": "revenue"}))

    if not out:
        return pd.DataFrame(columns=["date", "product", "revenue"])

    tidy_ttm = pd.concat(out, ignore_index=True)
    tidy_ttm = tidy_ttm.sort_values(["product", "date"]).reset_index(drop=True)
    return tidy_ttm


def estimate_growth(series: pd.Series, lookback_quarters: int = 8) -> tuple[float, float]:
    """
    Estimate quarterly growth mean and std from last N points of history.
    Returns (mean_q_growth, std_q_growth) in decimal, e.g. 0.03 = 3% per quarter.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) < 6:
        return 0.02, 0.05

    s = s.iloc[-lookback_quarters:] if len(s) > lookback_quarters else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 3:
        return 0.02, 0.05

    return float(g.mean()), float(g.std(ddof=1) if len(g) > 1 else 0.05)


def forecast_series(hist_df: pd.DataFrame, years: int, uplift_annual: float, lookback_quarters: int = 8) -> pd.DataFrame:
    """
    Given history df with columns [date, revenue], forecast next years*4 quarters.
    uplift_annual is extra CAGR as annual decimal (0.10 = 10%).
    """
    df = hist_df.sort_values("date").copy()
    df = df.dropna(subset=["revenue"])
    if df.empty:
        return pd.DataFrame(columns=["date", "forecast", "hi", "lo"])

    last_date = pd.to_datetime(df["date"].max())
    last_val = float(df.loc[df["date"] == last_date, "revenue"].iloc[-1])

    mean_q, std_q = estimate_growth(df["revenue"], lookback_quarters=lookback_quarters)

    uplift_q = (1.0 + float(uplift_annual)) ** (1.0 / 4.0) - 1.0
    q_growth = mean_q + uplift_q

    steps = int(years) * 4
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


def build_plot(hist: pd.DataFrame, fc: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["revenue"],
            mode="lines",
            name="Historical",
        )
    )

    if not fc.empty:
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
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="v"),
    )
    return fig


# -----------------------------
# SCRAPING: StockAnalysis segment history (Quarterly)
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_segment_quarterly_from_stockanalysis() -> pd.DataFrame:
    """
    Scrape the StockAnalysis 'History' table from revenue-by-segment page.
    Returns tidy quarterly: date, product, revenue.
    """
    r = requests.get(SEGMENT_URL, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # Find History section header, then first table after it
    history_hdr = None
    for h in soup.find_all(["h2", "h3"]):
        if h.get_text(strip=True).lower() == "history":
            history_hdr = h
            break

    table = history_hdr.find_next("table") if history_hdr else None

    # Fallback to largest table
    if table is None:
        tables = soup.find_all("table")
        if not tables:
            raise ValueError("No tables found on the page.")
        table = max(tables, key=lambda t: len(t.find_all("tr")))

    # Headers
    thead = table.find("thead")
    if thead:
        headers = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in thead.find_all(["th", "td"])]
    else:
        first_row = table.find("tr")
        headers = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in first_row.find_all(["th", "td"])]

    headers = [h.strip() for h in headers]
    if headers and headers[0].lower() != "date":
        headers[0] = "Date"

    rows = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        vals = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in cells]
        if vals and vals[0].strip().lower() == "date":
            continue

        if len(vals) < len(headers):
            vals += [""] * (len(headers) - len(vals))
        elif len(vals) > len(headers):
            vals = vals[: len(headers)]

        rows.append(vals)

    wide = pd.DataFrame(rows, columns=headers)
    wide.rename(columns={"Date": "date"}, inplace=True)
    wide["date"] = pd.to_datetime(wide["date"], errors="coerce")
    wide = wide.dropna(subset=["date"]).copy()

    # parse numeric
    for c in wide.columns:
        if c != "date":
            wide[c] = wide[c].apply(money_to_float)

    wide = wide.sort_values("date").reset_index(drop=True)

    tidy = (
        wide.melt("date", var_name="product", value_name="revenue")
        .dropna(subset=["revenue"])
        .sort_values(["product", "date"])
        .reset_index(drop=True)
    )

    # ensure QuarterEnd alignment
    tidy["date"] = quarter_end_dates_from_any(tidy["date"])

    return tidy


# -----------------------------
# INCOME STATEMENT: Yahoo Finance via yfinance
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_income_statement_from_yahoo(ticker: str = "GOOGL") -> pd.DataFrame:
    """
    Pull income statement from Yahoo Finance via yfinance.
    Returns tidy dataframe: date, metric, value
    """
    t = yf.Ticker(ticker)

    stmt = None

    # Try multiple APIs depending on yfinance version
    try:
        stmt = t.quarterly_income_stmt
        if not isinstance(stmt, pd.DataFrame) or stmt.empty:
            stmt = None
    except Exception:
        stmt = None

    if stmt is None:
        try:
            stmt = t.quarterly_financials
            if not isinstance(stmt, pd.DataFrame) or stmt.empty:
                stmt = None
        except Exception:
            stmt = None

    if stmt is None:
        try:
            stmt = t.get_income_stmt(freq="quarterly")
            if not isinstance(stmt, pd.DataFrame) or stmt.empty:
                stmt = None
        except Exception:
            stmt = None

    if stmt is None:
        raise ValueError("Could not load income statement from Yahoo Finance.")

    stmt = stmt.copy()
    stmt.columns = pd.to_datetime(stmt.columns, errors="coerce")
    stmt = stmt.loc[:, stmt.columns.notna()]
    stmt = stmt.sort_index(axis=1)

    tidy = (
        stmt.reset_index()
        .melt(id_vars=["index"], var_name="date", value_name="value")
        .rename(columns={"index": "metric"})
    )
    tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce")
    tidy["date"] = (tidy["date"] + pd.offsets.QuarterEnd(0)).astype("datetime64[ns]")
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")

    tidy = tidy.dropna(subset=["date", "value"])
    tidy = tidy.sort_values(["metric", "date"]).reset_index(drop=True)

    return tidy


# -----------------------------
# UI
# -----------------------------
st.sidebar.title("Controls")

force_refresh = st.sidebar.button("Force Refresh (ignore cache)")
if force_refresh:
    st.cache_data.clear()

years = st.sidebar.slider("Forecast years", min_value=1, max_value=10, value=3, step=1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario, annual)", min_value=0.00, max_value=0.30, value=0.00, step=0.01)

view_mode = st.sidebar.radio(
    "View mode",
    ["Quarterly", "TTM (matches StockAnalysis chart)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Notes")
st.sidebar.write("Segment data is scraped from StockAnalysis (History table).")
st.sidebar.write("TTM is computed as rolling 4-quarter sum (this matches their TTM chart).")
st.sidebar.write("Income statement uses Yahoo Finance data via yfinance.")


# Header
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
        Interactive segment-level quarterly + TTM analysis and scenario forecasting.  
        Data source: StockAnalysis (segment revenue) + Yahoo Finance (income statement).
        """
    )

st.markdown("---")

# Load segment data
try:
    seg_q = load_segment_quarterly_from_stockanalysis()
except Exception as e:
    st.error(f"Segment scrape failed: {e}")
    st.stop()

# Build TTM version
seg_ttm = compute_ttm_from_quarterly(seg_q)

# Pick active dataset
if view_mode.startswith("TTM"):
    seg_active = seg_ttm.copy()
    y_axis_name = "Revenue (USD, TTM)"
else:
    seg_active = seg_q.copy()
    y_axis_name = "Revenue (USD, Quarterly)"

# Products dropdown should match active data
products = sorted(seg_active["product"].dropna().unique().tolist())
default_product = "Advertising" if "Advertising" in products else (products[0] if products else None)

product = st.sidebar.selectbox(
    "Product",
    products,
    index=(products.index(default_product) if default_product in products else 0),
)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Segment Forecast", "Total Forecast", "Segment Table Check", "Income Statement", "Download"]
)

# -----------------------------
# TAB 1: Segment Forecast
# -----------------------------
with tab1:
    seg = seg_active[seg_active["product"] == product].sort_values("date").copy()

    if seg.empty:
        st.warning("No data available for this segment.")
    else:
        fc = forecast_series(seg[["date", "revenue"]], years=years, uplift_annual=uplift)

        end_fc = float(fc["forecast"].iloc[-1]) if not fc.empty else np.nan
        base_fc_df = forecast_series(seg[["date", "revenue"]], years=years, uplift_annual=0.0)
        base_fc = float(base_fc_df["forecast"].iloc[-1]) if not base_fc_df.empty else np.nan
        delta = (end_fc - base_fc) if (not np.isnan(end_fc) and not np.isnan(base_fc)) else np.nan

        k1, k2, k3 = st.columns(3)
        k1.metric("End Forecast (Scenario)", money_fmt(end_fc))
        k2.metric("Δ vs Baseline", money_fmt(delta))
        k3.metric("Last Reported Quarter", seg["date"].max().strftime("%b %d, %Y"))

        fig = build_plot(
            seg[["date", "revenue"]],
            fc,
            title=f"{product}: Historical + {years}-Year Forecast (uplift {uplift*100:.1f}%)",
            y_title=y_axis_name,
        )
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# TAB 2: Total Forecast
# -----------------------------
with tab2:
    # wide from ACTIVE (Quarterly or TTM)
    wide = seg_active.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide = wide.dropna(how="all")

    # total historical
    wide["TOTAL"] = wide.sum(axis=1)
    total_hist = wide["TOTAL"].reset_index().rename(columns={"TOTAL": "revenue"})

    # forecast total as sum of per-segment forecasts (like your original)
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
        if hist_p["revenue"].dropna().empty:
            continue
        fcp = forecast_series(hist_p[["date", "revenue"]], years=years, uplift_annual=uplift)
        if fcp.empty:
            continue
        total_fc_vals += fcp["forecast"].values
        total_hi_vals += fcp["hi"].values
        total_lo_vals += fcp["lo"].values

    total_fc = pd.DataFrame({"date": future_dates, "forecast": total_fc_vals, "hi": total_hi_vals, "lo": total_lo_vals})

    title_mode = "TTM" if view_mode.startswith("TTM") else "Quarterly"
    fig_total = build_plot(
        total_hist[["date", "revenue"]],
        total_fc,
        title=f"Total Alphabet Revenue Forecast ({title_mode}, Scenario)",
        y_title=("Revenue (USD, TTM)" if view_mode.startswith("TTM") else "Revenue (USD, Quarterly)"),
    )

    st.plotly_chart(fig_total, use_container_width=True)
    st.caption("Total forecast is the sum of per-segment scenario forecasts (not a separately fit model).")


# -----------------------------
# TAB 3: Segment Table Check
# -----------------------------
with tab3:
    st.subheader("StockAnalysis History Table (Quarterly) Check")
    st.write("This shows the latest row from the scraped History table so you can verify it matches the site.")

    # Build a wide quarterly table to display last row like your screenshot
    wide_q = seg_q.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_q = wide_q.dropna(how="all")

    if wide_q.empty:
        st.warning("No quarterly rows found.")
    else:
        last_dt = wide_q.index.max()
        last_row = wide_q.loc[[last_dt]].copy()
        last_row.insert(0, "Date", [last_dt.strftime("%b %d, %Y")])

        # Format numbers like B/M for display
        disp = last_row.copy()
        for c in disp.columns:
            if c != "Date":
                disp[c] = disp[c].apply(money_fmt)

        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.caption("If this row matches the StockAnalysis 'History' table, your scrape is correct.")


# -----------------------------
# TAB 4: Income Statement (Yahoo Finance)
# -----------------------------
with tab4:
    st.subheader("Income Statement (Yahoo Finance: GOOGL)")

    try:
        inc = load_income_statement_from_yahoo("GOOGL")
    except Exception as e:
        st.error(f"Income statement load failed: {e}")
        st.stop()

    metrics = sorted(inc["metric"].unique().tolist())
    preferred_defaults = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EBITDA"]
    default_metric = next((m for m in preferred_defaults if m in metrics), metrics[0])

    metric = st.selectbox("Metric", metrics, index=metrics.index(default_metric))

    inc_one = inc[inc["metric"] == metric].sort_values("date").copy()

    fig_inc = go.Figure()
    fig_inc.add_trace(go.Bar(x=inc_one["date"], y=inc_one["value"], name=metric))
    fig_inc.update_layout(
        title=f"GOOGL Income Statement: {metric}",
        height=520,
        xaxis_title="Quarter",
        yaxis_title=f"{metric} (USD)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_inc, use_container_width=True)

    show = inc_one.copy()
    show["value"] = show["value"].apply(money_fmt)
    show = show.rename(columns={"date": "Period Ending", "value": "Value"})
    st.dataframe(show[["Period Ending", "Value"]], use_container_width=True, hide_index=True)

    st.caption("Source: Yahoo Finance financials for GOOGL (pulled via yfinance).")


# -----------------------------
# TAB 5: Download
# -----------------------------
with tab5:
    st.subheader("Download data")
    st.write("Download segment revenue (quarterly + TTM) and Yahoo income statement.")

    # Segment tidy quarterly
    seg_q_out = seg_q.copy()
    seg_q_out["revenue"] = seg_q_out["revenue"].astype(float)

    st.download_button(
        label="Download segment revenue (Quarterly, tidy) CSV",
        data=seg_q_out.to_csv(index=False).encode("utf-8"),
        file_name="goog_segment_revenue_quarterly_tidy.csv",
        mime="text/csv",
    )

    # Segment tidy TTM
    seg_ttm_out = seg_ttm.copy()
    if not seg_ttm_out.empty:
        seg_ttm_out["revenue"] = seg_ttm_out["revenue"].astype(float)

    st.download_button(
        label="Download segment revenue (TTM rolling 4Q, tidy) CSV",
        data=seg_ttm_out.to_csv(index=False).encode("utf-8"),
        file_name="goog_segment_revenue_ttm_tidy.csv",
        mime="text/csv",
    )

    # Wide quarterly
    wide_q_out = seg_q.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_q_out = wide_q_out.reset_index()

    st.download_button(
        label="Download segment revenue (Quarterly, wide) CSV",
        data=wide_q_out.to_csv(index=False).encode("utf-8"),
        file_name="goog_segment_revenue_quarterly_wide.csv",
        mime="text/csv",
    )

    # Wide TTM
    wide_ttm_out = seg_ttm.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_ttm_out = wide_ttm_out.reset_index()

    st.download_button(
        label="Download segment revenue (TTM, wide) CSV",
        data=wide_ttm_out.to_csv(index=False).encode("utf-8"),
        file_name="goog_segment_revenue_ttm_wide.csv",
        mime="text/csv",
    )

    # Income statement
    try:
        inc_out = load_income_statement_from_yahoo("GOOGL").copy()
        inc_out["value"] = inc_out["value"].astype(float)
        st.download_button(
            label="Download income statement (Yahoo, quarterly tidy) CSV",
            data=inc_out.to_csv(index=False).encode("utf-8"),
            file_name="googl_income_statement_yahoo_quarterly_tidy.csv",
            mime="text/csv",
        )
    except Exception:
        st.caption("Income statement download unavailable (Yahoo/yfinance fetch failed).")

    st.caption("Segment source: StockAnalysis → GOOG → Metrics → Revenue by Segment (History table).")
