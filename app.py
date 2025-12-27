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
SEGMENT_BASE_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
UA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://stockanalysis.com/",
}

YF_TICKER = "GOOGL"  # Yahoo Finance ticker for income statement

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

    s = s.replace(",", "")
    s = s.replace("−", "-").replace("\xa0", " ")

    # try direct parse
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


def _pick_segment_table(html: str) -> pd.DataFrame:
    """
    Use read_html to find the segment History table.
    We select the table that contains at least these segment columns.
    """
    dfs = pd.read_html(html)
    # columns seen on StockAnalysis (may vary slightly)
    required_any = [
        "Google Search & Other",
        "Google Cloud",
        "YouTube Ads",
        "Google Network",
        "Other Bets",
    ]

    best = None
    best_score = -1

    for df in dfs:
        cols = [str(c).strip() for c in df.columns]
        score = sum(1 for c in required_any if c in cols)
        # must have Date and some segment columns
        if ("Date" in cols or "date" in [c.lower() for c in cols]) and score > best_score:
            best = df
            best_score = score

    if best is None or best_score < 3:
        raise ValueError("Could not locate the segment History table (layout changed or blocked).")

    # normalize Date column name
    cols = list(best.columns)
    for i, c in enumerate(cols):
        if str(c).strip().lower() == "date":
            cols[i] = "Date"
    best.columns = cols

    return best


def _looks_like_ttm(wide_df: pd.DataFrame) -> bool:
    """
    Detect if the table is TTM vs Quarterly by checking the magnitude of
    Google Search & Other in the most recent row.
    Quarterly is usually ~40-70B; TTM is usually ~150-250B.
    """
    if "Google Search & Other" not in wide_df.columns:
        return False

    x = wide_df["Google Search & Other"].dropna()
    if len(x) == 0:
        return False

    # take last numeric value
    v = x.iloc[0] if isinstance(x.iloc[0], (int, float, np.number)) else x.iloc[-1]
    try:
        v = float(v)
    except Exception:
        return False

    return v > 100e9  # > $100B strongly suggests TTM


def _fetch_stockanalysis_html(try_urls: list[str]) -> str:
    last_err = None
    for u in try_urls:
        try:
            r = requests.get(u, headers=UA_HEADERS, timeout=30)
            r.raise_for_status()
            text = r.text
            # crude block detection
            if "captcha" in text.lower() or "cloudflare" in text.lower():
                last_err = ValueError("Blocked by anti-bot / captcha.")
                continue
            return text
        except Exception as e:
            last_err = e
    raise last_err if last_err else ValueError("Failed to fetch StockAnalysis HTML.")


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_segment_data(mode: str) -> pd.DataFrame:
    """
    Returns tidy df: date, product, value
    mode: "Quarterly" or "TTM"
    We try common query params used by toggles to force the correct table.
    """
    mode = mode.strip().lower()

    # try a few patterns; StockAnalysis often uses ?p=ttm or ?p=quarterly (or similar)
    if mode == "quarterly":
        candidates = [
            SEGMENT_BASE_URL + "?p=quarterly",
            SEGMENT_BASE_URL + "?period=quarterly",
            SEGMENT_BASE_URL + "?view=quarterly",
            SEGMENT_BASE_URL + "?type=quarterly",
            SEGMENT_BASE_URL,  # fallback
        ]
    else:
        candidates = [
            SEGMENT_BASE_URL + "?p=ttm",
            SEGMENT_BASE_URL + "?period=ttm",
            SEGMENT_BASE_URL + "?view=ttm",
            SEGMENT_BASE_URL + "?type=ttm",
            SEGMENT_BASE_URL,  # fallback
        ]

    html = _fetch_stockanalysis_html(candidates)
    wide = _pick_segment_table(html)

    # convert values
    wide = wide.copy()
    wide["Date"] = pd.to_datetime(wide["Date"], errors="coerce")

    for c in wide.columns:
        if c != "Date":
            wide[c] = wide[c].apply(money_to_float)

    wide = wide.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # If user asked Quarterly but we accidentally received TTM, keep trying with alternate order
    if mode == "quarterly" and _looks_like_ttm(wide):
        # Try again, stronger ordering (sometimes base url defaults to TTM)
        candidates2 = [
            SEGMENT_BASE_URL + "?p=quarterly",
            SEGMENT_BASE_URL + "?view=quarterly",
            SEGMENT_BASE_URL + "?period=quarterly",
            SEGMENT_BASE_URL + "?type=quarterly",
        ]
        html2 = _fetch_stockanalysis_html(candidates2)
        wide2 = _pick_segment_table(html2)
        wide2["Date"] = pd.to_datetime(wide2["Date"], errors="coerce")
        for c in wide2.columns:
            if c != "Date":
                wide2[c] = wide2[c].apply(money_to_float)
        wide2 = wide2.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        # accept wide2 if it looks quarterly
        if not _looks_like_ttm(wide2):
            wide = wide2

    tidy = wide.melt("Date", var_name="product", value_name="value").dropna()
    tidy = tidy.rename(columns={"Date": "date"})
    tidy = tidy.sort_values(["product", "date"]).reset_index(drop=True)

    # remove obvious non-segment columns if any (sometimes Growth/Latest appear in tables)
    bad = {"growth", "latest", "change"}
    tidy = tidy[~tidy["product"].str.lower().isin(bad)].copy()

    return tidy


def compute_ttm_from_quarterly(quarterly_tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 4-quarter sum by product. This is the TTM that matches StockAnalysis charts.
    """
    df = quarterly_tidy.copy()
    df = df.sort_values(["product", "date"])
    df["ttm"] = df.groupby("product")["value"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    out = df.dropna(subset=["ttm"]).rename(columns={"ttm": "value"})
    out = out[["date", "product", "value"]].reset_index(drop=True)
    return out


def estimate_qoq_growth(series: pd.Series, lookback_quarters: int = 8) -> tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) < 6:
        return 0.02, 0.05
    s = s.iloc[-lookback_quarters:] if len(s) > lookback_quarters else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 3:
        return 0.02, 0.05
    return float(g.mean()), float(g.std(ddof=1) if len(g) > 1 else 0.05)


def forecast_series(hist_df: pd.DataFrame, years: int, uplift_annual: float, lookback_quarters: int = 8) -> pd.DataFrame:
    """
    Forecast next years*4 quarters from quarterly history using average QoQ growth.
    uplift_annual adds extra annual CAGR converted to quarterly.
    """
    df = hist_df.sort_values("date").dropna(subset=["value"]).copy()
    last_date = df["date"].max()
    last_val = float(df.loc[df["date"] == last_date, "value"].iloc[-1])

    mean_q, std_q = estimate_qoq_growth(df["value"], lookback_quarters=lookback_quarters)

    uplift_q = (1.0 + uplift_annual) ** (1.0 / 4.0) - 1.0
    q_growth = mean_q + uplift_q

    steps = years * 4
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


def build_line_plot(hist: pd.DataFrame, fc: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hist["date"], y=hist["value"], mode="lines", name="Historical"))

    fig.add_trace(go.Scatter(x=fc["date"], y=fc["hi"], mode="lines", name="80% high (approx)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lo"], mode="lines", name="80% low (approx)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], mode="lines", name="Forecast (Scenario)"))

    fig.update_layout(
        title=title,
        height=520,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Quarter",
        yaxis_title=y_title,
    )
    return fig


def build_bar_plot_dates_only(df: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    """
    Bar chart that ONLY shows real quarter end dates (no month expansion).
    """
    d = df.sort_values("date").copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["date"], y=d["value"], name=y_title))

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Period Ending",
        yaxis_title=y_title,
        hovermode="x unified",
    )

    # force ticks to be exactly the dates we have
    fig.update_xaxes(
        tickmode="array",
        tickvals=d["date"].tolist(),
        ticktext=[x.strftime("%Y-%m-%d") for x in d["date"]],
    )

    return fig


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_yahoo_income_statement_quarterly(ticker: str = YF_TICKER) -> pd.DataFrame:
    """
    Pull Yahoo Finance income statement via yfinance (quarterly).
    Returns long df with: date, metric, value
    """
    t = yf.Ticker(ticker)
    q = t.quarterly_financials  # rows=metrics, cols=period end
    if q is None or q.empty:
        raise ValueError("Yahoo Finance quarterly financials not available via yfinance right now.")

    df = q.copy()
    df.columns = pd.to_datetime(df.columns, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.T  # rows = dates
    df.index.name = "date"
    df = df.reset_index()

    # melt to long
    long = df.melt(id_vars=["date"], var_name="metric", value_name="value").dropna()
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["date", "value"])

    # yfinance returns in USD (already absolute dollars)
    return long.sort_values(["metric", "date"]).reset_index(drop=True)


# -----------------------------
# UI (Sidebar)
# -----------------------------
st.sidebar.title("Controls")

force_refresh = st.sidebar.button("Force Refresh (ignore cache)")

forecast_years = st.sidebar.slider("Forecast years", 1, 10, 5, 1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario, annual)", 0.00, 0.30, 0.00, 0.01)

st.sidebar.markdown("---")
view_mode = st.sidebar.radio("View mode", ["Quarterly", "TTM (matches StockAnalysis chart)"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Notes")
st.sidebar.write("Segment data: StockAnalysis (History table).")
st.sidebar.write("TTM is computed as rolling 4-quarter sum (matches their chart).")
st.sidebar.write("Income statement: Yahoo Finance via yfinance (quarterly).")

if force_refresh:
    load_segment_data.clear()
    load_yahoo_income_statement_quarterly.clear()
    st.cache_data.clear()

# -----------------------------
# HEADER
# -----------------------------
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
        Interactive segment-level **quarterly + TTM** analysis and scenario forecasting.  
        Data source: StockAnalysis (segment revenue) + Yahoo Finance (income statement).
        """
    )

st.markdown("---")

tabs = st.tabs(["Segment Forecast", "Total Forecast", "Segment Table Check", "Income Statement", "Download"])

# -----------------------------
# LOAD SEGMENT DATA
# -----------------------------
try:
    seg_quarterly = load_segment_data("Quarterly")  # always load quarterly
except Exception as e:
    st.error(f"Segment scrape failed: {e}")
    st.stop()

# Build TTM from quarterly (this is what should match their TTM chart)
seg_ttm = compute_ttm_from_quarterly(seg_quarterly)

# choose active series based on view mode
if view_mode.lower().startswith("ttm"):
    seg_active = seg_ttm
    y_title_seg = "Revenue (USD, TTM)"
else:
    seg_active = seg_quarterly
    y_title_seg = "Revenue (USD, Quarterly)"

products = sorted(seg_active["product"].unique().tolist())
default_product = "Advertising" if "Advertising" in products else products[0]
product = st.sidebar.selectbox("Product", products, index=products.index(default_product))

# -----------------------------
# TAB 1: Segment Forecast
# -----------------------------
with tabs[0]:
    dfp = seg_active[seg_active["product"] == product].sort_values("date").copy()

    # Forecast always works in "units of the active series"
    fc = forecast_series(dfp.rename(columns={"value": "value"}), years=forecast_years, uplift_annual=uplift)

    end_fc = float(fc["forecast"].iloc[-1])
    base_fc = forecast_series(dfp, years=forecast_years, uplift_annual=0.0)["forecast"].iloc[-1]
    delta = float(end_fc - base_fc)

    k1, k2, k3 = st.columns(3)
    k1.metric("End Forecast (Scenario)", money_fmt(end_fc))
    k2.metric("Δ vs Baseline", money_fmt(delta))
    k3.metric("Last Reported Period", dfp["date"].max().strftime("%Y-%m-%d"))

    fig = build_line_plot(
        hist=dfp[["date", "value"]],
        fc=fc,
        title=f"{product}: Historical + {forecast_years}-Year Forecast (uplift {uplift*100:.1f}%)",
        y_title=y_title_seg,
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 2: Total Forecast
# -----------------------------
with tabs[1]:
    # Total from segment sums (quarterly), then optionally convert to TTM for display
    wide_q = seg_quarterly.pivot_table(index="date", columns="product", values="value", aggfunc="sum").sort_index()
    wide_q["TOTAL"] = wide_q.sum(axis=1)

    total_q = wide_q["TOTAL"].reset_index().rename(columns={"TOTAL": "value"}).sort_values("date")

    if view_mode.lower().startswith("ttm"):
        # rolling 4-quarter sum for total
        total_active = total_q.copy()
        total_active["value"] = total_active["value"].rolling(4, min_periods=4).sum()
        total_active = total_active.dropna(subset=["value"])
        y_title_total = "Revenue (USD, TTM)"
        title_total = "Total Alphabet Revenue Forecast (TTM, Scenario)"
    else:
        total_active = total_q
        y_title_total = "Revenue (USD, Quarterly)"
        title_total = "Total Alphabet Revenue Forecast (Quarterly, Scenario)"

    fc_total = forecast_series(total_active, years=forecast_years, uplift_annual=uplift)

    fig_total = build_line_plot(
        hist=total_active[["date", "value"]],
        fc=fc_total,
        title=title_total,
        y_title=y_title_total,
    )
    st.plotly_chart(fig_total, use_container_width=True)

    st.caption("Total is computed as the **sum of segment series** (not a separately fit model).")

# -----------------------------
# TAB 3: Segment Table Check (debug)
# -----------------------------
with tabs[2]:
    st.subheader("Segment History Table Check")
    st.write("This helps you confirm the scrape matches the StockAnalysis History table.")
    sample = seg_quarterly.sort_values(["date", "product"]).copy()
    sample["value_fmt"] = sample["value"].apply(money_fmt)
    st.dataframe(sample.tail(50), use_container_width=True)

# -----------------------------
# TAB 4: Income Statement (Yahoo Finance)
# -----------------------------
with tabs[3]:
    st.subheader(f"Income Statement (Yahoo Finance: {YF_TICKER})")
    st.caption("Pulled via `yfinance` quarterly_financials. Chart shows ONLY the real quarter end dates.")

    try:
        inc = load_yahoo_income_statement_quarterly(YF_TICKER)
    except Exception as e:
        st.error(f"Could not load Yahoo Finance income statement: {e}")
        st.stop()

    metrics = sorted(inc["metric"].unique().tolist())
    default_metric = "Total Revenue" if "Total Revenue" in metrics else metrics[0]
    metric = st.selectbox("Metric", metrics, index=metrics.index(default_metric))

    inc_m = inc[inc["metric"] == metric].sort_values("date").copy()

    # yfinance values are absolute dollars; keep as-is
    inc_m_plot = inc_m.rename(columns={"value": "value"})
    fig_inc = build_bar_plot_dates_only(
        inc_m_plot[["date", "value"]],
        title=f"{YF_TICKER} Income Statement: {metric}",
        y_title=metric,
    )
    st.plotly_chart(fig_inc, use_container_width=True)

    # Table with ONLY quarter end dates (no extra months)
    show_tbl = inc_m_plot[["date", "value"]].copy()
    show_tbl["Period Ending"] = show_tbl["date"].dt.strftime("%Y-%m-%d")
    show_tbl["Value"] = show_tbl["value"].apply(money_fmt)
    show_tbl = show_tbl[["Period Ending", "Value"]]
    st.dataframe(show_tbl, use_container_width=True)

    st.caption("Source: Yahoo Finance financials for GOOGL (pulled via yfinance).")

# -----------------------------
# TAB 5: Download
# -----------------------------
with tabs[4]:
    st.subheader("Download data")

    # segment downloads
    q_csv = seg_quarterly.rename(columns={"value": "revenue"}).copy()
    ttm_csv = seg_ttm.rename(columns={"value": "revenue_ttm"}).copy()

    st.download_button(
        "Download segment quarterly (tidy) CSV",
        data=q_csv.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_segments_quarterly_tidy.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download segment TTM (tidy) CSV",
        data=ttm_csv.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_segments_ttm_tidy.csv",
        mime="text/csv",
    )

    # income statement download
    try:
        inc = load_yahoo_income_statement_quarterly(YF_TICKER)
        st.download_button(
            "Download Yahoo income statement (tidy) CSV",
            data=inc.to_csv(index=False).encode("utf-8"),
            file_name="googl_income_statement_quarterly_tidy.csv",
            mime="text/csv",
        )
    except Exception:
        pass

    st.caption("Segment source: StockAnalysis → GOOG → Metrics → Revenue by Segment. Income source: Yahoo Finance (yfinance).")
