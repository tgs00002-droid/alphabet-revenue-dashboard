# app.py
# Alphabet (Google) Revenue Forecast Dashboard (Streamlit)
# - Scrapes StockAnalysis "Revenue by Segment" History table (Quarterly)
# - Builds product forecasts + total forecast
# - Fixes the common "double-counting" issue by NOT summing the standalone "Advertising" column
#   if Search/YouTube/Network are present (those are the advertising components).
#
# Run:
#   streamlit run app.py

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
UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

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
    Returns value in dollars (not billions).
    """
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    s = s.replace(",", "")
    s = s.replace("−", "-").replace("\xa0", " ")

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


def _clean_colname(c: str) -> str:
    c = (c or "").replace("\xa0", " ").strip()
    c = re.sub(r"\s+", " ", c)
    return c


def _find_history_table(soup: BeautifulSoup):
    """
    Find the 'History' table on the Revenue by Segment page.
    Strategy:
      1) Find a header named 'History' and take the next table
      2) Else find a table with a 'Date' header and at least 5 columns
      3) Else fallback to the largest table
    """
    # 1) Header -> next table
    for h in soup.find_all(["h2", "h3"]):
        if h.get_text(" ", strip=True).lower() == "history":
            t = h.find_next("table")
            if t is not None:
                return t

    tables = soup.find_all("table")
    if not tables:
        return None

    # 2) Table with "Date" in header and looks wide enough
    best = None
    for t in tables:
        thead = t.find("thead")
        if not thead:
            continue
        header_cells = thead.find_all(["th", "td"])
        headers = [_clean_colname(c.get_text(" ", strip=True)) for c in header_cells]
        if headers and headers[0].lower() == "date" and len(headers) >= 5:
            best = t
            break
    if best is not None:
        return best

    # 3) Fallback largest by rows
    return max(tables, key=lambda tt: len(tt.find_all("tr")))


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_segment_quarterly(force_refresh: bool = False) -> pd.DataFrame:
    """
    Scrape the quarterly segment history table and return:
      tidy df: columns = [date, product, revenue]
    revenue is in dollars (float).

    IMPORTANT:
    StockAnalysis sometimes includes both:
      - Advertising (aggregate)
      - AND components (Search & Other, YouTube Ads, Google Network)
    Those should NOT all be summed together for a total (double-counting).
    We'll handle totals later by dropping 'Advertising' when components exist.
    """
    # force_refresh is used only to invalidate cache key
    _ = force_refresh

    r = requests.get(SEGMENT_URL, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    table = _find_history_table(soup)
    if table is None:
        raise ValueError("Could not find a History table on the segment page.")

    # Headers
    thead = table.find("thead")
    if thead:
        header_cells = thead.find_all(["th", "td"])
        headers = [_clean_colname(c.get_text(" ", strip=True)) for c in header_cells]
    else:
        first_row = table.find("tr")
        headers = [_clean_colname(c.get_text(" ", strip=True)) for c in first_row.find_all(["th", "td"])]

    if not headers or headers[0].lower() != "date":
        # try to force first col name
        if headers:
            headers[0] = "Date"
        else:
            raise ValueError("No headers detected in the segment History table.")

    # Rows
    rows = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        vals = [_clean_colname(c.get_text(" ", strip=True)) for c in cells]

        if vals and vals[0].lower() == "date":
            continue

        if len(vals) < len(headers):
            vals += [""] * (len(headers) - len(vals))
        elif len(vals) > len(headers):
            vals = vals[: len(headers)]
        rows.append(vals)

    wide = pd.DataFrame(rows, columns=headers)
    wide = wide.rename(columns={"Date": "date"})
    wide["date"] = pd.to_datetime(wide["date"], errors="coerce")

    # Parse numeric
    for c in wide.columns:
        if c != "date":
            wide[c] = wide[c].apply(money_to_float)

    wide = wide.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    tidy = wide.melt("date", var_name="product", value_name="revenue").dropna()
    tidy["product"] = tidy["product"].apply(_clean_colname)
    tidy = tidy.sort_values(["product", "date"]).reset_index(drop=True)

    # Quick sanity: if table got blocked, it often yields tiny/empty data
    if tidy.empty or tidy["date"].nunique() < 2:
        raise ValueError("Segment scrape returned empty/too-small data (possible blocking or layout change).")

    return tidy


def estimate_qoq(series: pd.Series, lookback_quarters: int = 8) -> tuple[float, float]:
    """
    Estimate mean/stdev of QoQ growth from last N points.
    """
    s = series.dropna().astype(float)
    if len(s) < 6:
        return 0.02, 0.05

    s = s.iloc[-lookback_quarters:] if len(s) > lookback_quarters else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 3:
        return 0.02, 0.05

    std = float(g.std(ddof=1)) if len(g) > 1 else 0.05
    return float(g.mean()), std


def forecast_series(hist_df: pd.DataFrame, years: int, uplift_annual: float, lookback_quarters: int = 8) -> pd.DataFrame:
    """
    Forecast next years*4 quarters using baseline QoQ mean + annual uplift converted to quarterly.
    Returns df with [date, forecast, hi, lo]
    """
    df = hist_df.sort_values("date").copy()
    df = df.dropna(subset=["revenue"])
    last_date = df["date"].max()
    last_val = float(df.loc[df["date"] == last_date, "revenue"].iloc[-1])

    mean_q, std_q = estimate_qoq(df["revenue"], lookback_quarters=lookback_quarters)
    uplift_q = (1.0 + uplift_annual) ** (1.0 / 4.0) - 1.0
    q_growth = mean_q + uplift_q

    steps = years * 4
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=steps, freq="Q")

    fc, hi, lo = [], [], []
    cur = last_val
    vol = std_q if std_q > 0 else 0.05

    for i in range(1, steps + 1):
        cur *= (1.0 + q_growth)
        fc.append(cur)

        band = vol * np.sqrt(i)
        hi.append(cur * (1.0 + band))
        lo.append(cur * (1.0 - band))

    return pd.DataFrame({"date": future_dates, "forecast": fc, "hi": hi, "lo": lo})


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


def build_plot(hist: pd.DataFrame, fc: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["revenue"], mode="lines", name="Historical"
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["hi"], mode="lines", name="80% high (approx)", line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["lo"], mode="lines", name="80% low (approx)", line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["forecast"], mode="lines", name="Forecast (Scenario)"
    ))

    fig.update_layout(
        title=title,
        height=520,
        xaxis_title="Quarter",
        yaxis_title="Revenue (USD)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def canonical_total_columns(cols: list[str]) -> list[str]:
    """
    Decide which columns to sum for TOTAL.

    If Search/YouTube/Network are present, we drop "Advertising" to avoid double-counting.
    Otherwise, if only "Advertising" exists (and components not present), include it.

    Also tries to keep obvious segment columns and ignores non-segment junk.
    """
    cols = [_clean_colname(c) for c in cols]
    colset = set(cols)

    ad_components = {"Google Search & Other", "YouTube Ads", "Google Network"}
    has_components = len(ad_components.intersection(colset)) >= 2  # allow if 2/3 exist

    use = [c for c in cols if c not in {"TOTAL"}]

    if has_components and "Advertising" in colset:
        use = [c for c in use if c != "Advertising"]

    # Keep only non-date numeric segment columns (caller ensures numeric frame)
    return use


# -----------------------------
# UI
# -----------------------------
st.sidebar.title("Controls")
force_refresh = st.sidebar.button("Force Refresh (ignore cache)")

try:
    tidy = load_segment_quarterly(force_refresh=force_refresh)
except Exception as e:
    st.error(f"Could not load live data from StockAnalysis.\n\nError: {e}")
    st.stop()

products = sorted(tidy["product"].unique().tolist())
default_product = "Advertising" if "Advertising" in products else (products[0] if products else None)

years = st.sidebar.slider("Forecast years", min_value=1, max_value=10, value=3, step=1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario)", min_value=0.00, max_value=0.30, value=0.00, step=0.01)

# Sidebar product dropdown (if there are products)
product = st.sidebar.selectbox("Product", products, index=products.index(default_product) if default_product in products else 0)

st.sidebar.markdown("---")
st.sidebar.subheader("Notes")
st.sidebar.write("Live data is scraped from StockAnalysis (History table).")
st.sidebar.write("If 'Advertising' AND its components are present, totals exclude 'Advertising' to avoid double-counting.")


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
        Interactive product-level quarterly revenue analysis and scenario forecasting.
        """
    )

st.caption("Data source: StockAnalysis → GOOG → Metrics → Revenue by Segment (History table).")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Product Forecast", "Total Forecast", "Assumptions & Notes", "Download"])


# -----------------------------
# TAB 1: Product Forecast
# -----------------------------
with tab1:
    seg = tidy[tidy["product"] == product].copy().sort_values("date")

    fc = forecast_series(seg[["date", "revenue"]], years=years, uplift_annual=uplift)
    end_fc = float(fc["forecast"].iloc[-1])

    base_fc = forecast_series(seg[["date", "revenue"]], years=years, uplift_annual=0.0)
    base_end = float(base_fc["forecast"].iloc[-1])
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

    st.subheader("Latest quarter breakdown check (sanity)")
    latest = seg.loc[seg["date"] == seg["date"].max(), ["date", "product", "revenue"]]
    st.write("Selected product latest value:", money_fmt(float(latest["revenue"].iloc[0])))


# -----------------------------
# TAB 2: Total Forecast
# -----------------------------
with tab2:
    # Build wide quarterly matrix
    wide = (
        tidy.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum")
        .sort_index()
        .dropna(how="all")
    )

    # Decide canonical columns to sum for TOTAL (avoid double-counting Advertising)
    use_cols = canonical_total_columns(list(wide.columns))
    wide_total = wide[use_cols].copy()

    wide_total["TOTAL"] = wide_total.sum(axis=1)

    # Forecast each included segment and sum forecasts
    future_steps = years * 4
    last_date = wide_total.index.max()
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=future_steps, freq="Q")

    total_fc_vals = np.zeros(future_steps, dtype=float)
    total_hi_vals = np.zeros(future_steps, dtype=float)
    total_lo_vals = np.zeros(future_steps, dtype=float)

    for p in use_cols:
        hist_p = wide_total[[p]].reset_index().rename(columns={p: "revenue"})
        fcp = forecast_series(hist_p[["date", "revenue"]], years=years, uplift_annual=uplift)
        total_fc_vals += fcp["forecast"].values
        total_hi_vals += fcp["hi"].values
        total_lo_vals += fcp["lo"].values

    total_hist = wide_total["TOTAL"].reset_index().rename(columns={"TOTAL": "revenue"})
    total_fc = pd.DataFrame({"date": future_dates, "forecast": total_fc_vals, "hi": total_hi_vals, "lo": total_lo_vals})

    fig_total = build_plot(
        total_hist[["date", "revenue"]],
        total_fc,
        title="Total Alphabet Revenue Forecast (Scenario)",
    )
    st.plotly_chart(fig_total, use_container_width=True)

    # Show which columns were used
    st.caption("TOTAL is computed by summing these columns (quarterly): " + ", ".join(use_cols))

    # Sanity check: show latest quarter sum vs components
    latest_q = wide_total.index.max()
    latest_row = wide_total.loc[latest_q, use_cols]
    latest_total = float(wide_total.loc[latest_q, "TOTAL"])
    st.subheader("Latest quarter total sanity check")
    st.write(f"Latest quarter: {latest_q.strftime('%b %d, %Y')}")
    st.write("Total:", money_fmt(latest_total))
    st.dataframe(
        pd.DataFrame({"Segment": latest_row.index, "Revenue": latest_row.values})
        .assign(Revenue_fmt=lambda d: d["Revenue"].map(money_fmt))
        .sort_values("Revenue", ascending=False),
        use_container_width=True,
    )


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
        - Computes recent quarter-over-quarter growth from the last few quarters
        - Uses that average growth as the baseline forward growth rate
        - Applies your extra CAGR uplift (annual) as an additional quarterly growth component
        - Adds an approximate uncertainty band that widens over time (based on recent growth volatility)
        """
    )

    st.subheader("Important: avoiding double-counting")
    st.write(
        """
        StockAnalysis may show both:
        - Advertising (aggregate)
        AND
        - Google Search & Other, YouTube Ads, Google Network (components)

        If components are present, this app excludes "Advertising" from the TOTAL sum.
        """
    )


# -----------------------------
# TAB 4: Download
# -----------------------------
with tab4:
    st.subheader("Download data")
    st.write("Use these downloads to share your results or reuse the dataset.")

    tidy_out = tidy.copy()
    tidy_out["revenue"] = tidy_out["revenue"].astype(float)

    st.download_button(
        label="Download tidy dataset (date, product, revenue) as CSV",
        data=tidy_out.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_revenue_by_segment_tidy.csv",
        mime="text/csv",
    )

    wide_out = tidy.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_out = wide_out.reset_index()

    st.download_button(
        label="Download wide dataset (date + all segments) as CSV",
        data=wide_out.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_revenue_by_segment_wide.csv",
        mime="text/csv",
    )

    st.caption("Source page: StockAnalysis → GOOG → Metrics → Revenue by Segment")
