import io
import re
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =========================
# CONFIG
# =========================
SEGMENT_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
FINANCIALS_URL = "https://stockanalysis.com/stocks/goog/financials/"

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

st.set_page_config(page_title="Alphabet Revenue Forecast Dashboard", layout="wide")

# =========================
# HELPERS
# =========================
def parse_money_to_float(x):
    """Convert strings like '56.57B', '344.00M', '-207.00M', '1.2T' to float dollars."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()

    if s in ["", "—", "-", "None", "nan", "NaN"]:
        return np.nan

    # remove commas and $ signs
    s = s.replace(",", "").replace("$", "")

    # Handle parentheses negatives e.g. (1.23B)
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # Match number + suffix
    m = re.match(r"^(-?\d+(\.\d+)?)([KMBT])?$", s, re.IGNORECASE)
    if not m:
        # Sometimes it's just a raw number
        try:
            val = float(s)
            return -val if neg else val
        except:
            return np.nan

    num = float(m.group(1))
    suf = (m.group(3) or "").upper()

    mult = 1.0
    if suf == "K":
        mult = 1e3
    elif suf == "M":
        mult = 1e6
    elif suf == "B":
        mult = 1e9
    elif suf == "T":
        mult = 1e12

    val = num * mult
    if neg:
        val = -val
    return val


def fetch_tables(url):
    """Fetch HTML and return list of tables via pandas.read_html."""
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    html = r.text
    tables = pd.read_html(io.StringIO(html))
    return tables


def standardize_quarter_date(col):
    """
    Convert quarter-like headers into timestamps.
    StockAnalysis often uses like 'Sep 30, 2025'.
    """
    try:
        return pd.to_datetime(col)
    except:
        return None


def wide_table_to_long_quarters(df_wide, label_col="Segment"):
    """
    Convert wide table:
    Segment | Sep 30, 2025 | Jun 30, 2025 | ...
    -> long:
    date, segment, value
    """
    df = df_wide.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Find segment column name
    if label_col not in df.columns:
        # try first column
        label_col = df.columns[0]

    # Identify date columns
    date_cols = []
    for c in df.columns:
        if c == label_col:
            continue
        dt = standardize_quarter_date(c)
        if dt is not None:
            date_cols.append(c)

    if not date_cols:
        # Nothing to parse
        return pd.DataFrame(columns=["date", "segment", "value"])

    # Melt
    tidy = df.melt(id_vars=[label_col], value_vars=date_cols,
                   var_name="date", value_name="value")
    tidy.rename(columns={label_col: "segment"}, inplace=True)

    tidy["date"] = pd.to_datetime(tidy["date"])
    tidy["value"] = tidy["value"].apply(parse_money_to_float)
    tidy = tidy.dropna(subset=["value"])

    # Sort oldest -> newest
    tidy = tidy.sort_values(["segment", "date"]).reset_index(drop=True)
    return tidy


def ttm_to_quarterly(ttm_long):
    """
    Correct conversion:
    Quarter(t) = TTM(t) - TTM(t-1)
    Per segment.
    """
    df = ttm_long.copy().sort_values(["segment", "date"])
    df["value_q"] = df.groupby("segment")["value"].diff(1)
    q = df.dropna(subset=["value_q"]).copy()
    q.rename(columns={"value_q": "value"}, inplace=True)
    q = q[["date", "segment", "value"]].sort_values(["segment", "date"]).reset_index(drop=True)
    return q


@st.cache_data(ttl=60 * 60)
def load_segment_quarterly():
    """
    Try to get the QUARTERLY segment table directly.
    If only TTM exists, convert TTM -> quarterly properly.
    """
    tables = fetch_tables(SEGMENT_URL)

    # Heuristic: pick a table that contains segment names like "Google Search" etc.
    candidate = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        firstcol = str(t.columns[0]).lower()
        if "segment" in firstcol or "segment" in " ".join(cols):
            # likely the segment table
            if t.shape[1] >= 4:
                candidate = t
                break

    if candidate is None:
        raise ValueError("Could not find segment table on the page.")

    # Convert wide -> long (attempt as quarterly)
    long_try = wide_table_to_long_quarters(candidate, label_col=candidate.columns[0])

    # Detect if it's TTM by checking if there is a column literally named "TTM"
    # or if the page served a TTM table instead of quarter dates.
    is_ttm = False
    raw_cols = [str(c).strip().upper() for c in candidate.columns]
    if "TTM" in raw_cols:
        is_ttm = True

    # Another detection: if dates exist but are not quarters and only one per year etc.
    # We'll primarily rely on explicit "TTM" or user scenario.
    if is_ttm:
        # If candidate includes a TTM column, reshape differently:
        # Expect: Segment | TTM | ... (maybe multiple)
        # We'll treat each date-like column as period, but if no dates, we cannot forecast.
        # Try to find any date columns in candidate; if none, fallback fail.
        ttm_long = wide_table_to_long_quarters(candidate, label_col=candidate.columns[0])
        if ttm_long.empty:
            raise ValueError("TTM table found but no time columns to convert.")
        q = ttm_to_quarterly(ttm_long)
        return q

    # If not flagged TTM, assume it's already quarterly
    # But if it looks like TTM (few columns, no quarter dates), try a different table
    if long_try.empty:
        # attempt: find any other table with many date columns
        best = None
        best_count = 0
        for t in tables:
            temp = wide_table_to_long_quarters(t, label_col=t.columns[0])
            if temp.shape[0] > best_count:
                best = temp
                best_count = temp.shape[0]
        if best is None or best.empty:
            raise ValueError("Could not parse any quarterly time series from the segment page.")
        return best

    return long_try


@st.cache_data(ttl=60 * 60)
def load_income_statement_quarterly():
    """
    Load quarterly income statement table from StockAnalysis financials page.
    Returns tidy: date, metric, value
    """
    tables = fetch_tables(FINANCIALS_URL)

    # Look for a table that has a first column like "Revenue" "Net Income" etc
    best = None
    for t in tables:
        # Income statements often have first column "Item" or something,
        # and then quarter columns.
        if t.shape[0] >= 10 and t.shape[1] >= 4:
            # try if it has common metrics in first column
            first_col = t.columns[0]
            values = t[first_col].astype(str).str.lower().tolist()
            if any("revenue" in v for v in values) and any("net income" in v for v in values):
                best = t
                break

    if best is None:
        raise ValueError("Could not find an income statement table on the financials page.")

    # Convert wide to long, but here rows are metrics and columns are dates.
    df = best.copy()
    df.columns = [str(c).strip() for c in df.columns]
    metric_col = df.columns[0]

    # Identify date columns
    date_cols = []
    for c in df.columns[1:]:
        if standardize_quarter_date(c) is not None:
            date_cols.append(c)

    if not date_cols:
        raise ValueError("Income statement table found but no quarterly date columns were detected.")

    long = df.melt(id_vars=[metric_col], value_vars=date_cols, var_name="date", value_name="value")
    long.rename(columns={metric_col: "metric"}, inplace=True)
    long["date"] = pd.to_datetime(long["date"])
    long["value"] = long["value"].apply(parse_money_to_float)
    long = long.dropna(subset=["value"])
    long = long.sort_values(["metric", "date"]).reset_index(drop=True)
    return long


def make_forecast(series_q, years=5, extra_cagr=0.0):
    """
    series_q: pd.Series indexed by date (quarter end), values in dollars
    Forecast uses CAGR from last 8 quarters (2 years) if possible.
    Creates quarterly forecast (years*4) with an 80% band from growth volatility.
    """
    s = series_q.dropna().copy()
    if len(s) < 6:
        # too short, fallback to flat growth
        base_cagr = 0.02
    else:
        # Use last 8 quarters to estimate CAGR
        tail = s.iloc[-8:] if len(s) >= 8 else s
        start = tail.iloc[0]
        end = tail.iloc[-1]
        n_years = max((len(tail) - 1) / 4.0, 0.25)
        if start <= 0 or end <= 0:
            base_cagr = 0.02
        else:
            base_cagr = (end / start) ** (1 / n_years) - 1

    scn_cagr = base_cagr + extra_cagr

    # quarterly growth rate
    qg = (1 + scn_cagr) ** (1/4.0) - 1

    # volatility from historical QoQ growth
    if len(s) >= 6:
        g = s.pct_change().dropna()
        vol = float(np.nanstd(g))
    else:
        vol = 0.05

    last_date = s.index[-1]
    last_val = float(s.iloc[-1])

    periods = int(years * 4)
    future_dates = pd.date_range(last_date, periods=periods+1, freq="Q")[1:]

    fc_vals = []
    hi_vals = []
    lo_vals = []

    cur = last_val
    for i in range(periods):
        cur = cur * (1 + qg)
        fc_vals.append(cur)

        # 80% band approx: +/- 1.28 * vol
        band = 1.28 * vol
        hi_vals.append(cur * (1 + band))
        lo_vals.append(cur * (1 - band))

    fc = pd.DataFrame({
        "date": future_dates,
        "forecast": fc_vals,
        "hi": hi_vals,
        "lo": lo_vals
    })
    return base_cagr, scn_cagr, fc


def money_fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e12:
        return f"{sign}${x/1e12:,.2f}T"
    if x >= 1e9:
        return f"{sign}${x/1e9:,.2f}B"
    if x >= 1e6:
        return f"{sign}${x/1e6:,.2f}M"
    return f"{sign}${x:,.0f}"


# =========================
# UI HEADER
# =========================
col1, col2 = st.columns([1, 12])
with col1:
    st.image(
        "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png",
        width=90
    )
with col2:
    st.markdown(
        """
        # Alphabet (Google) Revenue Forecast Dashboard
        Interactive product-level **quarterly** revenue analysis and scenario forecasting.
        """
    )

st.caption(
    "Data source: StockAnalysis (segment revenue + income statement). "
    "If the site blocks scraping, the app may need a redeploy or cached dataset."
)

# =========================
# LOAD DATA
# =========================
with st.spinner("Loading segment revenue data..."):
    seg_long = load_segment_quarterly()

with st.spinner("Loading income statement data..."):
    inc_long = load_income_statement_quarterly()

# Pivot segment quarterly for easier use
seg_piv = seg_long.pivot_table(index="date", columns="segment", values="value", aggfunc="sum").sort_index()

# Total revenue (sum of segments)
seg_piv["TOTAL (Sum of Segments)"] = seg_piv.sum(axis=1)

# Income statement pivot
inc_piv = inc_long.pivot_table(index="date", columns="metric", values="value", aggfunc="sum").sort_index()


# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Controls")

segments = [c for c in seg_piv.columns if c != "TOTAL (Sum of Segments)"]
default_seg = "Advertising" if "Advertising" in segments else segments[0]

product = st.sidebar.selectbox("Product / Segment", segments, index=segments.index(default_seg))
years = st.sidebar.slider("Forecast years", min_value=1, max_value=10, value=5, step=1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario)", min_value=0.0, max_value=0.10, value=0.00, step=0.005)

st.sidebar.divider()
st.sidebar.subheader("Notes")
st.sidebar.write(
    "- This app uses **quarterly values**.\n"
    "- If the webpage served **TTM**, we convert using:\n"
    "  **Quarter(t) = TTM(t) − TTM(t−1)**\n"
    "- Negative values (ex: hedging) are preserved."
)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Segment (Quarterly)", "Product Forecast", "Income Statement", "Download"])

# -------------------------
# TAB 1: STACKED SEGMENTS
# -------------------------
with tab1:
    st.subheader("Segment Revenue (Quarterly)")

    # build stacked bar like your screenshot
    df_stack = seg_piv.drop(columns=["TOTAL (Sum of Segments)"]).copy()
    df_stack = df_stack.dropna(how="all")
    df_stack = df_stack.tail(16)  # last 16 quarters

    fig = go.Figure()
    for seg in df_stack.columns:
        fig.add_trace(go.Bar(
            x=df_stack.index,
            y=df_stack[seg],
            name=seg
        ))

    fig.update_layout(
        barmode="stack",
        height=550,
        xaxis_title="Quarter",
        yaxis_title="Revenue (USD)",
        legend_title="Segments",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Tip: Hover a quarter to see each segment’s revenue. This uses **true quarterly values**.")

# -------------------------
# TAB 2: PRODUCT FORECAST
# -------------------------
with tab2:
    st.subheader(f"{product}: Historical + Forecast")

    s = seg_piv[product].dropna()
    base_cagr, scn_cagr, fc = make_forecast(s, years=years, extra_cagr=uplift)

    # KPIs
    end_fc = fc["forecast"].iloc[-1] if len(fc) else np.nan
    last_hist = s.iloc[-1] if len(s) else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Last Quarter", money_fmt(last_hist))
    k2.metric("End Forecast", money_fmt(end_fc))
    k3.metric("Baseline CAGR", f"{base_cagr*100:.2f}%")
    k4.metric("Scenario CAGR", f"{scn_cagr*100:.2f}%")

    # Plotly line + band
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Historical"))
    fig2.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], mode="lines", name="Forecast (Scenario)"))
    fig2.add_trace(go.Scatter(x=fc["date"], y=fc["hi"], mode="lines", name="80% High", line=dict(dash="dot")))
    fig2.add_trace(go.Scatter(x=fc["date"], y=fc["lo"], mode="lines", name="80% Low", line=dict(dash="dot")))

    fig2.update_layout(
        height=520,
        xaxis_title="Quarter",
        yaxis_title="Revenue (USD)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "Forecast method: baseline CAGR estimated from recent quarters, scenario adds uplift. "
        "Band is based on historical QoQ growth volatility (approx 80%)."
    )

# -------------------------
# TAB 3: INCOME STATEMENT
# -------------------------
with tab3:
    st.subheader("Income Statement (Quarterly)")

    metrics = list(inc_piv.columns)
    default_metric = "Revenue" if "Revenue" in metrics else metrics[0]
    metric = st.selectbox("Metric", metrics, index=metrics.index(default_metric))

    s2 = inc_piv[metric].dropna()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=s2.index, y=s2.values, mode="lines+markers", name=metric))
    fig3.update_layout(
        height=520,
        xaxis_title="Quarter",
        yaxis_title=f"{metric} (USD)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        s2.reset_index().rename(columns={"index": "date", metric: "value"}),
        use_container_width=True
    )

# -------------------------
# TAB 4: DOWNLOAD
# -------------------------
with tab4:
    st.subheader("Download Data")

    seg_csv = seg_long.copy()
    seg_csv["value"] = seg_csv["value"].astype(float)
    seg_csv_out = seg_csv.to_csv(index=False).encode("utf-8")

    inc_csv = inc_long.copy()
    inc_csv["value"] = inc_csv["value"].astype(float)
    inc_csv_out = inc_csv.to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Segment Revenue (Quarterly) CSV",
            data=seg_csv_out,
            file_name="alphabet_segment_revenue_quarterly.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "Download Income Statement (Quarterly) CSV",
            data=inc_csv_out,
            file_name="alphabet_income_statement_quarterly.csv",
            mime="text/csv"
        )

    st.caption("If you deploy this on Streamlit Community Cloud, share the app URL and people can interact with it live.")
