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
SOURCE_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

st.set_page_config(
    page_title="Alphabet (Google) Revenue Forecast Dashboard",
    layout="wide",
)

# -----------------------------
# PARSING HELPERS
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

    s = s.replace(",", "").replace("−", "-").replace("\xa0", " ")

    # Try plain numeric
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


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_revenue_by_segment_wide() -> pd.DataFrame:
    """
    Scrape the History table and return a wide dataframe:
      date + one column per segment (values as float USD)
    """
    r = requests.get(SOURCE_URL, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    # Try to find the History section
    history_h2 = None
    for h in soup.find_all(["h2", "h3"]):
        if h.get_text(strip=True).lower() == "history":
            history_h2 = h
            break

    table = history_h2.find_next("table") if history_h2 else None

    # Fallback: largest table
    if table is None:
        tables = soup.find_all("table")
        if not tables:
            raise ValueError("No tables found on the page.")
        table = max(tables, key=lambda t: len(t.find_all("tr")))

    # Headers
    thead = table.find("thead")
    if thead:
        header_cells = thead.find_all(["th", "td"])
        headers = [c.get_text(" ", strip=True) for c in header_cells]
    else:
        first_row = table.find("tr")
        headers = [c.get_text(" ", strip=True) for c in first_row.find_all(["th", "td"])]

    headers = [h.replace("\xa0", " ").strip() for h in headers]
    if headers and headers[0].lower() != "date":
        headers[0] = "Date"

    # Rows
    rows = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        vals = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in cells]

        if vals and vals[0].lower() == "date":
            continue

        if len(vals) < len(headers):
            vals += [""] * (len(headers) - len(vals))
        if len(vals) > len(headers):
            vals = vals[:len(headers)]

        rows.append(vals)

    wide = pd.DataFrame(rows, columns=headers)

    wide.rename(columns={"Date": "date"}, inplace=True)
    wide["date"] = pd.to_datetime(wide["date"], errors="coerce")

    for c in wide.columns:
        if c != "date":
            wide[c] = wide[c].apply(money_to_float)

    wide = wide.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return wide


# -----------------------------
# TTM -> QUARTERLY CONVERSION
# -----------------------------
def ttm_to_quarterly(ttm: pd.Series, seed_method: str = "ttm_div_4") -> pd.Series:
    """
    Convert a TTM quarterly series into an estimated quarterly series.

    Recurrence:
      Q_t = TTM_t - TTM_{t-1} + Q_{t-4}

    Needs seeding for first 4 quarters.
    seed_method:
      - "ttm_div_4": Q_0..Q_3 = TTM_0/4, TTM_1/4, TTM_2/4, TTM_3/4  (simple, stable)
    """
    s = ttm.astype(float).copy()
    q = pd.Series(index=s.index, dtype=float)

    if len(s) == 0:
        return q

    # Seed first 4
    for i in range(min(4, len(s))):
        if seed_method == "ttm_div_4":
            q.iloc[i] = s.iloc[i] / 4.0
        else:
            q.iloc[i] = s.iloc[i] / 4.0

    # Recurrence for the rest
    for i in range(4, len(s)):
        if pd.isna(s.iloc[i]) or pd.isna(s.iloc[i - 1]) or pd.isna(q.iloc[i - 4]):
            q.iloc[i] = np.nan
        else:
            q.iloc[i] = (s.iloc[i] - s.iloc[i - 1]) + q.iloc[i - 4]

    return q


def convert_wide_ttm_to_quarterly(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Apply TTM->Quarterly conversion to every segment column.
    Returns a new wide dataframe (same columns), with quarterly values.
    """
    out = wide.copy()
    for c in out.columns:
        if c == "date":
            continue
        out[c] = ttm_to_quarterly(out[c], seed_method="ttm_div_4")
    return out


# -----------------------------
# FORECAST
# -----------------------------
def estimate_qoq_growth(series: pd.Series, lookback_quarters: int = 8) -> tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) < 6:
        return 0.02, 0.05

    s = s.iloc[-lookback_quarters:] if len(s) > lookback_quarters else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 3:
        return 0.02, 0.05

    return float(g.mean()), float(g.std(ddof=1) if len(g) > 1 else 0.05)


def forecast_quarterly(hist_df: pd.DataFrame, years: int, uplift: float, lookback_quarters: int = 8) -> pd.DataFrame:
    """
    Forecast quarterly values forward.
    uplift is extra ANNUAL CAGR (converted to quarterly).
    """
    df = hist_df.sort_values("date").dropna(subset=["revenue"]).copy()
    last_date = df["date"].max()
    last_val = float(df.loc[df["date"] == last_date, "revenue"].iloc[-1])

    mean_q, std_q = estimate_qoq_growth(df["revenue"], lookback_quarters=lookback_quarters)

    uplift_q = (1.0 + uplift) ** (1.0 / 4.0) - 1.0
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


def money_fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
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


def build_plot(hist: pd.DataFrame, fc: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hist["date"], y=hist["revenue"], mode="lines", name="Historical"))

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


# -----------------------------
# LOAD + BUILD DATASETS
# -----------------------------
wide_ttm = load_revenue_by_segment_wide()
wide_q = convert_wide_ttm_to_quarterly(wide_ttm)

# Build tidy views for UI
tidy_ttm = wide_ttm.melt("date", var_name="product", value_name="revenue").dropna()
tidy_q = wide_q.melt("date", var_name="product", value_name="revenue").dropna()

products = sorted(tidy_q["product"].unique().tolist())
default_product = "Advertising" if "Advertising" in products else products[0]

# -----------------------------
# UI
# -----------------------------
st.sidebar.title("Controls")

product = st.sidebar.selectbox("Product", products, index=products.index(default_product))
years = st.sidebar.slider("Forecast years", min_value=1, max_value=10, value=3, step=1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario)", min_value=0.00, max_value=0.30, value=0.00, step=0.01)
view_mode = st.sidebar.radio("View mode", ["Quarterly (converted)", "TTM (raw)"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("TTM → Quarterly uses Qₜ = TTMₜ − TTMₜ₋₁ + Qₜ₋₄ with first 4 quarters seeded as TTM/4.")

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
        Interactive segment forecasting with scenario-based CAGR uplift. Data source is a TTM table; quarterly values are reconstructed for modeling.
        """
    )

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Product Forecast", "Total Forecast", "Assumptions & Notes", "Download"])

# Choose which series to show in chart table
tidy_show = tidy_q if view_mode.startswith("Quarterly") else tidy_ttm
y_label = "Revenue (USD) — Quarterly" if view_mode.startswith("Quarterly") else "Revenue (USD) — TTM"

# -----------------------------
# TAB 1: Product Forecast (FORECAST ALWAYS USES QUARTERLY)
# -----------------------------
with tab1:
    # Display series based on view_mode
    show_seg = tidy_show[tidy_show["product"] == product].sort_values("date")

    # Forecast uses quarterly reconstructed
    seg_q = tidy_q[tidy_q["product"] == product].sort_values("date")
    fc = forecast_quarterly(seg_q[["date", "revenue"]], years=years, uplift=uplift)

    end_fc = float(fc["forecast"].iloc[-1])
    base_end = float(forecast_quarterly(seg_q[["date", "revenue"]], years=years, uplift=0.0)["forecast"].iloc[-1])
    delta = end_fc - base_end

    k1, k2, k3 = st.columns(3)
    k1.metric("End Forecast (Scenario)", money_fmt(end_fc))
    k2.metric("Δ vs Baseline", money_fmt(delta))
    k3.metric("Last Reported Quarter", show_seg["date"].max().strftime("%b %d, %Y"))

    fig = build_plot(
        show_seg[["date", "revenue"]],
        fc,
        title=f"{product}: Historical + {years}-Year Forecast (uplift {uplift*100:.1f}%)",
        y_label=y_label,
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 2: Total Forecast (sum of segment forecasts, quarterly)
# -----------------------------
with tab2:
    # Quarterly wide for totals
    wide_q2 = wide_q.set_index("date").sort_index().dropna(how="all")

    # total historical (quarterly)
    total_hist = wide_q2.sum(axis=1).reset_index()
    total_hist.columns = ["date", "revenue"]

    # forecast each segment quarterly and sum forecasts
    future_steps = years * 4
    last_date = wide_q2.index.max()
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=future_steps, freq="Q")

    total_fc_vals = np.zeros(future_steps, dtype=float)
    total_hi_vals = np.zeros(future_steps, dtype=float)
    total_lo_vals = np.zeros(future_steps, dtype=float)

    for c in wide_q2.columns:
        hist_c = wide_q2[[c]].reset_index().rename(columns={c: "revenue"})
        fcc = forecast_quarterly(hist_c[["date", "revenue"]], years=years, uplift=uplift)
        total_fc_vals += fcc["forecast"].values
        total_hi_vals += fcc["hi"].values
        total_lo_vals += fcc["lo"].values

    total_fc = pd.DataFrame({"date": future_dates, "forecast": total_fc_vals, "hi": total_hi_vals, "lo": total_lo_vals})

    fig_total = build_plot(
        total_hist[["date", "revenue"]],
        total_fc,
        title="Total Alphabet Revenue Forecast (Scenario) — Quarterly (sum of segments)",
        y_label="Revenue (USD) — Quarterly",
    )
    st.plotly_chart(fig_total, use_container_width=True)
    st.caption("Total forecast = sum of per-segment quarterly forecasts (not a separately fit total model).")

# -----------------------------
# TAB 3: Assumptions
# -----------------------------
with tab3:
    st.subheader("TTM → Quarterly conversion")
    st.write(
        """
        The source table is TTM (trailing-twelve-month) revenue at each quarter-end.

        We reconstruct quarterly revenue per segment using:
        **Qₜ = TTMₜ − TTMₜ₋₁ + Qₜ₋₄**

        The first 4 quarters are seeded as **TTM/4** (a standard seeding approach).
        """
    )

    st.subheader("Forecast method")
    st.write(
        """
        Forecasting is done on the reconstructed quarterly series:
        - Compute recent QoQ growth over a lookback window
        - Apply your extra annual CAGR uplift (converted to quarterly)
        - Add a widening uncertainty band based on recent growth volatility
        """
    )

# -----------------------------
# TAB 4: Download
# -----------------------------
with tab4:
    st.subheader("Download data")

    st.download_button(
        "Download RAW TTM (wide) as CSV",
        data=wide_ttm.to_csv(index=False).encode("utf-8"),
        file_name="goog_revenue_by_segment_ttm_wide.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download QUARTERLY (reconstructed wide) as CSV",
        data=wide_q.to_csv(index=False).encode("utf-8"),
        file_name="goog_revenue_by_segment_quarterly_wide.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download QUARTERLY (reconstructed tidy) as CSV",
        data=tidy_q.to_csv(index=False).encode("utf-8"),
        file_name="goog_revenue_by_segment_quarterly_tidy.csv",
        mime="text/csv",
    )

    st.caption("Source: StockAnalysis → GOOG → Metrics → Revenue by Segment (History table)")
