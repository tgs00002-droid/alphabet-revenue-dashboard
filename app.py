import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from io import StringIO
import plotly.graph_objects as go
import streamlit as st

# Optional: statsmodels for forecasting (works but can warn on short series)
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -----------------------------
# Page config (makes it look legit)
# -----------------------------
st.set_page_config(
    page_title="Alphabet Revenue Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .kpi-card {
        border: 1px solid #eee;
        border-radius: 16px;
        padding: 16px 18px;
        background: #fff;
        box-shadow: 0 1px 10px rgba(0,0,0,0.04);
        height: 100%;
    }
    .small-muted { color: #666; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Helpers
# -----------------------------
def money_fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = float(x)
    if abs(x) >= 1e12:
        return f"${x/1e12:,.2f}T"
    if abs(x) >= 1e9:
        return f"${x/1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:,.2f}M"
    return f"${x:,.0f}"

def parse_money(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "").strip()
    if s.endswith("B"):
        return float(s[:-1]) * 1e9
    if s.endswith("M"):
        return float(s[:-1]) * 1e6
    return float(s)

# -----------------------------
# Data loader (with fallback)
# -----------------------------
URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

FALLBACK_CSV = """date,product,revenue
2021-03-31,Advertising,44560000000
2021-06-30,Advertising,50700000000
2021-09-30,Advertising,53000000000
2021-12-31,Advertising,61700000000
2022-03-31,Advertising,54000000000
2022-06-30,Advertising,56400000000
2022-09-30,Advertising,54400000000
2022-12-31,Advertising,59300000000
2023-03-31,Advertising,54000000000
2023-06-30,Advertising,58100000000
2023-09-30,Advertising,59600000000
2023-12-31,Advertising,65500000000
2024-03-31,Advertising,61600000000
2024-06-30,Advertising,64600000000
2024-09-30,Advertising,65500000000
2024-12-31,Advertising,72100000000
2021-03-31,Google Cloud,4047000000
2021-06-30,Google Cloud,4628000000
2021-09-30,Google Cloud,4990000000
2021-12-31,Google Cloud,4990000000
2022-03-31,Google Cloud,5580000000
2022-06-30,Google Cloud,6260000000
2022-09-30,Google Cloud,6868000000
2022-12-31,Google Cloud,7261000000
2023-03-31,Google Cloud,7488000000
2023-06-30,Google Cloud,8000000000
2023-09-30,Google Cloud,8330000000
2023-12-31,Google Cloud,9170000000
2024-03-31,Google Cloud,9534000000
2024-06-30,Google Cloud,10340000000
2024-09-30,Google Cloud,11350000000
2024-12-31,Google Cloud,12100000000
2021-03-31,Other Bets,198000000
2021-06-30,Other Bets,192000000
2021-09-30,Other Bets,182000000
2021-12-31,Other Bets,181000000
2022-03-31,Other Bets,440000000
2022-06-30,Other Bets,193000000
2022-09-30,Other Bets,209000000
2022-12-31,Other Bets,226000000
2023-03-31,Other Bets,288000000
2023-06-30,Other Bets,285000000
2023-09-30,Other Bets,297000000
2023-12-31,Other Bets,292000000
2024-03-31,Other Bets,495000000
2024-06-30,Other Bets,365000000
2024-09-30,Other Bets,388000000
2024-12-31,Other Bets,400000000
"""

@st.cache_data(show_spinner=False, ttl=60*60*6)
def load_data():
    # Try live scrape first
    try:
        r = requests.get(URL, headers=UA_HEADERS, timeout=25)
        r.raise_for_status()

        # FIX: pandas read_html prefers file-like for HTML strings
        tables = pd.read_html(StringIO(r.text))
        wide = tables[0].copy()
        wide.rename(columns={wide.columns[0]: "date"}, inplace=True)

        # Convert to quarter end timestamps
        wide["date"] = pd.to_datetime(wide["date"]).dt.to_period("Q").dt.end_time.dt.normalize()

        for c in wide.columns:
            if c != "date":
                wide[c] = wide[c].apply(parse_money)

        tidy = wide.melt("date", var_name="product", value_name="revenue").dropna()
        tidy = tidy.sort_values(["product", "date"]).reset_index(drop=True)
        tidy["source"] = "stockanalysis.com"
        return tidy
    except Exception:
        df = pd.read_csv(StringIO(FALLBACK_CSV))
        df["date"] = pd.to_datetime(df["date"])
        df["source"] = "fallback_sample"
        return df

# -----------------------------
# Forecasting
# -----------------------------
def sarimax_forecast(series, steps):
    # series: pd.Series indexed by date, values revenue
    y = series.astype(float).dropna()
    if len(y) < 8:
        # Too short; fallback to flat forecast
        last = float(y.iloc[-1])
        fc = np.array([last] * steps)
        return fc, fc, fc

    # Simple SARIMAX; we keep it stable
    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    mean = pred.predicted_mean.values
    ci = pred.conf_int(alpha=0.20)  # 80% interval
    lo = ci.iloc[:, 0].values
    hi = ci.iloc[:, 1].values
    return mean, lo, hi

def build_forecast(df, product, years, uplift):
    hist = df[df["product"] == product].copy()
    hist = hist.sort_values("date")
    hist = hist.dropna(subset=["revenue"])

    y = hist.set_index("date")["revenue"]
    steps = int(years * 4)

    base_mean, base_lo, base_hi = sarimax_forecast(y, steps)

    # Scenario: apply CAGR uplift on top of baseline forecast path
    # Convert uplift slider into an annual extra growth rate (e.g., 0.02 = +2% CAGR)
    uplift = float(uplift)
    t = np.arange(1, steps + 1)
    growth_factor = (1.0 + uplift) ** (t / 4.0)

    scn_mean = base_mean * growth_factor
    scn_lo = base_lo * growth_factor
    scn_hi = base_hi * growth_factor

    future_dates = pd.period_range(y.index.max().to_period("Q") + 1, periods=steps, freq="Q").end_time.normalize()

    fc = pd.DataFrame({
        "date": future_dates,
        "baseline": base_mean,
        "scenario": scn_mean,
        "lo80": scn_lo,
        "hi80": scn_hi
    })

    latest_hist = float(y.iloc[-1])
    start_fc = float(fc["baseline"].iloc[0])
    end_scn = float(fc["scenario"].iloc[-1])
    end_base = float(fc["baseline"].iloc[-1])
    delta = end_scn - end_base

    # implied forecast CAGR from first baseline forecast point to end scenario
    cagr = (end_scn / start_fc) ** (1 / years) - 1 if start_fc > 0 else np.nan

    return hist, fc, latest_hist, cagr, end_scn, delta

def make_plot(hist, fc, product, years, uplift):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["revenue"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["scenario"], mode="lines", name="Forecast (Scenario)"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["hi80"], mode="lines", name="80% high", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lo80"], mode="lines", name="80% low", line=dict(dash="dot")))
    fig.update_layout(
        title=f"{product}: Historical + {years}-Year Forecast (uplift {uplift*100:.1f}%)",
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis_title="Revenue (USD)",
        xaxis_title=""
    )
    return fig

# -----------------------------
# Header (with Google logo)
# -----------------------------
col1, col2 = st.columns([0.08, 0.92], vertical_alignment="center")
with col1:
    # Pulling logo from a public URL (simple). If you want, we can embed a local file too.
    st.image("https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", use_container_width=True)
with col2:
    st.markdown("## Alphabet (Google) Revenue Forecast Dashboard")
    st.markdown(
        "<div class='small-muted'>Interactive product-level revenue forecasting with a scenario-based CAGR uplift. "
        "Use the sliders to stress-test assumptions and export results.</div>",
        unsafe_allow_html=True
    )

st.divider()

# -----------------------------
# Sidebar controls
# -----------------------------
df = load_data()
products = sorted(df["product"].unique().tolist())

with st.sidebar:
    st.markdown("### Controls")
    product = st.selectbox("Product", products, index=products.index("Advertising") if "Advertising" in products else 0)
    years = st.slider("Forecast years", 3, 10, 5, 1)
    uplift = st.slider("Extra CAGR uplift (scenario)", 0.00, 0.10, 0.00, 0.01)
    st.markdown("---")
    st.markdown("**Notes**")
    st.caption("If live data is blocked, the app automatically uses a fallback sample dataset so the dashboard still runs.")

# -----------------------------
# Main view
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Product Forecast", "Total Forecast", "Assumptions & Notes", "Download"])

hist, fc, latest_hist, cagr, end_scn, delta = build_forecast(df, product, years, uplift)
fig = make_plot(hist, fc, product, years, uplift)

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='kpi-card'><div class='small-muted'>Latest Quarter (Hist)</div><h2>{money_fmt(latest_hist)}</h2></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi-card'><div class='small-muted'>Forecast CAGR (Scenario)</div><h2>{'—' if np.isnan(cagr) else f'{cagr*100:.2f}%'} </h2></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi-card'><div class='small-muted'>End Forecast (Scenario)</div><h2>{money_fmt(end_scn)}</h2></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi-card'><div class='small-muted'>Δ vs Baseline (end)</div><h2>{money_fmt(delta)}</h2></div>", unsafe_allow_html=True)

with tab1:
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Data source: {df['source'].iloc[0]} (auto-fallback enabled).")

with tab2:
    # Build total by summing across products
    totals = []
    for p in products:
        h2, f2, *_ = build_forecast(df, p, years, uplift)
        f2 = f2[["date", "scenario"]].rename(columns={"scenario": p})
        totals.append(f2.set_index("date"))
    total_df = pd.concat(totals, axis=1).fillna(0.0)
    total_df["Total"] = total_df.sum(axis=1)
    total_plot = go.Figure()
    total_plot.add_trace(go.Scatter(x=total_df.index, y=total_df["Total"], mode="lines", name="Total (Scenario)"))
    total_plot.update_layout(title="Total Alphabet Revenue Forecast (Scenario)", height=520, yaxis_title="Revenue (USD)")
    st.plotly_chart(total_plot, use_container_width=True)
    st.caption("This is the sum of per-product scenario forecasts (not a separately fit model).")

with tab3:
    st.markdown("### What this is")
    st.write(
        "This dashboard forecasts quarterly revenue by product (e.g., Advertising, Google Cloud, Other Bets) "
        "and lets you apply an **extra CAGR uplift** to stress-test growth assumptions."
    )
    st.markdown("### How to read it")
    st.write(
        "- **Historical** = observed quarterly revenue.\n"
        "- **Forecast (Scenario)** = baseline time-series forecast adjusted by your uplift.\n"
        "- **80% band** = uncertainty bounds from the underlying model."
    )
    st.markdown("### Caveats")
    st.write(
        "- Short histories can cause model instability; in those cases the app falls back to a simple flat projection.\n"
        "- This is a portfolio-style demo showing forecasting + scenario analysis, not investment advice."
    )

with tab4:
    st.markdown("### Download scenario forecast")
    out = fc.copy()
    out.insert(0, "product", product)
    st.dataframe(out, use_container_width=True)
    st.download_button(
        "Download CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"{product.lower().replace(' ','_')}_forecast.csv",
        mime="text/csv"
    )
