import re
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go


# =============================
# CONFIG
# =============================
SEGMENT_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
REVENUE_URL = "https://stockanalysis.com/stocks/goog/revenue/"

UA_HEADERS_LIST = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    },
]

st.set_page_config(page_title="Alphabet Revenue Intelligence Dashboard", layout="wide")


# =============================
# HELPERS
# =============================
def money_to_float(x: str) -> float:
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "—", "-", "n/a"}:
        return np.nan

    s = s.replace(",", "").replace("−", "-").replace("\xa0", " ")

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

    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) * mult if m else np.nan


def money_fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = float(x)
    sign = "-" if x < 0 else ""
    ax = abs(x)
    if ax >= 1e12:
        return f"{sign}${ax/1e12:,.2f}T"
    if ax >= 1e9:
        return f"{sign}${ax/1e9:,.2f}B"
    if ax >= 1e6:
        return f"{sign}${ax/1e6:,.2f}M"
    return f"{sign}${ax:,.0f}"


def snap_to_quarter_end(dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt):
        return dt
    dt = pd.Timestamp(dt).normalize()
    dt = dt + pd.offsets.QuarterEnd(0)
    return pd.Timestamp(dt.date())


def fetch_html(url: str, params: Optional[dict] = None, timeout: int = 30) -> str:
    last_err = None
    for i, headers in enumerate(UA_HEADERS_LIST, 1):
        try:
            time.sleep(0.35)
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Request failed for {url}. Last error: {last_err}")


def read_html_best(html: str) -> List[pd.DataFrame]:
    # pandas.read_html is usually the most robust approach for sites like this
    return pd.read_html(html)


# =============================
# LOADERS (ROBUST)
# =============================
@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_total_revenue_quarterly() -> pd.DataFrame:
    """
    Official quarterly total revenue from:
    https://stockanalysis.com/stocks/goog/revenue/
    """
    # Try quarterly parameter first
    html = None
    try:
        html = fetch_html(REVENUE_URL, params={"p": "quarterly"})
    except Exception:
        # fallback (some pages ignore params)
        html = fetch_html(REVENUE_URL)

    dfs = read_html_best(html)

    # Find the revenue history table by columns
    best = None
    for df in dfs:
        cols = [c.lower() for c in df.columns.astype(str).tolist()]
        if any("quarter" in c for c in cols) and any("revenue" in c for c in cols):
            best = df
            break
    if best is None:
        raise ValueError("Could not find Revenue History table on /revenue/ page.")

    # Normalize column names
    col_map = {}
    for c in best.columns:
        cl = str(c).lower()
        if "quarter" in cl or "date" in cl:
            col_map[c] = "Quarter Ended"
        elif "revenue" in cl:
            col_map[c] = "Revenue"
        elif "change" in cl:
            col_map[c] = "Change"
        elif "growth" in cl:
            col_map[c] = "Growth"
    best = best.rename(columns=col_map)

    if "Quarter Ended" not in best.columns or "Revenue" not in best.columns:
        raise ValueError("Revenue table structure changed (missing Quarter Ended / Revenue).")

    out = best[["Quarter Ended", "Revenue"]].copy()
    out["date"] = pd.to_datetime(out["Quarter Ended"], errors="coerce").apply(snap_to_quarter_end)
    out["revenue"] = out["Revenue"].apply(money_to_float)
    out = out.dropna(subset=["date", "revenue"]).sort_values("date").reset_index(drop=True)
    out = out[["date", "revenue"]]

    if len(out) < 6:
        raise ValueError("Parsed too few quarterly revenue rows from /revenue/ page.")
    return out


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_segment_revenue_quarterly() -> pd.DataFrame:
    """
    Quarterly segment revenue from:
    https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/
    """
    html = fetch_html(SEGMENT_URL)
    dfs = read_html_best(html)

    # The segment page usually provides a wide table with Date + segment columns
    # We pick the widest table that contains a Date column.
    candidate = None
    best_width = -1
    for df in dfs:
        cols = [str(c).strip().lower() for c in df.columns]
        if any("date" == c or "quarter" in c for c in cols):
            if df.shape[1] > best_width:
                candidate = df
                best_width = df.shape[1]

    if candidate is None:
        raise ValueError("Could not locate a segment table with a Date/Quarter column.")

    # Standardize the first col as date
    dfw = candidate.copy()
    first_col = dfw.columns[0]
    dfw = dfw.rename(columns={first_col: "date"})
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce").apply(snap_to_quarter_end)
    dfw = dfw.dropna(subset=["date"]).copy()

    # Convert all other cols to numeric dollars
    for c in dfw.columns:
        if c == "date":
            continue
        dfw[c] = dfw[c].apply(money_to_float)

    dfw = dfw.sort_values("date").reset_index(drop=True)

    tidy = dfw.melt("date", var_name="segment", value_name="revenue").dropna(subset=["revenue"])
    tidy["segment"] = tidy["segment"].astype(str)
    tidy = tidy.sort_values(["segment", "date"]).reset_index(drop=True)

    if len(tidy) < 20:
        raise ValueError("Parsed too few segment rows. Segment table may have changed.")
    return tidy


# =============================
# FORECAST (job-ready: simple, explainable, stable)
# log-linear trend on quarterly revenue + scenario growth uplift
# =============================
def estimate_qoq_stats(series: pd.Series, lookback: int = 12) -> Tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) < 8:
        return 0.02, 0.06
    s = s.iloc[-lookback:] if len(s) > lookback else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 4:
        return 0.02, 0.06
    return float(g.mean()), float(g.std(ddof=1) if len(g) > 1 else 0.06)


def forecast_quarterly(hist: pd.DataFrame, years: int, scenario_uplift_annual: float, lookback: int = 12) -> pd.DataFrame:
    """
    Forecast future quarters using:
      base quarterly growth = avg QoQ over lookback
      scenario uplift adds extra annual growth converted to quarterly
    Uncertainty uses volatility of QoQ growth.
    """
    df = hist.sort_values("date").dropna(subset=["revenue"]).copy()
    last_date = pd.to_datetime(df["date"].max())
    last_val = float(df.loc[df["date"] == last_date, "revenue"].iloc[-1])

    mean_q, std_q = estimate_qoq_stats(df["revenue"], lookback=lookback)
    uplift_q = (1.0 + scenario_uplift_annual) ** (1.0 / 4.0) - 1.0
    gq = mean_q + uplift_q

    steps = years * 4
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=steps, freq="Q")
    future_dates = pd.to_datetime(pd.Series(future_dates).apply(snap_to_quarter_end))

    fc, hi, lo = [], [], []
    cur = last_val
    for i in range(1, steps + 1):
        cur *= (1.0 + gq)
        fc.append(cur)
        band = (std_q if std_q > 0 else 0.06) * 1.28 * np.sqrt(i)  # ~80% band
        hi.append(cur * (1.0 + band))
        lo.append(max(0, cur * (1.0 - band)))

    return pd.DataFrame({"date": future_dates, "forecast": fc, "hi": hi, "lo": lo})


# =============================
# PLOTS
# =============================
def plot_total(hist: pd.DataFrame, fc: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["revenue"],
        mode="lines+markers", name="Actual",
        line=dict(width=3), marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["hi"],
        mode="lines", name="Upper (80%)",
        line=dict(dash="dot", color="rgba(120,120,120,0.35)")
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["lo"],
        mode="lines", name="Lower (80%)",
        line=dict(dash="dot", color="rgba(120,120,120,0.35)"),
        fill="tonexty", fillcolor="rgba(120,120,120,0.12)"
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(width=2, dash="dash"), marker=dict(size=5)
    ))

    fig.update_layout(
        title=title,
        height=520,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Quarter (Period Ending)",
        yaxis_title="Revenue (USD, Quarterly)"
    )
    fig.update_xaxes(tickformat="%b %Y", showgrid=True, gridcolor="rgba(200,200,200,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.2)")
    return fig


def plot_segment_mix(seg_tidy: pd.DataFrame) -> go.Figure:
    wide = seg_tidy.pivot_table(index="date", columns="segment", values="revenue", aggfunc="sum").sort_index()
    wide = wide.dropna(how="all")

    fig = go.Figure()
    for c in wide.columns:
        fig.add_trace(go.Scatter(
            x=wide.index, y=wide[c],
            mode="lines",
            name=str(c),
            stackgroup="one"
        ))

    fig.update_layout(
        title="Segment Revenue Mix (Stacked, Quarterly)",
        height=520,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Quarter (Period Ending)",
        yaxis_title="Revenue (USD, Quarterly)"
    )
    fig.update_xaxes(tickformat="%b %Y", showgrid=True, gridcolor="rgba(200,200,200,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.2)")
    return fig


def plot_qoq_bars(total_hist: pd.DataFrame) -> go.Figure:
    df = total_hist.sort_values("date").copy()
    df["qoq"] = df["revenue"].diff()
    fig = go.Figure(go.Bar(x=df["date"], y=df["qoq"], name="QoQ Change"))
    fig.update_layout(
        title="Quarter-over-Quarter Revenue Change (USD)",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Quarter",
        yaxis_title="Change (USD)"
    )
    fig.update_xaxes(tickformat="%b %Y", showgrid=True, gridcolor="rgba(200,200,200,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.2)")
    return fig


# =============================
# UI
# =============================
st.title("Alphabet (Google) Revenue Intelligence Dashboard")
st.caption(
    "Quarterly total revenue is sourced from StockAnalysis /revenue/ (official Revenue History table). "
    "Segment mix is sourced from StockAnalysis revenue-by-segment page."
)

with st.sidebar:
    st.header("Controls")
    if st.button("Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Forecast Scenarios")
    years = st.slider("Forecast horizon (years)", 1, 10, 3, 1)
    lookback = st.slider("Lookback quarters (trend)", 8, 20, 12, 1)

    st.markdown("**Scenario uplift (extra annual growth)**")
    base_uplift = st.slider("Base uplift (%)", 0, 25, 0, 1) / 100.0
    bull_uplift = st.slider("Optimistic uplift (%)", 0, 35, 8, 1) / 100.0
    bear_uplift = st.slider("Conservative uplift (%)", 0, 25, 0, 1) / 100.0

    st.markdown("---")
    st.subheader("Segment View")
    seg_top_n = st.slider("Show top segments (by latest quarter)", 3, 12, 8, 1)

# Load data
with st.spinner("Loading TOTAL revenue (quarterly) from StockAnalysis..."):
    total_hist = load_total_revenue_quarterly()

with st.spinner("Loading segment revenue (quarterly) from StockAnalysis..."):
    seg_tidy = load_segment_revenue_quarterly()

# Prepare key metrics
last_date = pd.to_datetime(total_hist["date"].max())
last_rev = float(total_hist.loc[total_hist["date"] == last_date, "revenue"].iloc[-1])

qoq = total_hist.sort_values("date")["revenue"].diff().iloc[-1]
qoq_pct = total_hist.sort_values("date")["revenue"].pct_change().iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Quarter", last_date.strftime("%b %Y"))
c2.metric("Total Revenue (Latest)", money_fmt(last_rev))
c3.metric("QoQ Change", money_fmt(qoq), f"{(qoq_pct*100):.2f}%" if pd.notna(qoq_pct) else "—")
mean_q, std_q = estimate_qoq_stats(total_hist["revenue"], lookback=lookback)
c4.metric("Avg QoQ Growth (lookback)", f"{mean_q*100:.2f}%", f"Vol {std_q*100:.2f}%")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Total Forecast", "Segments", "Story & Insights", "Download"])

with tab1:
    st.subheader("Total Revenue Forecast (Quarterly, matches /revenue/ page)")

    fc_base = forecast_quarterly(total_hist, years=years, scenario_uplift_annual=base_uplift, lookback=lookback)
    fc_bull = forecast_quarterly(total_hist, years=years, scenario_uplift_annual=bull_uplift, lookback=lookback)
    fc_bear = forecast_quarterly(total_hist, years=years, scenario_uplift_annual=bear_uplift, lookback=lookback)

    # Plot base
    st.plotly_chart(
        plot_total(total_hist, fc_base, f"Base Scenario Forecast (+{base_uplift*100:.0f}% annual uplift)"),
        use_container_width=True
    )

    # Scenario comparison (end values)
    end_base = float(fc_base["forecast"].iloc[-1])
    end_bull = float(fc_bull["forecast"].iloc[-1])
    end_bear = float(fc_bear["forecast"].iloc[-1])

    s1, s2, s3 = st.columns(3)
    s1.metric("End Forecast (Base)", money_fmt(end_base))
    s2.metric("End Forecast (Optimistic)", money_fmt(end_bull))
    s3.metric("End Forecast (Conservative)", money_fmt(end_bear))

    st.plotly_chart(plot_qoq_bars(total_hist), use_container_width=True)

    with st.expander("Show the exact quarterly history (this should match StockAnalysis)"):
        show = total_hist.copy()
        show["Period"] = pd.to_datetime(show["date"]).dt.strftime("%b %d, %Y")
        show["Revenue"] = show["revenue"].apply(money_fmt)
        show["QoQ %"] = show["revenue"].pct_change().apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
        st.dataframe(show[["Period", "Revenue", "QoQ %"]].iloc[::-1], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Segment Revenue Mix")

    # Top-N segments by latest quarter revenue
    latest = seg_tidy["date"].max()
    latest_seg = seg_tidy[seg_tidy["date"] == latest].copy()
    top_segments = (
        latest_seg.sort_values("revenue", ascending=False)["segment"]
        .head(seg_top_n)
        .tolist()
    )
    seg_plot = seg_tidy[seg_tidy["segment"].isin(top_segments)].copy()

    st.plotly_chart(plot_segment_mix(seg_plot), use_container_width=True)

    # Contribution table (latest quarter)
    total_latest = float(latest_seg["revenue"].sum())
    contrib = latest_seg.copy()
    contrib["Share"] = contrib["revenue"] / total_latest
    contrib = contrib.sort_values("revenue", ascending=False)
    contrib["Revenue"] = contrib["revenue"].apply(money_fmt)
    contrib["Share"] = contrib["Share"].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(contrib[["segment", "Revenue", "Share"]].rename(columns={"segment": "Segment"}), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Story & Insights (Readable, Interview-Ready)")

    df = total_hist.sort_values("date").copy()
    df["qoq_pct"] = df["revenue"].pct_change()
    last4 = df.tail(4).copy()
    yoy = None
    if len(df) >= 5:
        # YoY for quarterly: compare to same quarter last year (4 quarters back)
        yoy = (df["revenue"].iloc[-1] / df["revenue"].iloc[-5] - 1.0) if df["revenue"].iloc[-5] != 0 else None

    st.markdown(
        f"""
**What the data says (total revenue):**
- Latest reported quarter ends **{last_date.strftime("%b %d, %Y")}** with revenue of **{money_fmt(last_rev)}**.
- Most recent QoQ change is **{money_fmt(qoq)}** ({(qoq_pct*100):.2f}%).
- Over the last {lookback} quarters, average QoQ growth is **{mean_q*100:.2f}%** with volatility of **{std_q*100:.2f}%**.
"""
    )

    if yoy is not None:
        st.markdown(f"- Approximate YoY change vs same quarter last year: **{yoy*100:.2f}%**.")

    st.markdown("---")
    st.markdown(
        """
**How to talk about this in an interview:**
- “I used an official source for quarterly totals and built a transparent growth + uncertainty model.”
- “Then I decomposed the story using segment mix to show what’s driving the total.”
- “Finally, I added scenario planning so stakeholders can stress-test assumptions.”
"""
    )

    st.markdown("---")
    st.subheader("Data Audit")
    st.write(f"Total quarters loaded: **{len(total_hist)}**")
    st.write(f"Segments records loaded: **{len(seg_tidy)}**")
    st.write(f"Segment date range: **{seg_tidy['date'].min().strftime('%b %Y')} → {seg_tidy['date'].max().strftime('%b %Y')}**")
    st.write(f"Total date range: **{total_hist['date'].min().strftime('%b %Y')} → {total_hist['date'].max().strftime('%b %Y')}**")

with tab4:
    st.subheader("Download Data")

    total_out = total_hist.copy()
    total_out["date"] = pd.to_datetime(total_out["date"]).dt.strftime("%Y-%m-%d")
    seg_out = seg_tidy.copy()
    seg_out["date"] = pd.to_datetime(seg_out["date"]).dt.strftime("%Y-%m-%d")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download TOTAL Revenue (Quarterly CSV)",
            data=total_out.to_csv(index=False).encode("utf-8"),
            file_name="goog_total_revenue_quarterly.csv",
            mime="text/csv",
        )

    with c2:
        st.download_button(
            "Download Segment Revenue (Quarterly CSV)",
            data=seg_out.to_csv(index=False).encode("utf-8"),
            file_name="goog_segment_revenue_quarterly.csv",
            mime="text/csv",
        )

    st.caption("Tip: Put this on GitHub with a clean README and screenshots. Recruiters love that.")
