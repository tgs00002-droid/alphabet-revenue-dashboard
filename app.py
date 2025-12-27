import re
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import plotly.graph_objects as go
import streamlit as st


# =============================
# CONFIG
# =============================
SEGMENT_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
YAHOO_FIN_URL = "https://finance.yahoo.com/quote/GOOGL/financials"

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}

st.set_page_config(
    page_title="Alphabet (Google) Revenue Forecast Dashboard",
    layout="wide",
)


# =============================
# UTILS
# =============================
def money_to_float(x: str) -> float:
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


def snap_to_quarter_end(dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt):
        return dt
    dt = pd.Timestamp(dt).normalize()
    dt = dt + pd.offsets.QuarterEnd(0)
    return pd.Timestamp(dt.date())


def quarter_tickvals(dates: pd.Series) -> List[pd.Timestamp]:
    d = pd.to_datetime(dates).dropna().sort_values().unique()
    d = pd.Series(d).apply(snap_to_quarter_end).drop_duplicates().sort_values()
    return list(pd.to_datetime(d))


def build_plot_lines(hist: pd.DataFrame, fc: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["revenue"],
        mode="lines", name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["hi"],
        mode="lines", name="80% high (approx)",
        line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["lo"],
        mode="lines", name="80% low (approx)",
        line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["forecast"],
        mode="lines", name="Forecast (Scenario)"
    ))

    ticks = quarter_tickvals(pd.concat([hist["date"], fc["date"]], ignore_index=True))
    fig.update_layout(
        title=title,
        height=520,
        xaxis_title="Quarter (Period Ending)",
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        tickformat="%b %d, %Y",
        showgrid=False
    )
    return fig


def build_bar_income(df: pd.DataFrame, metric: str) -> go.Figure:
    d = df[df["metric"] == metric].sort_values("date").copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["date"], y=d["value"], name=metric))

    ticks = quarter_tickvals(d["date"])
    fig.update_layout(
        title=f"GOOGL Income Statement: {metric}",
        height=520,
        xaxis_title="Quarter (Period Ending)",
        yaxis_title=f"{metric} (USD)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        tickformat="%b %d, %Y",
        showgrid=False
    )
    return fig


# =============================
# STOCKANALYSIS SEGMENT SCRAPE (QUARTERLY)
# =============================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_revenue_by_segment_quarterly() -> pd.DataFrame:
    r = requests.get(SEGMENT_URL, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    history_h = None
    for h in soup.find_all(["h2", "h3"]):
        if h.get_text(strip=True).lower() == "history":
            history_h = h
            break

    table = history_h.find_next("table") if history_h else None
    if table is None:
        tables = soup.find_all("table")
        if not tables:
            raise ValueError("No tables found on StockAnalysis page.")
        table = max(tables, key=lambda t: len(t.find_all("tr")))

    thead = table.find("thead")
    if thead:
        headers = [c.get_text(" ", strip=True) for c in thead.find_all(["th", "td"])]
    else:
        first_row = table.find("tr")
        headers = [c.get_text(" ", strip=True) for c in first_row.find_all(["th", "td"])]

    headers = [h.replace("\xa0", " ").strip() for h in headers]
    if headers and headers[0].lower() != "date":
        headers[0] = "Date"

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
            vals = vals + [""] * (len(headers) - len(vals))
        elif len(vals) > len(headers):
            vals = vals[: len(headers)]

        rows.append(vals)

    wide = pd.DataFrame(rows, columns=headers)
    wide.rename(columns={"Date": "date"}, inplace=True)
    wide["date"] = pd.to_datetime(wide["date"], errors="coerce")
    wide = wide.dropna(subset=["date"]).copy()

    for c in wide.columns:
        if c != "date":
            wide[c] = wide[c].apply(money_to_float)

    wide = wide.sort_values("date").reset_index(drop=True)
    tidy = wide.melt("date", var_name="product", value_name="revenue").dropna()
    tidy = tidy.sort_values(["product", "date"]).reset_index(drop=True)
    tidy["date"] = pd.to_datetime(tidy["date"])
    return tidy


def compute_ttm_from_quarterly(tidy_q: pd.DataFrame) -> pd.DataFrame:
    df = tidy_q.copy().sort_values(["product", "date"])
    df["revenue_ttm"] = df.groupby("product")["revenue"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    out = df.dropna(subset=["revenue_ttm"]).rename(columns={"revenue_ttm": "revenue"})
    return out[["date", "product", "revenue"]].reset_index(drop=True)


def segment_sum_total_from_quarterly(seg_q: pd.DataFrame) -> pd.DataFrame:
    """
    Build TOTAL from the *true quarterly* segment row sums.
    This fixes the issue where totals looked like TTM when user wanted Quarterly.
    """
    wide_q = seg_q.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_q = wide_q.dropna(how="all")

    # Use ALL segment columns in the table (this matches your screenshot perfectly)
    # because Search + Cloud + Subscriptions + YouTube + Network + Other Bets + Hedging
    # sums to Yahoo Total Revenue (ex: 102.346B).
    total = wide_q.sum(axis=1, min_count=1)

    out = total.reset_index().rename(columns={0: "revenue"})
    out.columns = ["date", "revenue"]
    return out


# =============================
# FORECASTING
# =============================
def estimate_growth_q(series: pd.Series, lookback_quarters: int = 8) -> Tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) < 6:
        return 0.02, 0.05
    s = s.iloc[-lookback_quarters:] if len(s) > lookback_quarters else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 3:
        return 0.02, 0.05
    return float(g.mean()), float(g.std(ddof=1) if len(g) > 1 else 0.05)


def forecast_series(hist_df: pd.DataFrame, years: int, uplift_annual: float, lookback_quarters: int = 8) -> pd.DataFrame:
    df = hist_df.sort_values("date").copy()
    df = df.dropna(subset=["revenue"])
    if df.empty:
        return pd.DataFrame(columns=["date", "forecast", "hi", "lo"])

    last_date = pd.to_datetime(df["date"].max())
    last_val = float(df.loc[df["date"] == last_date, "revenue"].iloc[-1])

    mean_q, std_q = estimate_growth_q(df["revenue"], lookback_quarters=lookback_quarters)
    uplift_q = (1.0 + uplift_annual) ** (1.0 / 4.0) - 1.0
    q_growth = mean_q + uplift_q

    steps = years * 4
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=steps, freq="Q")
    future_dates = pd.to_datetime(pd.Series(future_dates).apply(snap_to_quarter_end))

    fc, hi, lo = [], [], []
    cur = last_val
    for i in range(1, steps + 1):
        cur = cur * (1.0 + q_growth)
        fc.append(cur)
        band = (std_q if std_q > 0 else 0.05) * np.sqrt(i)
        hi.append(cur * (1.0 + band))
        lo.append(cur * (1.0 - band))

    return pd.DataFrame({"date": future_dates, "forecast": fc, "hi": hi, "lo": lo})


# =============================
# YAHOO INCOME STATEMENT (QUARTERLY, DATES SNAPPED)
# =============================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_yahoo_income_quarterly() -> pd.DataFrame:
    r = requests.get(YAHOO_FIN_URL, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    html = r.text

    m = re.search(r"root\.App\.main\s*=\s*(\{.*?\});\s*\n", html, flags=re.DOTALL)
    if not m:
        raise ValueError("Could not find Yahoo embedded JSON (root.App.main).")

    data = json.loads(m.group(1))
    stores = data.get("context", {}).get("dispatcher", {}).get("stores", {})
    qss = stores.get("QuoteSummaryStore", {})

    inc_q = qss.get("incomeStatementHistoryQuarterly", {}).get("incomeStatementHistory", [])
    if not inc_q:
        raise ValueError("Yahoo quarterly income statement data not found.")

    rows = []
    for entry in inc_q:
        end = entry.get("endDate", {}).get("raw", None)
        if end is None:
            continue

        dt = pd.to_datetime(int(end), unit="s", errors="coerce")
        dt = snap_to_quarter_end(dt)

        for k, v in entry.items():
            if k in ["maxAge", "endDate"]:
                continue
            if isinstance(v, dict) and "raw" in v:
                val = v.get("raw", None)
                if val is None:
                    continue
                rows.append({"date": dt, "metric": k, "value": float(val)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Yahoo income statement parsed but produced no rows.")

    pretty = {
        "totalRevenue": "Total Revenue",
        "costOfRevenue": "Cost of Revenue",
        "grossProfit": "Gross Profit",
        "researchDevelopment": "Research & Development",
        "sellingGeneralAdministrative": "Selling, General & Admin",
        "totalOperatingExpenses": "Operating Expenses",
        "operatingIncome": "Operating Income",
        "interestExpense": "Interest Expense",
        "interestIncome": "Interest & Investment Income",
        "incomeBeforeTax": "EBT (Income Before Tax)",
        "incomeTaxExpense": "Income Tax Expense",
        "netIncome": "Net Income",
        "ebit": "EBIT",
        "ebitda": "EBITDA",
    }
    df["metric"] = df["metric"].map(lambda x: pretty.get(x, x))
    df = df.sort_values(["metric", "date"]).reset_index(drop=True)
    return df


# =============================
# HEADER
# =============================
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
        Data sources: **StockAnalysis** (segment revenue) + **Yahoo Finance** (income statement).
        """
    )

st.markdown("---")


# =============================
# SIDEBAR
# =============================
st.sidebar.title("Controls")
if st.sidebar.button("Force Refresh (ignore cache)"):
    st.cache_data.clear()
    st.rerun()

years = st.sidebar.slider("Forecast years", min_value=1, max_value=10, value=3, step=1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario, annual)", min_value=0.00, max_value=0.30, value=0.00, step=0.01)

view_mode = st.sidebar.radio(
    "View mode",
    ["Quarterly", "TTM (matches StockAnalysis chart)"],
    index=0
)


# =============================
# LOAD DATA
# =============================
seg_q = load_revenue_by_segment_quarterly()
seg_active = seg_q if view_mode == "Quarterly" else compute_ttm_from_quarterly(seg_q)

products = sorted(seg_active["product"].unique().tolist())
default_product = "Advertising" if "Advertising" in products else products[0]
product = st.sidebar.selectbox("Product", products, index=products.index(default_product))


# =============================
# TABS
# =============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Segment Forecast", "Total Forecast", "Segment Table Check", "Income Statement", "Download"]
)

# TAB 1
with tab1:
    seg = seg_active[seg_active["product"] == product].copy().sort_values("date")
    fc = forecast_series(seg[["date", "revenue"]], years=years, uplift_annual=uplift)

    end_fc = float(fc["forecast"].iloc[-1]) if not fc.empty else np.nan
    base_fc = forecast_series(seg[["date", "revenue"]], years=years, uplift_annual=0.0)
    base_end = float(base_fc["forecast"].iloc[-1]) if not base_fc.empty else np.nan
    delta = end_fc - base_end if (not np.isnan(end_fc) and not np.isnan(base_end)) else np.nan

    k1, k2, k3 = st.columns(3)
    k1.metric("End Forecast (Scenario)", money_fmt(end_fc))
    k2.metric("Δ vs Baseline", money_fmt(delta))
    k3.metric("Last Reported Quarter", pd.to_datetime(seg["date"].max()).strftime("%b %d, %Y"))

    y_title = "Revenue (USD, Quarterly)" if view_mode == "Quarterly" else "Revenue (USD, TTM)"
    fig = build_plot_lines(
        seg[["date", "revenue"]],
        fc,
        title=f"{product}: Historical + {years}-Year Forecast (uplift {uplift*100:.1f}%)",
        y_title=y_title
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 2 (TOTAL FIXED)
with tab2:
    # Historical TOTAL:
    # - If Quarterly: sum the quarterly segment row (matches Yahoo Total Revenue)
    # - If TTM: sum the TTM series (rolling 4Q)
    if view_mode == "Quarterly":
        total_hist = segment_sum_total_from_quarterly(seg_q)
        y_title = "Revenue (USD, Quarterly)"
        title_mode = "Quarterly"
    else:
        # TTM: build total by summing the already-built TTM segment series
        wide_ttm = seg_active.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
        total_hist = wide_ttm.sum(axis=1, min_count=1).reset_index().rename(columns={0: "revenue"})
        total_hist.columns = ["date", "revenue"]
        y_title = "Revenue (USD, TTM)"
        title_mode = "TTM"

    total_hist = total_hist.sort_values("date").dropna(subset=["revenue"])

    # Forecast TOTAL by forecasting each segment (on the correct underlying mode)
    wide = seg_active.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index().dropna(how="all")

    future_steps = years * 4
    last_date = wide.index.max()
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=future_steps, freq="Q")
    future_dates = pd.to_datetime(pd.Series(future_dates).apply(snap_to_quarter_end))

    total_fc_vals = np.zeros(future_steps, dtype=float)
    total_hi_vals = np.zeros(future_steps, dtype=float)
    total_lo_vals = np.zeros(future_steps, dtype=float)

    for p in wide.columns:
        hist_p = wide[[p]].reset_index().rename(columns={p: "revenue"}).dropna(subset=["revenue"])
        if hist_p.empty:
            continue
        fcp = forecast_series(hist_p[["date", "revenue"]], years=years, uplift_annual=uplift)
        if fcp.empty:
            continue
        total_fc_vals += fcp["forecast"].values
        total_hi_vals += fcp["hi"].values
        total_lo_vals += fcp["lo"].values

    total_fc = pd.DataFrame({"date": future_dates, "forecast": total_fc_vals, "hi": total_hi_vals, "lo": total_lo_vals})

    fig_total = build_plot_lines(
        total_hist[["date", "revenue"]],
        total_fc,
        title=f"Total Alphabet Revenue Forecast ({title_mode}, Scenario)",
        y_title=y_title
    )
    st.plotly_chart(fig_total, use_container_width=True)

    # Sanity check vs Yahoo Total Revenue (quarterly)
    try:
        inc = load_yahoo_income_quarterly()
        yrev = inc[inc["metric"] == "Total Revenue"].sort_values("date")
        if not yrev.empty:
            last_q = total_hist["date"].max()
            seg_total_last = float(total_hist.loc[total_hist["date"] == last_q, "revenue"].iloc[-1])

            ymatch = yrev[yrev["date"] == last_q]
            if not ymatch.empty:
                yahoo_last = float(ymatch["value"].iloc[-1])
                diff = seg_total_last - yahoo_last

                c1, c2, c3 = st.columns(3)
                c1.metric("Segment-sum TOTAL (last quarter)", money_fmt(seg_total_last))
                c2.metric("Yahoo Total Revenue (same quarter)", money_fmt(yahoo_last))
                c3.metric("Difference", money_fmt(diff))

                st.caption("If the difference is near $0, your segment total matches Yahoo Total Revenue for that quarter.")
            else:
                st.caption("Yahoo Total Revenue doesn’t have the exact same quarter-end date loaded yet (still fine).")
    except Exception:
        st.caption("Yahoo sanity check unavailable (scrape blocked or changed).")

# TAB 3
with tab3:
    st.subheader("Segment table (latest quarter sanity check)")
    latest_dt = seg_q["date"].max()
    chk = seg_q[seg_q["date"] == latest_dt].copy().sort_values("revenue", ascending=False)
    chk["revenue_fmt"] = chk["revenue"].apply(money_fmt)
    st.write(f"Latest quarter in StockAnalysis table: **{pd.to_datetime(latest_dt).strftime('%b %d, %Y')}**")
    st.dataframe(chk[["product", "revenue_fmt"]], use_container_width=True)

# TAB 4
with tab4:
    st.subheader("Income Statement (Yahoo Finance: GOOGL)")
    inc = load_yahoo_income_quarterly()
    metric_list = sorted(inc["metric"].unique().tolist())
    default_metric = "Total Revenue" if "Total Revenue" in metric_list else metric_list[0]
    metric = st.selectbox("Metric", metric_list, index=metric_list.index(default_metric))

    fig_inc = build_bar_income(inc, metric)
    st.plotly_chart(fig_inc, use_container_width=True)

    mdf = inc[inc["metric"] == metric].sort_values("date", ascending=False).copy()
    mdf["Period Ending"] = pd.to_datetime(mdf["date"]).dt.strftime("%b %d, %Y")
    mdf["Value"] = mdf["value"].apply(money_fmt)
    st.dataframe(mdf[["Period Ending", "Value"]], use_container_width=True)

# TAB 5
with tab5:
    st.subheader("Download data")

    seg_download = seg_active.copy()
    seg_download["revenue"] = seg_download["revenue"].astype(float)

    st.download_button(
        label="Download segment dataset (tidy) as CSV",
        data=seg_download.to_csv(index=False).encode("utf-8"),
        file_name=("alphabet_segments_quarterly.csv" if view_mode == "Quarterly" else "alphabet_segments_ttm.csv"),
        mime="text/csv",
    )

    wide_out = seg_active.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_out.reset_index(inplace=True)

    st.download_button(
        label="Download segment dataset (wide) as CSV",
        data=wide_out.to_csv(index=False).encode("utf-8"),
        file_name=("alphabet_segments_wide_quarterly.csv" if view_mode == "Quarterly" else "alphabet_segments_wide_ttm.csv"),
        mime="text/csv",
    )

    st.download_button(
        label="Download Yahoo income statement (quarterly) as CSV",
        data=inc.to_csv(index=False).encode("utf-8"),
        file_name="yahoo_income_statement_quarterly.csv",
        mime="text/csv",
    )
