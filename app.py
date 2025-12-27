import re
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import plotly.graph_objects as go
import streamlit as st

# Income statement (Yahoo Finance via yfinance)
import yfinance as yf


# -----------------------------
# CONFIG
# -----------------------------
SEGMENT_URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
UA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
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
    Convert strings like '56.57B', '344.00M', '-207.00M', '0' -> float USD.
    """
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan

    s = s.replace(",", "").replace("−", "-").replace("\xa0", " ").strip()

    # quick numeric
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


def find_best_history_table(soup: BeautifulSoup):
    """
    StockAnalysis pages can change. We pick the table that:
    - Has a 'Date' first column (header)
    - Has key segment columns like 'Google Search' / 'YouTube' / 'Cloud'
    - Has many rows
    """
    tables = soup.find_all("table")
    if not tables:
        return None

    def table_score(tbl):
        ths = [th.get_text(" ", strip=True) for th in tbl.find_all("th")]
        header_text = " | ".join(ths).lower()
        score = 0

        # must have date-ish
        if "date" in header_text:
            score += 5

        # segment keywords
        keywords = ["google search", "youtube", "cloud", "network", "other bets", "subscriptions"]
        score += sum(3 for k in keywords if k in header_text)

        # row count
        score += min(len(tbl.find_all("tr")), 200) / 10.0
        return score

    ranked = sorted(tables, key=table_score, reverse=True)
    best = ranked[0]
    # sanity: if it doesn't even mention date, return None
    ths = [th.get_text(" ", strip=True).lower() for th in best.find_all("th")]
    if not any("date" == t or t.startswith("date") for t in ths):
        return None
    return best


def parse_table_to_wide(table: BeautifulSoup) -> pd.DataFrame:
    # headers
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

    # rows
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
    wide = wide.rename(columns={"Date": "date"})
    wide["date"] = pd.to_datetime(wide["date"], errors="coerce")

    for c in wide.columns:
        if c != "date":
            wide[c] = wide[c].apply(money_to_float)

    wide = wide.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return wide


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_segment_quarterly() -> pd.DataFrame:
    """
    Scrape the quarterly segment 'History' table from StockAnalysis.
    Returns tidy: date, product, revenue_quarterly.
    """
    r = requests.get(SEGMENT_URL, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    table = find_best_history_table(soup)
    if table is None:
        raise ValueError("Could not find a usable History table on StockAnalysis (layout change or blocking).")

    wide = parse_table_to_wide(table)

    # sanity check: quarterly 'Google Search & Other' should be around ~40B-70B recently, not ~200B+
    # If the latest is huge, you likely scraped TTM view. We can still proceed, but your TTM chart would then be wrong.
    # We warn and still return it.
    check_cols = [c for c in wide.columns if c != "date"]
    if check_cols:
        recent_row = wide.iloc[-1]
        vals = [recent_row[c] for c in check_cols if pd.notna(recent_row[c])]
        if vals:
            # if most values are > 100B, it's probably TTM
            big_ratio = np.mean([v > 1e11 for v in vals])  # > 100B
            if big_ratio > 0.5:
                st.warning(
                    "It looks like StockAnalysis returned TTM-style values in the History table. "
                    "This app expects the Quarterly History table. "
                    "If your TTM chart doesn't match, try Force Refresh or run locally."
                )

    tidy = wide.melt("date", var_name="product", value_name="revenue_q").dropna()
    tidy = tidy.sort_values(["product", "date"]).reset_index(drop=True)
    return tidy


def add_ttm_from_quarterly(tidy_q: pd.DataFrame) -> pd.DataFrame:
    """
    Adds revenue_ttm computed as rolling 4-quarter sum per product.
    """
    df = tidy_q.copy()
    df = df.sort_values(["product", "date"])
    df["revenue_ttm"] = (
        df.groupby("product")["revenue_q"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return df


def estimate_q_growth(series: pd.Series, lookback_quarters: int = 8) -> tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) < 6:
        return 0.02, 0.05
    s = s.iloc[-lookback_quarters:] if len(s) > lookback_quarters else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 3:
        return 0.02, 0.05
    std = float(g.std(ddof=1)) if len(g) > 1 else 0.05
    return float(g.mean()), std


def forecast_quarterly(hist_q: pd.DataFrame, years: int, uplift_annual: float, lookback_quarters: int = 8) -> pd.DataFrame:
    """
    Forecast future quarterly values. uplift_annual is extra annual CAGR added to growth.
    """
    df = hist_q.sort_values("date").dropna(subset=["revenue_q"]).copy()
    last_date = df["date"].max()
    last_val = float(df.loc[df["date"] == last_date, "revenue_q"].iloc[-1])

    mean_q, std_q = estimate_q_growth(df["revenue_q"], lookback_quarters=lookback_quarters)

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

    return pd.DataFrame({"date": future_dates, "forecast_q": fc, "hi_q": hi, "lo_q": lo})


def to_ttm_view(hist_q: pd.DataFrame, fc_q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert quarterly history+forecast into TTM series:
    - For history: rolling4 sum
    - For forecast: compute rolling on concatenated series, then slice the forecast portion
    """
    h = hist_q.sort_values("date").copy()
    f = fc_q.sort_values("date").copy()

    combined = pd.concat(
        [h[["date", "revenue_q"]].rename(columns={"revenue_q": "q"}),
         f[["date", "forecast_q"]].rename(columns={"forecast_q": "q"})],
        ignore_index=True
    ).sort_values("date").reset_index(drop=True)

    combined["ttm"] = combined["q"].rolling(4, min_periods=4).sum()

    # history ttm (aligned to history dates)
    hist_ttm = combined[combined["date"].isin(h["date"])][["date", "ttm"]].rename(columns={"ttm": "revenue_ttm"})
    hist_ttm = hist_ttm.dropna(subset=["revenue_ttm"])

    # forecast ttm (aligned to forecast dates)
    fc_ttm = combined[combined["date"].isin(f["date"])][["date", "ttm"]].rename(columns={"ttm": "forecast_ttm"})

    # uncertainty bands for TTM: roll the quarterly hi/lo as well (approx)
    combined_hi = pd.concat(
        [h[["date"]].assign(v=h["revenue_q"]),
         f[["date"]].assign(v=f["hi_q"])],
        ignore_index=True
    ).sort_values("date").reset_index(drop=True)
    combined_lo = pd.concat(
        [h[["date"]].assign(v=h["revenue_q"]),
         f[["date"]].assign(v=f["lo_q"])],
        ignore_index=True
    ).sort_values("date").reset_index(drop=True)

    hi_ttm = combined_hi["v"].rolling(4, min_periods=4).sum()
    lo_ttm = combined_lo["v"].rolling(4, min_periods=4).sum()

    combined_bands = combined[["date"]].copy()
    combined_bands["hi_ttm"] = hi_ttm
    combined_bands["lo_ttm"] = lo_ttm

    fc_ttm = fc_ttm.merge(combined_bands, on="date", how="left")
    fc_ttm = fc_ttm.dropna(subset=["forecast_ttm"])

    return hist_ttm, fc_ttm


def build_plot(hist_x: pd.DataFrame, fc_x: pd.DataFrame, title: str, y_label: str,
               hist_col: str, fc_col: str, hi_col: str, lo_col: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_x["date"], y=hist_x[hist_col],
        mode="lines", name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=fc_x["date"], y=fc_x[hi_col],
        mode="lines", name="80% high (approx)",
        line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=fc_x["date"], y=fc_x[lo_col],
        mode="lines", name="80% low (approx)",
        line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=fc_x["date"], y=fc_x[fc_col],
        mode="lines", name="Forecast (Scenario)"
    ))

    fig.update_layout(
        title=title,
        height=520,
        xaxis_title="Quarter",
        yaxis_title=y_label,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_income_statement_yahoo(ticker: str = "GOOGL") -> pd.DataFrame:
    """
    Load quarterly income statement from Yahoo via yfinance.
    Returns tidy: date, metric, value
    """
    t = yf.Ticker(ticker)

    # Different yfinance versions expose different APIs — try several.
    stmt = None
    for attempt in ["quarterly_income_stmt", "quarterly_financials"]:
        try:
            stmt = getattr(t, attempt)
            if stmt is not None and isinstance(stmt, pd.DataFrame) and stmt.shape[1] > 0:
                break
        except Exception:
            stmt = None

    # yfinance newer method
    if stmt is None or not isinstance(stmt, pd.DataFrame) or stmt.shape[1] == 0:
        try:
            stmt = t.get_income_stmt(freq="quarterly")
        except Exception:
            stmt = None

    if stmt is None or not isinstance(stmt, pd.DataFrame) or stmt.shape[1] == 0:
        raise ValueError("Could not load income statement from Yahoo via yfinance.")

    # stmt index=metrics, columns=period end dates
    stmt = stmt.copy()
    # make sure columns are datetimes
    stmt.columns = pd.to_datetime(stmt.columns, errors="coerce")
    stmt = stmt.loc[:, stmt.columns.notna()]
    stmt = stmt.sort_index()
    stmt = stmt.sort_index(axis=1)

    tidy = (
        stmt.reset_index()
        .melt(id_vars=["index"], var_name="date", value_name="value")
        .rename(columns={"index": "metric"})
    )
    tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce")
    tidy = tidy.dropna(subset=["date"])
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.dropna(subset=["value"])
    return tidy


# -----------------------------
# UI
# -----------------------------
st.sidebar.title("Controls")
force = st.sidebar.button("Force Refresh (ignore cache)")

years = st.sidebar.slider("Forecast years", min_value=1, max_value=10, value=3, step=1)
uplift = st.sidebar.slider("Extra CAGR uplift (scenario, annual)", min_value=0.00, max_value=0.30, value=0.00, step=0.01)

view_mode = st.sidebar.radio(
    "View mode",
    ["Quarterly", "TTM (matches StockAnalysis chart)"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Notes")
st.sidebar.write("Segment revenue is scraped from StockAnalysis (History table).")
st.sidebar.write("TTM is computed as rolling 4-quarter sum (this matches their TTM charts).")
st.sidebar.write("Income Statement comes from Yahoo Finance via yfinance (more reliable than scraping Yahoo HTML).")

# Force refresh cache
if force:
    load_segment_quarterly.clear()
    load_income_statement_yahoo.clear()

# Load data
try:
    seg_q = load_segment_quarterly()
except Exception as e:
    st.error(f"Segment scrape failed: {e}")
    st.stop()

seg_q_ttm = add_ttm_from_quarterly(seg_q)

products = sorted(seg_q["product"].unique().tolist())
default_product = "Advertising" if "Advertising" in products else products[0]
product = st.sidebar.selectbox("Product", products, index=products.index(default_product))

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
        Interactive segment-level **quarterly + TTM** analysis and scenario forecasting.  
        Data source: StockAnalysis (segment revenue) + Yahoo (income statement via yfinance).
        """
    )

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Segment Forecast", "Total Forecast", "Segment Table Check", "Income Statement", "Download"]
)

# -----------------------------
# TAB 1: Segment Forecast
# -----------------------------
with tab1:
    seg_one = seg_q[seg_q["product"] == product].sort_values("date").copy()
    hist_q = seg_one.rename(columns={"revenue_q": "revenue_q"})[["date", "revenue_q"]]

    fc_q = forecast_quarterly(hist_q, years=years, uplift_annual=uplift)

    if view_mode.startswith("TTM"):
        hist_ttm, fc_ttm = to_ttm_view(hist_q, fc_q)

        end_fc = float(fc_ttm["forecast_ttm"].iloc[-1])
        base_fc = to_ttm_view(hist_q, forecast_quarterly(hist_q, years=years, uplift_annual=0.0))[1]["forecast_ttm"].iloc[-1]
        delta = float(end_fc - base_fc)

        k1, k2, k3 = st.columns(3)
        k1.metric("End Forecast (Scenario)", money_fmt(end_fc))
        k2.metric("Δ vs Baseline", money_fmt(delta))
        k3.metric("Last Reported Quarter", hist_q["date"].max().strftime("%b %d, %Y"))

        fig = build_plot(
            hist_x=hist_ttm, fc_x=fc_ttm,
            title=f"{product}: TTM Historical + {years}-Year Forecast (uplift {uplift*100:.1f}%)",
            y_label="Revenue (USD, TTM)",
            hist_col="revenue_ttm",
            fc_col="forecast_ttm",
            hi_col="hi_ttm",
            lo_col="lo_ttm",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        end_fc = float(fc_q["forecast_q"].iloc[-1])
        base_fc = forecast_quarterly(hist_q, years=years, uplift_annual=0.0)["forecast_q"].iloc[-1]
        delta = float(end_fc - base_fc)

        k1, k2, k3 = st.columns(3)
        k1.metric("End Forecast (Scenario)", money_fmt(end_fc))
        k2.metric("Δ vs Baseline", money_fmt(delta))
        k3.metric("Last Reported Quarter", hist_q["date"].max().strftime("%b %d, %Y"))

        # plot in quarterly units
        fc_x = fc_q.rename(columns={"forecast_q": "forecast", "hi_q": "hi", "lo_q": "lo"})
        hist_x = hist_q.rename(columns={"revenue_q": "revenue"})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_x["date"], y=hist_x["revenue"], mode="lines", name="Historical"))
        fig.add_trace(go.Scatter(x=fc_x["date"], y=fc_x["hi"], mode="lines", name="80% high (approx)", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=fc_x["date"], y=fc_x["lo"], mode="lines", name="80% low (approx)", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=fc_x["date"], y=fc_x["forecast"], mode="lines", name="Forecast (Scenario)"))
        fig.update_layout(
            title=f"{product}: Quarterly Historical + {years}-Year Forecast (uplift {uplift*100:.1f}%)",
            height=520,
            xaxis_title="Quarter",
            yaxis_title="Revenue (USD, Quarterly)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 2: Total Forecast
# -----------------------------
with tab2:
    wide_q = (
        seg_q.pivot_table(index="date", columns="product", values="revenue_q", aggfunc="sum")
        .sort_index()
        .dropna(how="all")
    )

    wide_q["TOTAL_Q"] = wide_q.sum(axis=1)
    total_hist_q = wide_q["TOTAL_Q"].reset_index().rename(columns={"TOTAL_Q": "revenue_q"})

    # forecast each product quarterly then sum
    future_steps = years * 4
    last_date = wide_q.index.max()
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=future_steps, freq="Q")

    total_fc_q = np.zeros(future_steps, dtype=float)
    total_hi_q = np.zeros(future_steps, dtype=float)
    total_lo_q = np.zeros(future_steps, dtype=float)

    for p in wide_q.columns:
        if p == "TOTAL_Q":
            continue
        hist_p = wide_q[[p]].reset_index().rename(columns={p: "revenue_q"})
        fcp = forecast_quarterly(hist_p[["date", "revenue_q"]], years=years, uplift_annual=uplift)
        total_fc_q += fcp["forecast_q"].values
        total_hi_q += fcp["hi_q"].values
        total_lo_q += fcp["lo_q"].values

    total_future_q = pd.DataFrame({"date": future_dates, "forecast_q": total_fc_q, "hi_q": total_hi_q, "lo_q": total_lo_q})

    if view_mode.startswith("TTM"):
        hist_ttm, fc_ttm = to_ttm_view(total_hist_q[["date", "revenue_q"]], total_future_q)
        fig_total = build_plot(
            hist_x=hist_ttm, fc_x=fc_ttm,
            title="Total Alphabet Revenue Forecast (TTM, Scenario)",
            y_label="Revenue (USD, TTM)",
            hist_col="revenue_ttm",
            fc_col="forecast_ttm",
            hi_col="hi_ttm",
            lo_col="lo_ttm",
        )
        st.plotly_chart(fig_total, use_container_width=True)
        st.caption("This total is built by forecasting each segment quarterly, summing, then converting to TTM (rolling 4Q).")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=total_hist_q["date"], y=total_hist_q["revenue_q"], mode="lines", name="Historical"))
        fig.add_trace(go.Scatter(x=total_future_q["date"], y=total_future_q["hi_q"], mode="lines", name="80% high (approx)", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=total_future_q["date"], y=total_future_q["lo_q"], mode="lines", name="80% low (approx)", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=total_future_q["date"], y=total_future_q["forecast_q"], mode="lines", name="Forecast (Scenario)"))
        fig.update_layout(
            title="Total Alphabet Revenue Forecast (Quarterly, Scenario)",
            height=520,
            xaxis_title="Quarter",
            yaxis_title="Revenue (USD, Quarterly)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This total is the sum of per-segment quarterly scenario forecasts (not a separately fit model).")

# -----------------------------
# TAB 3: Segment Table Check
# -----------------------------
with tab3:
    st.subheader("Segment History Table (Sanity Check)")
    st.write("This is the scraped quarterly History table, then we compute TTM as rolling 4-quarter sum.")

    latest_date = seg_q["date"].max()
    show_latest = seg_q[seg_q["date"] == latest_date].sort_values("revenue_q", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Latest Quarter (Quarterly)**")
        st.dataframe(
            show_latest[["date", "product", "revenue_q"]].assign(revenue_q=lambda d: d["revenue_q"].map(money_fmt)),
            use_container_width=True,
            hide_index=True
        )

    with c2:
        st.markdown("**Latest Quarter (TTM computed)**")
        latest_ttm = seg_q_ttm[(seg_q_ttm["date"] == latest_date)].dropna(subset=["revenue_ttm"])
        latest_ttm = latest_ttm.sort_values("revenue_ttm", ascending=False)
        st.dataframe(
            latest_ttm[["date", "product", "revenue_ttm"]].assign(revenue_ttm=lambda d: d["revenue_ttm"].map(money_fmt)),
            use_container_width=True,
            hide_index=True
        )

# -----------------------------
# TAB 4: Income Statement (Yahoo via yfinance)
# -----------------------------
with tab4:
    st.subheader("Income Statement (Yahoo Finance via yfinance)")

    try:
        inc = load_income_statement_yahoo("GOOGL")
    except Exception as e:
        st.error(f"Income statement load failed: {e}")
        st.stop()

    metrics = sorted(inc["metric"].unique().tolist())
    default_metric = "Operating Income" if "Operating Income" in metrics else metrics[0]
    metric = st.selectbox("Metric", metrics, index=metrics.index(default_metric))

    inc_one = inc[inc["metric"] == metric].sort_values("date").copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=inc_one["date"], y=inc_one["value"], name=metric))
    fig.update_layout(
        title=f"Alphabet Income Statement: {metric}",
        height=520,
        xaxis_title="Quarter",
        yaxis_title=f"{metric} (USD)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Source: Yahoo Finance (yfinance). Values are as-provided by Yahoo; units are USD.")

# -----------------------------
# TAB 5: Download
# -----------------------------
with tab5:
    st.subheader("Download data")

    out_q = seg_q.copy()
    out_q["revenue_q"] = out_q["revenue_q"].astype(float)
    st.download_button(
        label="Download segment revenue (Quarterly, tidy) CSV",
        data=out_q.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_segment_revenue_quarterly_tidy.csv",
        mime="text/csv",
    )

    out_ttm = seg_q_ttm.dropna(subset=["revenue_ttm"]).copy()
    out_ttm["revenue_ttm"] = out_ttm["revenue_ttm"].astype(float)
    st.download_button(
        label="Download segment revenue (TTM computed, tidy) CSV",
        data=out_ttm.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_segment_revenue_ttm_tidy.csv",
        mime="text/csv",
    )

    wide_q = seg_q.pivot_table(index="date", columns="product", values="revenue_q", aggfunc="sum").sort_index()
    wide_q.reset_index(inplace=True)
    st.download_button(
        label="Download segment revenue (Quarterly, wide) CSV",
        data=wide_q.to_csv(index=False).encode("utf-8"),
        file_name="alphabet_segment_revenue_quarterly_wide.csv",
        mime="text/csv",
    )

    st.caption("Segment source: StockAnalysis → GOOG → Metrics → Revenue by Segment (History table).")
