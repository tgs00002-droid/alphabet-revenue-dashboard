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
    """
    IMPORTANT FIX:
    Yahoo sometimes hands back month-end dates that are NOT true quarter ends.
    We snap to the next quarter end (QuarterEnd(0) => quarter-end on/after dt).
    Example: Nov 30 -> Dec 31, Jan 31 -> Mar 31, May 31 -> Jun 30, etc.
    """
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
        xaxis_title="Quarter",
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
    fig.add_trace(go.Bar(
        x=d["date"],
        y=d["value"],
        name=metric
    ))

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
# STOCKANALYSIS SEGMENT SCRAPE
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
        header_cells = thead.find_all(["th", "td"])
        headers = [c.get_text(" ", strip=True) for c in header_cells]
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
    df = tidy_q.copy()
    df = df.sort_values(["product", "date"])
    df["revenue_ttm"] = df.groupby("product")["revenue"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    out = df.dropna(subset=["revenue_ttm"]).rename(columns={"revenue_ttm": "revenue"})
    return out[["date", "product", "revenue"]].reset_index(drop=True)


def pick_total_components(cols: List[str]) -> List[str]:
    preferred_leaf = [
        "Google Search & Other",
        "YouTube Ads",
        "Google Network",
        "Google Cloud",
        "Google Subscriptions, Platforms & Devices",
        "Other Bets",
        "Hedging Gains",
    ]
    available_leaf = [c for c in preferred_leaf if c in cols]

    ad_components = ["Google Search & Other", "YouTube Ads", "Google Network"]
    has_components = all(c in cols for c in ad_components)

    comps = available_leaf if available_leaf else cols.copy()

    if has_components and "Advertising" in comps:
        comps = [c for c in comps if c != "Advertising"]

    return comps


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
# YAHOO INCOME STATEMENT SCRAPE (FIXED JSON PARSE + FIXED DATES)
# =============================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_yahoo_income_quarterly() -> pd.DataFrame:
    def extract_root_app_main_json(html: str) -> dict:
        marker = "root.App.main ="
        i = html.find(marker)
        if i == -1:
            raise ValueError("Could not find 'root.App.main =' in Yahoo HTML.")

        j = html.find("{", i)
        if j == -1:
            raise ValueError("Could not find opening '{' for Yahoo embedded JSON.")

        depth = 0
        in_str = False
        esc = False
        for k in range(j, len(html)):
            ch = html[k]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_text = html[j:k+1]
                        return json.loads(json_text)

        raise ValueError("Failed to extract Yahoo embedded JSON by brace-matching.")

    r = requests.get(YAHOO_FIN_URL, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    html = r.text

    data = extract_root_app_main_json(html)

    stores = data.get("context", {}).get("dispatcher", {}).get("stores", {})
    qss = stores.get("QuoteSummaryStore", {})

    inc_q = qss.get("incomeStatementHistoryQuarterly", {}).get("incomeStatementHistory", [])
    if not inc_q:
        raise ValueError("Yahoo quarterly income statement data not found (incomeStatementHistoryQuarterly empty).")

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
# SIDEBAR CONTROLS
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


# TAB 2
with tab2:
    wide = seg_active.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide = wide.dropna(how="all")

    cols = list(wide.columns)
    total_components = pick_total_components(cols)

    wide["TOTAL"] = wide[total_components].sum(axis=1, min_count=1)
    total_hist = wide["TOTAL"].reset_index().rename(columns={"TOTAL": "revenue"})

    future_steps = years * 4
    last_date = wide.index.max()
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=future_steps, freq="Q")
    future_dates = pd.to_datetime(pd.Series(future_dates).apply(snap_to_quarter_end))

    total_fc_vals = np.zeros(future_steps, dtype=float)
    total_hi_vals = np.zeros(future_steps, dtype=float)
    total_lo_vals = np.zeros(future_steps, dtype=float)

    for p in total_components:
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

    title_mode = "Quarterly" if view_mode == "Quarterly" else "TTM"
    y_title = "Revenue (USD, Quarterly)" if view_mode == "Quarterly" else "Revenue (USD, TTM)"

    fig_total = build_plot_lines(
        total_hist[["date", "revenue"]],
        total_fc,
        title=f"Total Alphabet Revenue Forecast ({title_mode}, Scenario)",
        y_title=y_title
    )
    st.plotly_chart(fig_total, use_container_width=True)
    st.caption("Total uses non-overlapping segment components to avoid double counting.")


# TAB 3
with tab3:
    st.subheader("Segment table (latest quarter sanity check)")
    latest_dt = seg_q["date"].max()
    chk = seg_q[seg_q["date"] == latest_dt].copy().sort_values("revenue", ascending=False)
    chk["revenue_fmt"] = chk["revenue"].apply(money_fmt)
    st.write(f"Latest quarter in StockAnalysis table: **{pd.to_datetime(latest_dt).strftime('%b %d, %Y')}**")
    st.dataframe(chk[["product", "revenue_fmt"]], use_container_width=True)


# TAB 4 (INCOME STATEMENT FIXED)
# =============================
# YAHOO INCOME STATEMENT SCRAPE (CLOUD-SAFE: Yahoo -> fallback to StockAnalysis)
# =============================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_yahoo_income_quarterly() -> pd.DataFrame:
    """
    Streamlit Cloud often gets blocked by Yahoo Finance (403/429).
    This function:
      1) Tries Yahoo first (same data you wanted)
      2) If Yahoo fails, falls back to StockAnalysis quarterly income statement table.
    Output schema stays the same: columns = [date, metric, value]
    """

    # ---------- helper: pretty metric names ----------
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

    # ---------- 1) TRY YAHOO ----------
    try:
        def extract_root_app_main_json(html: str) -> dict:
            marker = "root.App.main ="
            i = html.find(marker)
            if i == -1:
                raise ValueError("Could not find 'root.App.main =' in Yahoo HTML.")

            j = html.find("{", i)
            if j == -1:
                raise ValueError("Could not find opening '{' for Yahoo embedded JSON.")

            depth = 0
            in_str = False
            esc = False
            for k in range(j, len(html)):
                ch = html[k]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            json_text = html[j:k+1]
                            return json.loads(json_text)

            raise ValueError("Failed to extract Yahoo embedded JSON by brace-matching.")

        r = requests.get(YAHOO_FIN_URL, headers=UA_HEADERS, timeout=30)
        r.raise_for_status()
        html = r.text

        data = extract_root_app_main_json(html)
        stores = data.get("context", {}).get("dispatcher", {}).get("stores", {})
        qss = stores.get("QuoteSummaryStore", {})

        inc_q = qss.get("incomeStatementHistoryQuarterly", {}).get("incomeStatementHistory", [])
        if not inc_q:
            raise ValueError("Yahoo quarterly income statement data not found (empty store).")

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
                    rows.append({"date": dt, "metric": pretty.get(k, k), "value": float(val)})

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Yahoo income statement parsed but produced no rows.")

        return df.sort_values(["metric", "date"]).reset_index(drop=True)

    except Exception:
        # ---------- 2) FALLBACK: STOCKANALYSIS QUARTERLY FINANCIALS ----------
        # This is what makes it work on Streamlit Cloud.
        STOCKANALYSIS_FIN_Q = "https://stockanalysis.com/stocks/goog/financials/?p=quarterly"

        r2 = requests.get(STOCKANALYSIS_FIN_Q, headers=UA_HEADERS, timeout=30)
        r2.raise_for_status()

        # Use pandas read_html for stable parsing
        tables = pd.read_html(r2.text)
        if not tables:
            raise ValueError("No tables found on StockAnalysis financials page.")

        # Income statement is typically the first table
        t = tables[0].copy()

        # First column is the row label (metric name)
        first_col = t.columns[0]
        t = t.rename(columns={first_col: "Metric"}).copy()

        # Columns after Metric are dates like "Sep 2025", "Jun 2025", etc.
        date_cols = [c for c in t.columns if c != "Metric"]
        if not date_cols:
            raise ValueError("StockAnalysis income statement table has no date columns.")

        # Map StockAnalysis row names -> your app metric names
        # (We keep your labels so the dropdown looks clean)
        row_map = {
            "Revenue": "Total Revenue",
            "Cost of Revenue": "Cost of Revenue",
            "Gross Profit": "Gross Profit",
            "Research & Development": "Research & Development",
            "Selling, General & Admin": "Selling, General & Admin",
            "Operating Expenses": "Operating Expenses",
            "Operating Income": "Operating Income",
            "Pretax Income": "EBT (Income Before Tax)",
            "Income Tax": "Income Tax Expense",
            "Net Income": "Net Income",
            "EBITDA": "EBITDA",
            "EBIT": "EBIT",
        }

        # Build tidy output
        out_rows = []
        for _, row in t.iterrows():
            raw_name = str(row["Metric"]).strip()
            if raw_name not in row_map:
                continue
            nice_name = row_map[raw_name]

            for dc in date_cols:
                date_str = str(dc).strip()
                # Parse "Sep 2025" -> quarter end
                dt = pd.to_datetime(date_str, errors="coerce")
                if pd.isna(dt):
                    # Sometimes headers are like "Sep '25"
                    try:
                        dt = pd.to_datetime(date_str.replace("'", ""), format="%b %y", errors="coerce")
                    except Exception:
                        dt = pd.NaT
                if pd.isna(dt):
                    continue

                dt = snap_to_quarter_end(dt)

                v = row.get(dc, None)
                val = money_to_float(v)

                # StockAnalysis financials are usually shown in millions for big companies
                # Example: 102,354 -> $102.354B
                if not np.isnan(val):
                    val = float(val) * 1e6

                if np.isnan(val):
                    continue

                out_rows.append({"date": dt, "metric": nice_name, "value": float(val)})

        df2 = pd.DataFrame(out_rows)
        if df2.empty:
            raise ValueError("StockAnalysis fallback worked but no mapped income statement rows were found.")

        return df2.sort_values(["metric", "date"]).reset_index(drop=True)



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

