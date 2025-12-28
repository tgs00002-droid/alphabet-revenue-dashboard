import re
import json
from typing import Tuple, List
import time

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
FINANCIALS_URL = "https://stockanalysis.com/stocks/goog/financials/"

# Rotate user agents to avoid blocking
UA_HEADERS_LIST = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
]

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
        mode="lines+markers", name="Historical",
        line=dict(width=3), marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["hi"],
        mode="lines", name="Upper bound",
        line=dict(dash="dot", color="rgba(100,100,100,0.3)")
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["lo"],
        mode="lines", name="Lower bound",
        line=dict(dash="dot", color="rgba(100,100,100,0.3)"),
        fill='tonexty', fillcolor='rgba(100,100,100,0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(width=2, dash="dash"), marker=dict(size=5)
    ))

    ticks = quarter_tickvals(pd.concat([hist["date"], fc["date"]], ignore_index=True))
    fig.update_layout(
        title=title,
        height=520,
        xaxis_title="Quarter (Period Ending)",
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        tickformat="%b %Y",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)'
    )
    fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
    return fig


# =============================
# STOCKANALYSIS SEGMENT SCRAPE (QUARTERLY)
# =============================
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_revenue_by_segment_quarterly() -> pd.DataFrame:
    """Scrape quarterly segment revenue from StockAnalysis"""
    for attempt, headers in enumerate(UA_HEADERS_LIST, 1):
        try:
            time.sleep(0.5)
            r = requests.get(SEGMENT_URL, headers=headers, timeout=30)
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, "lxml")

            # Find the History section
            history_h = None
            for h in soup.find_all(["h2", "h3"]):
                if "history" in h.get_text(strip=True).lower():
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
            
            # Validation
            if len(tidy) < 10:
                raise ValueError(f"Too few rows parsed: {len(tidy)}")
            
            return tidy
            
        except Exception as e:
            if attempt == len(UA_HEADERS_LIST):
                st.error(f"Failed to load StockAnalysis data after {attempt} attempts: {str(e)}")
                raise
            continue
    
    raise ValueError("Failed to load segment data")


def segment_sum_total_from_quarterly(seg_q: pd.DataFrame) -> pd.DataFrame:
    """Build TOTAL from the quarterly segment row sums"""
    wide_q = seg_q.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_q = wide_q.dropna(how="all")
    total = wide_q.sum(axis=1, min_count=1)
    out = total.reset_index().rename(columns={0: "revenue"})
    out.columns = ["date", "revenue"]
    return out


# =============================
# STOCKANALYSIS FINANCIALS SCRAPE
# =============================
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_total_revenue_from_financials() -> pd.DataFrame:
    """Scrape total revenue from StockAnalysis financials page (quarterly)"""
    for attempt, headers in enumerate(UA_HEADERS_LIST, 1):
        try:
            time.sleep(0.5)
            r = requests.get(FINANCIALS_URL + "?p=quarterly", headers=headers, timeout=30)
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, "lxml")
            
            # Find the main financial table
            tables = soup.find_all("table")
            if not tables:
                raise ValueError("No tables found on financials page")
            
            table = tables[0]  # Main income statement table
            
            # Get headers
            thead = table.find("thead")
            if not thead:
                raise ValueError("No table headers found")
            
            header_rows = thead.find_all("tr")
            dates = []
            for th in header_rows[0].find_all("th")[1:]:  # Skip first column (label)
                date_text = th.get_text(strip=True)
                # Parse dates like "Sep '25" or "Dec '24"
                if date_text and date_text not in ["TTM", "FY 2024", "FY 2023"]:
                    dates.append(date_text)
            
            # Get Revenue row
            tbody = table.find("tbody")
            if not tbody:
                raise ValueError("No table body found")
            
            revenue_row = None
            for row in tbody.find_all("tr"):
                first_cell = row.find("td")
                if first_cell and "revenue" in first_cell.get_text(strip=True).lower():
                    # Make sure it's the main Revenue row, not "Revenue Growth"
                    if first_cell.get_text(strip=True).lower() == "revenue":
                        revenue_row = row
                        break
            
            if not revenue_row:
                raise ValueError("Revenue row not found in table")
            
            # Parse revenue values
            values = []
            for td in revenue_row.find_all("td")[1:len(dates)+1]:
                val_text = td.get_text(strip=True).replace(",", "")
                values.append(money_to_float(val_text))
            
            # Create dataframe
            df_data = []
            for i, date_str in enumerate(dates):
                if i < len(values):
                    # Parse date (e.g., "Sep '25" -> "2025-09-30")
                    try:
                        parts = date_str.replace("'", "").split()
                        if len(parts) == 2:
                            month_str, year_str = parts
                            year = int("20" + year_str) if len(year_str) == 2 else int(year_str)
                            date_obj = pd.to_datetime(f"{month_str} {year}", format="%b %Y")
                            date_obj = snap_to_quarter_end(date_obj)
                            df_data.append({"date": date_obj, "revenue": values[i] * 1e6})  # Convert millions to actual
                    except:
                        continue
            
            df = pd.DataFrame(df_data)
            if df.empty or len(df) < 4:
                raise ValueError("Insufficient quarterly data parsed")
            
            return df.sort_values("date").reset_index(drop=True)
            
        except Exception as e:
            if attempt == len(UA_HEADERS_LIST):
                st.warning(f"Could not load financials data: {str(e)}")
                return pd.DataFrame(columns=["date", "revenue"])
            continue
    
    return pd.DataFrame(columns=["date", "revenue"])


# =============================
# FORECASTING
# =============================
def estimate_growth_q(series: pd.Series, lookback_quarters: int = 8) -> Tuple[float, float]:
    """Estimate quarterly growth rate and volatility"""
    s = series.dropna().astype(float)
    if len(s) < 6:
        return 0.02, 0.05
    s = s.iloc[-lookback_quarters:] if len(s) > lookback_quarters else s
    g = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 3:
        return 0.02, 0.05
    return float(g.mean()), float(g.std(ddof=1) if len(g) > 1 else 0.05)


def forecast_series(hist_df: pd.DataFrame, years: int, uplift_annual: float, lookback_quarters: int = 8) -> pd.DataFrame:
    """Generate forecast with uncertainty bands"""
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
        # Use 1.28 std devs for ~80% confidence interval
        band = (std_q if std_q > 0 else 0.05) * 1.28 * np.sqrt(i)
        hi.append(cur * (1.0 + band))
        lo.append(max(0, cur * (1.0 - band)))  # Revenue can't be negative

    return pd.DataFrame({"date": future_dates, "forecast": fc, "hi": hi, "lo": lo})


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
        Interactive segment-level **quarterly** analysis and scenario forecasting.  
        **Data sources:** StockAnalysis (segment revenue) + WallStreetZen (total revenue validation)
        """
    )

st.markdown("---")


# =============================
# SIDEBAR
# =============================
st.sidebar.title("Controls")

if st.sidebar.button("Force Refresh Data", help="Clear cache and reload all data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Parameters")

years = st.sidebar.slider("Forecast years", min_value=1, max_value=10, value=3, step=1)
uplift = st.sidebar.slider(
    "Extra annual growth (%)", 
    min_value=0, max_value=30, value=0, step=1,
    help="Add extra optimistic growth on top of historical trends"
) / 100.0

lookback = st.sidebar.slider(
    "Lookback quarters for trend",
    min_value=4, max_value=16, value=8, step=2,
    help="How many recent quarters to use for estimating growth rate"
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Increase 'Extra annual growth' to model optimistic scenarios")


# =============================
# LOAD DATA
# =============================
with st.spinner("Loading segment data from StockAnalysis..."):
    try:
        seg_q = load_revenue_by_segment_quarterly()
        st.sidebar.success(f"Loaded {len(seg_q)} segment records")
    except Exception as e:
        st.error(f"Failed to load segment data: {str(e)}")
        st.stop()

products = sorted(seg_q["product"].unique().tolist())
default_product = "Advertising" if "Advertising" in products else products[0]
product = st.sidebar.selectbox("Product Segment", products, index=products.index(default_product))


# =============================
# TABS
# =============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Segment Forecast", "Total Revenue Forecast", "Data Check", "Download"]
)

# TAB 1: SEGMENT FORECAST
with tab1:
    st.subheader(f"Revenue Forecast: {product}")
    
    seg = seg_q[seg_q["product"] == product].copy().sort_values("date")
    
    if seg.empty:
        st.warning(f"No data available for {product}")
    else:
        fc = forecast_series(seg[["date", "revenue"]], years=years, uplift_annual=uplift, lookback_quarters=lookback)

        # Metrics
        last_q_date = pd.to_datetime(seg["date"].max())
        last_q_rev = float(seg.loc[seg["date"] == last_q_date, "revenue"].iloc[-1])
        
        end_fc = float(fc["forecast"].iloc[-1]) if not fc.empty else np.nan
        base_fc = forecast_series(seg[["date", "revenue"]], years=years, uplift_annual=0.0, lookback_quarters=lookback)
        base_end = float(base_fc["forecast"].iloc[-1]) if not base_fc.empty else np.nan
        delta = end_fc - base_end if (not np.isnan(end_fc) and not np.isnan(base_end)) else np.nan
        
        # Calculate CAGR
        if not np.isnan(end_fc) and last_q_rev > 0:
            cagr = ((end_fc / last_q_rev) ** (1 / years) - 1) * 100
        else:
            cagr = np.nan

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Reported (Quarterly)", money_fmt(last_q_rev), last_q_date.strftime("%b %Y"))
        col2.metric("End Forecast", money_fmt(end_fc), f"+{money_fmt(end_fc - last_q_rev)}" if not np.isnan(end_fc) else "—")
        col3.metric("Uplift vs Baseline", money_fmt(delta), f"+{uplift*100:.0f}% scenario")
        col4.metric("Implied CAGR", f"{cagr:.1f}%" if not np.isnan(cagr) else "—")

        # Chart
        y_title = "Revenue (USD, Quarterly)"
        fig = build_plot_lines(
            seg[["date", "revenue"]],
            fc,
            title=f"{product}: {years}-Year Forecast (uplift +{uplift*100:.0f}%, lookback {lookback}Q)",
            y_title=y_title
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent data
        with st.expander("View recent quarterly data"):
            recent = seg.tail(12).copy()
            recent["Period"] = pd.to_datetime(recent["date"]).dt.strftime("%b %Y")
            recent["Revenue"] = recent["revenue"].apply(money_fmt)
            recent["QoQ Growth"] = seg["revenue"].pct_change().tail(12).apply(lambda x: f"{x*100:.1f}%" if not np.isnan(x) else "—")
            st.dataframe(recent[["Period", "Revenue", "QoQ Growth"]].iloc[::-1], use_container_width=True)

# TAB 2: TOTAL FORECAST
with tab2:
    st.subheader("Total Alphabet Revenue Forecast")
    
    # Historical TOTAL from quarterly segment sums
    total_hist = segment_sum_total_from_quarterly(seg_q)
    total_hist = total_hist.sort_values("date").dropna(subset=["revenue"])

    # Forecast TOTAL by forecasting each segment using QUARTERLY data
    wide_q = seg_q.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index().dropna(how="all")

    future_steps = years * 4
    last_date = wide_q.index.max()
    future_dates = pd.date_range(last_date + pd.offsets.QuarterEnd(1), periods=future_steps, freq="Q")
    future_dates = pd.to_datetime(pd.Series(future_dates).apply(snap_to_quarter_end))

    total_fc_vals = np.zeros(future_steps, dtype=float)
    total_hi_vals = np.zeros(future_steps, dtype=float)
    total_lo_vals = np.zeros(future_steps, dtype=float)

    # Sum forecasts from all segments
    for p in wide_q.columns:
        hist_p = wide_q[[p]].reset_index().rename(columns={p: "revenue"}).dropna(subset=["revenue"])
        if hist_p.empty:
            continue
        fcp = forecast_series(hist_p[["date", "revenue"]], years=years, uplift_annual=uplift, lookback_quarters=lookback)
        if fcp.empty:
            continue
        total_fc_vals += fcp["forecast"].values
        total_hi_vals += fcp["hi"].values
        total_lo_vals += fcp["lo"].values

    total_fc = pd.DataFrame({
        "date": future_dates, 
        "forecast": total_fc_vals, 
        "hi": total_hi_vals, 
        "lo": total_lo_vals
    })

    # Metrics
    last_q_total = float(total_hist.loc[total_hist["date"] == last_date, "revenue"].iloc[-1])
    end_fc_total = float(total_fc["forecast"].iloc[-1]) if not total_fc.empty else np.nan
    
    if not np.isnan(end_fc_total) and last_q_total > 0:
        cagr_total = ((end_fc_total / last_q_total) ** (1 / years) - 1) * 100
    else:
        cagr_total = np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Reported Total", money_fmt(last_q_total), pd.to_datetime(last_date).strftime("%b %Y"))
    col2.metric("Forecasted Total", money_fmt(end_fc_total), f"+{money_fmt(end_fc_total - last_q_total)}" if not np.isnan(end_fc_total) else "—")
    col3.metric("Implied CAGR", f"{cagr_total:.1f}%" if not np.isnan(cagr_total) else "—")

    # Chart
    fig_total = build_plot_lines(
        total_hist[["date", "revenue"]],
        total_fc,
        title=f"Total Alphabet Revenue: {years}-Year Forecast (Quarterly, +{uplift*100:.0f}% uplift)",
        y_title="Revenue (USD, Quarterly)"
    )
    st.plotly_chart(fig_total, use_container_width=True)

# TAB 3: DATA CHECK
with tab3:
    st.subheader("Segment Data Quality Check")
    
    latest_dt = seg_q["date"].max()
    latest_dt_str = pd.to_datetime(latest_dt).strftime("%b %d, %Y")
    
    st.info(f"**Latest quarter in dataset:** {latest_dt_str}")
    
    # Check data freshness
    days_old = (pd.Timestamp.now() - pd.Timestamp(latest_dt)).days
    if days_old > 120:
        st.warning(f"Data may be stale ({days_old} days old). Consider refreshing.")
    
    # Latest quarter breakdown
    chk = seg_q[seg_q["date"] == latest_dt].copy().sort_values("revenue", ascending=False)
    chk["revenue_fmt"] = chk["revenue"].apply(money_fmt)
    chk["% of Total"] = (chk["revenue"] / chk["revenue"].sum() * 100).apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        chk[["product", "revenue_fmt", "% of Total"]].rename(columns={
            "product": "Segment",
            "revenue_fmt": "Revenue"
        }), 
        use_container_width=True,
        hide_index=True
    )
    
    # Historical data availability
    st.markdown("---")
    st.subheader("Data Availability by Segment")
    
    avail = seg_q.groupby("product").agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        num_quarters=("date", "count")
    ).reset_index()
    avail["first_date"] = pd.to_datetime(avail["first_date"]).dt.strftime("%b %Y")
    avail["last_date"] = pd.to_datetime(avail["last_date"]).dt.strftime("%b %Y")
    
    st.dataframe(
        avail.rename(columns={
            "product": "Segment",
            "first_date": "First Quarter",
            "last_date": "Last Quarter",
            "num_quarters": "# Quarters"
        }),
        use_container_width=True,
        hide_index=True
    )

# TAB 4: DOWNLOAD
with tab4:
    st.subheader("Download Data")

    st.markdown("Export the underlying data used in this dashboard for further analysis.")
    
    # Segment data (tidy format)
    seg_download = seg_q.copy()
    seg_download["revenue"] = seg_download["revenue"].astype(float)
    seg_download["date"] = pd.to_datetime(seg_download["date"]).dt.strftime("%Y-%m-%d")

    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Download Segment Data (Tidy Format)",
            data=seg_download.to_csv(index=False).encode("utf-8"),
            file_name="alphabet_segments_quarterly_tidy.csv",
            mime="text/csv",
            help="Long format with columns: date, product, revenue"
        )

    # Segment data (wide format)
    wide_out = seg_q.pivot_table(index="date", columns="product", values="revenue", aggfunc="sum").sort_index()
    wide_out.reset_index(inplace=True)
    wide_out["date"] = pd.to_datetime(wide_out["date"]).dt.strftime("%Y-%m-%d")
    
    with col2:
        st.download_button(
            label="Download Segment Data (Wide Format)",
            data=wide_out.to_csv(index=False).encode("utf-8"),
            file_name="alphabet_segments_quarterly_wide.csv",
            mime="text/csv",
            help="Wide format with date as rows and segments as columns"
        )
    
    st.markdown("---")
    st.caption("All data is exported in CSV format for easy import into Excel, Python, R, or other analysis tools.")
