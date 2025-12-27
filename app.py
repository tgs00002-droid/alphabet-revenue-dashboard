import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.image(
    "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png",
    width=120
)

st.title("Alphabet (Google) Revenue Forecast Dashboard")
st.caption("Quarterly segment revenue + income statement. Source: StockAnalysis")

# Load cached CSVs (NO SCRAPING HERE)
seg = pd.read_csv("data/segment_raw.csv")
inc = pd.read_csv("data/income_raw.csv")

st.subheader("Segment Revenue (Raw Quarterly Data)")
st.dataframe(seg, use_container_width=True)

st.subheader("Income Statement")
st.dataframe(inc, use_container_width=True)

# Example chart
value_cols = seg.columns[1:]
seg_long = seg.melt(id_vars=[seg.columns[0]], value_vars=value_cols,
                    var_name="date", value_name="revenue")

fig = px.line(
    seg_long,
    x="date",
    y="revenue",
    color=seg.columns[0],
    title="Segment Revenue Over Time"
)

st.plotly_chart(fig, use_container_width=True)
