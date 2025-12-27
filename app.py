import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gradio as gr
import requests
from io import StringIO

URL = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

FALLBACK_CSV = """date,product,revenue
2021-03-31,Advertising,44560
2021-06-30,Advertising,50700
2021-09-30,Advertising,53000
2021-12-31,Advertising,61700
2022-03-31,Advertising,54000
2022-06-30,Advertising,56400
2022-09-30,Advertising,54400
2022-12-31,Advertising,59300
2023-03-31,Advertising,54000
2023-06-30,Advertising,58100
2023-09-30,Advertising,59600
2023-12-31,Advertising,65500
"""

def load_data():
    try:
        r = requests.get(URL, headers=HEADERS, timeout=10)
        tables = pd.read_html(r.text)
        df = tables[0]
        df.columns = ["date", "Advertising"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.melt("date", var_name="product", value_name="revenue")
        return df
    except:
        df = pd.read_csv(StringIO(FALLBACK_CSV))
        df["date"] = pd.to_datetime(df["date"])
        return df

DATA = load_data()

def build_forecast(df, product, years, uplift):
    hist = df[df["product"] == product].copy()
    hist = hist.sort_values("date")

    base_growth = hist["revenue"].pct_change().mean()
    growth = base_growth + uplift

    last_date = hist["date"].max()
    last_rev = hist["revenue"].iloc[-1]

    future_dates = pd.date_range(last_date, periods=years + 1, freq="Y")[1:]
    future_rev = [last_rev * ((1 + growth) ** i) for i in range(1, years + 1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["revenue"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=future_dates, y=future_rev, mode="lines", name="Forecast"))

    fig.update_layout(
        title=f"{product} Revenue Forecast",
        yaxis_title="Revenue (USD Millions)",
        height=500
    )

    kpi = f"""
**Base CAGR:** {base_growth:.2%}  
**Scenario CAGR:** {growth:.2%}  
**Final Year Revenue:** ${future_rev[-1]:,.0f}M
"""

    return fig, kpi

def render(product, years, uplift):
    return build_forecast(DATA, product, years, uplift)

with gr.Blocks(title="Alphabet Revenue Forecast") as demo:
    gr.Markdown("# 📈 Alphabet (Google) Revenue Forecast Dashboard")

    with gr.Row():
        product = gr.Dropdown(["Advertising"], value="Advertising", label="Product")
        years = gr.Slider(3, 10, value=5, label="Forecast Years")
        uplift = gr.Slider(0, 0.10, value=0.00, step=0.01, label="Extra CAGR Uplift")
        run = gr.Button("Run Forecast")

    chart = gr.Plot()
    kpi = gr.Markdown()

    run.click(render, [product, years, uplift], [chart, kpi])
    demo.load(render, [product, years, uplift], [chart, kpi])

if __name__ == "__main__":
    demo.launch()
