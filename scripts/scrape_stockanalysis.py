import pandas as pd
import requests
import io

URL_SEG = "https://stockanalysis.com/stocks/goog/metrics/revenue-by-segment/"
URL_INC = "https://stockanalysis.com/stocks/goog/financials/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def scrape(url, outname):
    html = requests.get(url, headers=HEADERS, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    best = max(tables, key=lambda t: t.shape[1])
    best.to_csv(outname, index=False)

if __name__ == "__main__":
    scrape(URL_SEG, "data/segment_raw.csv")
    scrape(URL_INC, "data/income_raw.csv")
