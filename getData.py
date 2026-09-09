import yfinance as yf

data = yf.download(
    "AAPL",
    start="2015-01-01",
    end="2026-01-01",
    auto_adjust=False
)

data = data[["Open", "High", "Low", "Close", "Volume"]]

data.to_csv("AAPL.csv")

print(data.head())