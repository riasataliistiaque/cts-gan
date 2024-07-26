from yfinance import download
from pathlib import Path


def p0_1_raw_to_px():
    tickers = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT']
    s_date = '2012-05-18'
    e_date = '2023-10-28'
    folder = './raw/'

    stock_db = download(tickers, s_date, e_date)
    stock_px = stock_db['Adj Close'].dropna()
    stock_px.columns = tickers

    Path(folder).mkdir(parents=True, exist_ok=True)
    round(stock_px, 5).to_csv(f'{folder}stock_px.csv')


if __name__ == '__main__':
    p0_1_raw_to_px()
