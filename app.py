import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.set_page_config(page_title="Momentum Scanner", layout="wide")
st.title("Momentum-Pullback / Trendfolge Scanner")

@st.cache_data(ttl=1800)  # 30 Minuten Cache
def load_watchlist(path="watchlist.csv"):
    df = pd.read_csv(path)
    return df["Ticker"].dropna().astype(str).unique().tolist()

@st.cache_data(ttl=1800)
def fetch_ohlc(ticker: str, interval: str, period: str):
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    df = df.dropna()
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema9"] = EMAIndicator(out["Close"], window=9).ema_indicator()
    out["ema21"] = EMAIndicator(out["Close"], window=21).ema_indicator()

    out["rsi14"] = RSIIndicator(out["Close"], window=14).rsi()

    macd = MACD(out["Close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()

    # VWAP (klassisch intraday). Fallback: über gesamten DF berechnet.
    tp = (out["High"] + out["Low"] + out["Close"]) / 3.0
    out["vwap"] = (tp * out["Volume"]).cumsum() / (out["Volume"].cumsum().replace(0, np.nan))

    return out

def score_row(h15, d1):
    # Direction über Daily EMA50/200 (Trend-Bias)
    ema50 = EMAIndicator(d1["Close"], window=50).ema_indicator().iloc[-1]
    ema200 = EMAIndicator(d1["Close"], window=200).ema_indicator().iloc[-1]
    price = h15["Close"].iloc[-1]

    if price > ema50 > ema200:
        direction = "LONG"
        trend_pts = 30
    elif price < ema50 < ema200:
        direction = "SHORT"
        trend_pts = 30
    else:
        direction = "NEUTRAL"
        trend_pts = 10

    h = add_indicators(h15)
    last = h.iloc[-1]

    # EMA9/21 Struktur
    ema_pts = 0
    if last["Close"] > last["ema9"] > last["ema21"]:
        ema_pts = 25
    elif last["Close"] < last["ema9"] < last["ema21"]:
        ema_pts = 25

    # RSI
    rsi_pts = 0
    if direction == "LONG" and 50 <= last["rsi14"] <= 65:
        rsi_pts = 20
    if direction == "SHORT" and 35 <= last["rsi14"] <= 50:
        rsi_pts = 20

    # MACD
    macd_pts = 15 if last["macd"] > last["macd_signal"] else 0

    # VWAP
    vwap_pts = 10 if last["Close"] > last["vwap"] else 0

    raw = trend_pts + ema_pts + rsi_pts + macd_pts + vwap_pts  # 0..100
    return raw, direction, float(price)

tickers = load_watchlist()

colA, colB = st.columns([1, 1])
with colA:
    interval = st.selectbox("Intraday-Intervall", ["15m", "30m", "60m"], index=0)
with colB:
    run = st.button("Full Refresh (scan)")

if run:
    rows = []
    prog = st.progress(0)
    for i, t in enumerate(tickers):
        h15 = fetch_ohlc(t, interval=interval, period="60d")
        d1 = fetch_ohlc(t, interval="1d", period="2y")
        if h15 is None or d1 is None or len(d1) < 220:
            continue

        s, direction, price = score_row(h15, d1)
        if direction == "NEUTRAL":
            continue

        rows.append({"Ticker": t, "Price": price, "Direction": direction, "Score": s})
        prog.progress((i + 1) / len(tickers))

    res = pd.DataFrame(rows)
    if res.empty:
        st.warning("Keine Treffer. (Daten/Intervalle/Ticker prüfen)")
    else:
        long_df = res[res["Direction"] == "LONG"].sort_values("Score", ascending=False).head(25)
        short_df = res[res["Direction"] == "SHORT"].sort_values("Score", ascending=False).head(25)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 25 LONG")
            st.dataframe(long_df, use_container_width=True)
        with c2:
            st.subheader("Top 25 SHORT")
            st.dataframe(short_df, use_container_width=True)
else:
    st.info("Klicke auf ‘Full Refresh (scan)’.")
