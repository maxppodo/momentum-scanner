import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.set_page_config(page_title="Momentum Scanner", layout="wide")
st.title("Momentum-Pullback / Trendfolge Scanner (15m + Daily)")
st.caption("Fix: robuste OHLCV-Normalisierung (MultiIndex/2D Close)")

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _to_1d_series(x) -> pd.Series:
    """ta erwartet 1D; macht aus (n,1)-DataFrame eine Series."""
    if isinstance(x, pd.DataFrame):
        # (n,1) -> Series
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        # Wenn doch mehrere Spalten: erste nehmen (sollte nicht passieren nach normalize)
        return x.iloc[:, 0]
    return x


def normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Macht aus yfinance-Output garantiert Spalten: Open/High/Low/Close/Volume als 1D."""
    if df is None or df.empty:
        return None

    out = df.copy()

    # yfinance kann MultiIndex-Spalten liefern (z.B. (Price, Ticker)) [web:192]
    if isinstance(out.columns, pd.MultiIndex):
        # Häufig: Level 0 = Price (Open/High/...), Level 1 = Ticker
        # oder umgekehrt. Wir versuchen beides robust.
        cols = out.columns
        lvl0 = set(cols.get_level_values(0))
        lvl1 = set(cols.get_level_values(1))

        if set(REQUIRED_COLS).issubset(lvl0):
            # (Price, Ticker) -> nimm Price-Level
            out.columns = cols.get_level_values(0)
        elif set(REQUIRED_COLS).issubset(lvl1):
            # (Ticker, Price) -> nimm Price-Level
            out.columns = cols.get_level_values(1)
        else:
            # Fallback: flatten
            out.columns = ["_".join([str(a), str(b)]) for a, b in cols.to_flat_index()]

    # Wenn nach flatten sowas wie "Close_TSLA" existiert: extrahiere OHLCV
    if not set(REQUIRED_COLS).issubset(out.columns):
        # Suche columns, die mit OHLCV anfangen
        mapped = {}
        for c in out.columns:
            for k in REQUIRED_COLS:
                if str(c).startswith(k):
                    mapped[k] = c
        if set(REQUIRED_COLS).issubset(mapped.keys()):
            out = out[[mapped[k] for k in REQUIRED_COLS]].rename(columns={mapped[k]: k for k in REQUIRED_COLS})
        else:
            # Letzter Versuch: bei Single-Ticker: manchmal ist nur 'Close' als (n,1)
            if "Close" in out.columns and "Volume" in out.columns:
                pass
            else:
                return None

    # Sicherstellen: nur benötigte Spalten, keine 2D-Spalten
    out = out[REQUIRED_COLS].copy()
    for c in REQUIRED_COLS:
        out[c] = _to_1d_series(out[[c]] if isinstance(out[c], pd.DataFrame) else out[c]).astype(float)

    out = out.dropna()
    return out if not out.empty else None


@st.cache_data(ttl=1800)
def load_watchlist(path: str = "watchlist.csv") -> list[str]:
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError("watchlist.csv braucht eine Spalte 'Ticker'")
    tickers = df["Ticker"].dropna().astype(str).str.strip()
    return tickers[tickers != ""].unique().tolist()


@st.cache_data(ttl=1800)
def fetch_ohlc(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    # multi_level_index=False kann je nach yfinance-Version helfen; normalize_ohlcv bleibt trotzdem nötig. [web:192]
    raw = yf.download(
        tickers=ticker,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=False,
    )
    return normalize_ohlcv(raw, ticker)


def add_indicators_intraday(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close = _to_1d_series(out["Close"])
    out["ema9"] = EMAIndicator(close, window=9).ema_indicator()
    out["ema21"] = EMAIndicator(close, window=21).ema_indicator()
    out["rsi14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()

    # VWAP pro Tag (15m-Daten enthalten mehrere Tage)
    idx = out.index
    # Sicherstellen, dass DatetimeIndex vorhanden ist
    if not isinstance(idx, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
        idx = out.index

    day = idx.tz_localize(None).date if idx.tz is not None else idx.date
    tp = (out["High"] + out["Low"] + out["Close"]) / 3.0
    pv = tp * out["Volume"]

    out["_day"] = day
    out["_pv"] = pv
    out["_vol"] = out["Volume"]

    out["vwap"] = out.groupby("_day")["_pv"].cumsum() / out.groupby("_day")["_vol"].cumsum().replace(0, np.nan)
    out.drop(columns=["_day", "_pv", "_vol"], inplace=True)

    return out


def score_row(h15: pd.DataFrame, d1: pd.DataFrame) -> tuple[float, str, float]:
    # Daily Trendfilter
    dclose = _to_1d_series(d1["Close"])
    ema50 = EMAIndicator(dclose, window=50).ema_indicator().iloc[-1]
    ema200 = EMAIndicator(dclose, window=200).ema_indicator().iloc[-1]

    price = float(_to_1d_series(h15["Close"]).iloc[-1])

    if price > ema50 > ema200:
        direction = "LONG"
        trend_pts = 30
    elif price < ema50 < ema200:
        direction = "SHORT"
        trend_pts = 30
    else:
        direction = "NEUTRAL"
        trend_pts = 10

    h = add_indicators_intraday(h15)
    last = h.iloc[-1]

    # EMA9/21 Struktur
    ema_pts = 0
    if last["Close"] > last["ema9"] > last["ema21"]:
        ema_pts = 25
    elif last["Close"] < last["ema9"] < last["ema21"]:
        ema_pts = 25

    # RSI passend zur Richtung
    rsi_pts = 0
    if direction == "LONG" and 50 <= last["rsi14"] <= 65:
        rsi_pts = 20
    elif direction == "SHORT" and 35 <= last["rsi14"] <= 50:
        rsi_pts = 20

    # MACD
    macd_pts = 15 if last["macd"] > last["macd_signal"] else 0

    # VWAP
    vwap_pts = 10 if last["Close"] > last["vwap"] else 0

    raw = trend_pts + ema_pts + rsi_pts + macd_pts + vwap_pts  # 0..100
    raw = float(max(0, min(100, raw)))

    return raw, direction, price


def perf_from_daily(d1: pd.DataFrame, price: float) -> tuple[float, float]:
    dclose = _to_1d_series(d1["Close"])
    one_week = float((price / dclose.iloc[-5] - 1.0) * 100.0) if len(dclose) >= 6 else np.nan
    one_month = float((price / dclose.iloc[-21] - 1.0) * 100.0) if len(dclose) >= 22 else np.nan
    return one_week, one_month


# UI
tickers = load_watchlist()

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    interval = st.selectbox("Intraday-Intervall", ["15m", "30m", "60m"], index=0)
with col2:
    limit = st.number_input("Max. Ticker scannen (Start klein!)", min_value=50, max_value=5000, value=min(200, len(tickers)), step=50)
with col3:
    run = st.button("Full Refresh (scan)", type="primary")

st.caption("Tipp: Erst mit 200 Tickern stabil deployen, dann hochdrehen (Yahoo kann rate-limiten).")

if run:
    scan_list = tickers[: int(limit)]
    rows = []
    prog = st.progress(0)
    status = st.empty()

    for i, t in enumerate(scan_list):
        status.write(f"Scanne: {t} ({i+1}/{len(scan_list)})")
        h15 = fetch_ohlc(t, interval=interval, period="60d")
        d1 = fetch_ohlc(t, interval="1d", period="2y")

        if h15 is None or d1 is None or len(d1) < 220 or len(h15) < 50:
            prog.progress((i + 1) / len(scan_list))
            continue

        score, direction, price = score_row(h15, d1)
        if direction == "NEUTRAL":
            prog.progress((i + 1) / len(scan_list))
            continue

        # Intraday perf (letzte Kerze Open->Close)
        intraday = float((price / float(h15["Open"].iloc[-1]) - 1.0) * 100.0)
        w1, m1 = perf_from_daily(d1, price)

        rows.append(
            {
                "Ticker": t,
                "Price": round(price, 4),
                "Direction": direction,
                "Score": round(score, 1),
                "%Intraday": round(intraday, 2),
                "%1W": None if np.isnan(w1) else round(w1, 2),
                "%1M": None if np.isnan(m1) else round(m1, 2),
            }
        )

        prog.progress((i + 1) / len(scan_list))

    res = pd.DataFrame(rows)
    if res.empty:
        st.error("Keine Ergebnisse. Prüfe Ticker/Intervalle/Logs.")
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

        st.download_button(
            "Download Result CSV",
            data=res.to_csv(index=False).encode("utf-8"),
            file_name="scan_results.csv",
            mime="text/csv",
        )
else:
    st.info("Klicke auf ‘Full Refresh (scan)’.")
