# ==========================
# MRS Strategy Live (Bollinger Band RSI and Wave combination)
# ==========================


import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dt_time
import time
import os
import requests
import pytz

# ---------------- SETTINGS ----------------
symbols = [
    "360ONE.NS", "ABB.NS", "APLAPOLLO.NS", "AUBANK.NS", "ADANIENSOL.NS", "ADANIENT.NS",
    "ADANIGREEN.NS", "ADANIPORTS.NS", "ATGL.NS", "ABCAPITAL.NS", "ABFRL.NS", "ALKEM.NS", "AMBER.NS", "AMBUJACEM.NS",
    "ANGELONE.NS", "APOLLOHOSP.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "AUROPHARMA.NS", "DMART.NS",
    "AXISBANK.NS", "BSE.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BANDHANBNK.NS", "BANKBARODA.NS",
    "BANKINDIA.NS", "BDL.NS", "BEL.NS", "BHARATFORG.NS", "BHEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BIOCON.NS",
    "BLUESTARCO.NS", "BOSCHLTD.NS", "BRITANNIA.NS", "CESC.NS", "CGPOWER.NS", "CANBK.NS", "CDSL.NS", "CHOLAFIN.NS",
    "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CAMS.NS", "CONCOR.NS", "CROMPTON.NS", "CUMMINSIND.NS",
    "CYIENT.NS", "DLF.NS", "DABUR.NS", "DALBHARAT.NS", "DELHIVERY.NS", "DIVISLAB.NS", "DIXON.NS", "DRREDDY.NS",
    "ETERNAL.NS", "EICHERMOT.NS", "EXIDEIND.NS", "NYKAA.NS", "FORTIS.NS", "GAIL.NS", "GMRAIRPORT.NS", "GLENMARK.NS",
    "GODREJCP.NS", "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HFCL.NS", "HAVELLS.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HAL.NS", "HINDPETRO.NS",
    "HINDUNILVR.NS", "HINDZINC.NS", "HUDCO.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDFCFIRSTB.NS",
    "IIFL.NS", "IRB.NS", "ITC.NS", "INDIANB.NS", "IEX.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS", "IREDA.NS", "IGL.NS",
    "INDUSTOWER.NS", "INDUSINDBK.NS", "NAUKRI.NS", "INFY.NS", "INOXWIND.NS", "INDIGO.NS", "JSWENERGY.NS", "JSWSTEEL.NS",
    "JSL.NS", "JINDALSTEL.NS", "JIOFIN.NS", "JUBLFOOD.NS", "KEI.NS", "KPITTECH.NS", "KALYANKJIL.NS", "KAYNES.NS",
    "KFINTECH.NS", "KOTAKBANK.NS", "LTF.NS", "LICHSGFIN.NS", "LTIM.NS", "LT.NS", "LAURUSLABS.NS", "LICI.NS",
    "LODHA.NS", "LUPIN.NS", "M&M.NS", "MANAPPURAM.NS", "MANKIND.NS", "MARICO.NS", "MARUTI.NS", "MFSL.NS",
    "MAXHEALTH.NS", "MAZDOCK.NS", "MPHASIS.NS", "MCX.NS", "MUTHOOTFIN.NS", "NBCC.NS", "NCC.NS", "NHPC.NS",
    "NMDC.NS", "NTPC.NS", "NATIONALUM.NS", "NESTLEIND.NS", "NUVAMA.NS", "OBEROIRLTY.NS", "ONGC.NS", "OIL.NS",
    "PAYTM.NS", "OFSS.NS", "POLICYBZR.NS", "PGEL.NS", "PIIND.NS", "PNBHOUSING.NS", "PAGEIND.NS", "PATANJALI.NS",
    "PERSISTENT.NS", "PETRONET.NS", "PIDILITIND.NS", "PPLPHARMA.NS", "POLYCAB.NS", "POONAWALLA.NS", "PFC.NS",
    "POWERGRID.NS", "PRESTIGE.NS", "PNB.NS", "RBLBANK.NS", "RECLTD.NS", "RVNL.NS", "RELIANCE.NS", "SBICARD.NS",
    "SBILIFE.NS", "SHREECEM.NS", "SJVN.NS", "SRF.NS", "MOTHERSON.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SOLARINDS.NS",
    "SONACOMS.NS", "SBIN.NS", "SAIL.NS", "SUNPHARMA.NS", "SUPREMEIND.NS", "SUZLON.NS", "SYNGENE.NS", "TATACONSUM.NS",
    "TITAGARH.NS", "TVSMOTOR.NS", "TATACHEM.NS", "TCS.NS", "TATAELXSI.NS", "TATAPOWER.NS", "TATASTEEL.NS",
    "TATATECH.NS", "TECHM.NS", "FEDERALBNK.NS", "INDHOTEL.NS", "PHOENIXLTD.NS", "TITAN.NS", "TORNTPHARM.NS",
    "TORNTPOWER.NS", "TRENT.NS", "TIINDIA.NS", "UNOMINDA.NS", "UPL.NS", "ULTRACEMCO.NS", "UNIONBANK.NS",
    "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "IDEA.NS", "VOLTAS.NS", "WIPRO.NS", "YESBANK.NS", "ZYDUSLIFE.NS"
]
interval = "15m"
period = "50d"
n1, n2 = 10, 21
crossZone = 80
ist = pytz.timezone('Asia/Kolkata')

TELEGRAM_TOKEN = "8401397406:AAFdTIQkK6Ky6eqvQRiXZqcraTr5wMYtLXM"
CHAT_ID = "684584763"

#TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
#CHAT_ID = os.environ['CHAT_ID']

# ---------- UTILITIES ----------

def send_telegram_message(telegram_token: str, chat_id: str, text: str, parse_mode: str = "Markdown"):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Telegram send error: {e}")
        return None

def clean_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

# ---------- INDICATOR FUNCTIONS ----------
def calculate_wavetrend(df, n1=10, n2=21):
    df["hlc3"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["esa"] = df["hlc3"].ewm(span=n1, adjust=False).mean()
    df["d"] = df["hlc3"].sub(df["esa"]).abs().ewm(span=n1, adjust=False).mean()
    df["ci"] = (df["hlc3"] - df["esa"]) / (0.015 * df["d"])
    df["tci"] = df["ci"].ewm(span=n2, adjust=False).mean()
    df["wt1"] = df["tci"]
    df["wt2"] = df["wt1"].rolling(4).mean()
    return df

def calculate_rsi(df, period=14):
    delta = df["Close"].diff().fillna(0).to_numpy().ravel()
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    gain_series = pd.Series(gain, index=df.index)
    loss_series = pd.Series(loss, index=df.index)
    avg_gain = gain_series.rolling(window=period, min_periods=1).mean()
    avg_loss = loss_series.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger(df, period=20, std_dev=2):
    close = df["Close"].astype(float).to_numpy().ravel()
    mid = pd.Series(close, index=df.index).rolling(window=period, min_periods=1).mean()
    std = pd.Series(close, index=df.index).rolling(window=period, min_periods=1).std(ddof=0)
    df["BB_Mid"] = mid
    df["BB_Upper"] = mid + (std_dev * std)
    df["BB_Lower"] = mid - (std_dev * std)
    return df

def generate_signals(df, crossZone=75):
    cross_up = (df["wt1"].shift(1) < df["wt2"].shift(1)) & (df["wt1"] > df["wt2"]) & (df["wt1"] < -crossZone)
    cross_down = (df["wt1"].shift(1) > df["wt2"].shift(1)) & (df["wt1"] < df["wt2"]) & (df["wt1"] > crossZone)
    rsi_buy = df["RSI"].shift(1) < 30
    rsi_sell = df["RSI"].shift(1) > 70
    prev_close_below_lower = df["Close"].shift(1) < df["BB_Lower"].shift(1)
    prev_close_above_upper = df["Close"].shift(1) > df["BB_Upper"].shift(1)
    curr_close_above_lower = df["Close"] > df["BB_Lower"]
    curr_close_below_upper = df["Close"] < df["BB_Upper"]
    green_candle = df["Close"] > df["Open"]
    red_candle = df["Close"] < df["Open"]
    df["BuySignal"] = cross_up & rsi_buy & prev_close_below_lower & curr_close_above_lower & green_candle
    df["SellSignal"] = cross_down & rsi_sell & prev_close_above_upper & curr_close_below_upper & red_candle
    df["Signal"] = np.where(df["BuySignal"], "Buy", np.where(df["SellSignal"], "Sell", ""))
    return df

# ---------- SCAN & NOTIFY ----------
def run_scan_once():
    for symbol in symbols:

        print(f"📡 Scanning {symbol}")

        try:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
            if df.empty:
                continue

            df = clean_df(df)
            df = calculate_wavetrend(df)
            df = calculate_rsi(df)
            df = calculate_bollinger(df)
            df = generate_signals(df)

            last = df.iloc[-1]
            if last["Signal"] == "":
                continue

            sig = last["Signal"]
            price = round(last["Close"], 2)
            sig_emoji = "🟢" if sig == "Buy" else "🔴"

            dt = df.index[-1].tz_localize("UTC").astimezone(ist).strftime("%Y-%m-%d %H:%M")

            msg = f"*{symbol}*\n{sig_emoji} {sig} (MRS)\n{dt} IST\nClose: `{price}`"

            if send_telegram_message(TELEGRAM_TOKEN, CHAT_ID, msg):
                print(f"✔️ Alert sent for {symbol}")

        except Exception as e:
            print(f"❌ Error: {symbol}: {e}")


# Run once and exit
run_scan_once()
print("\n✅ Scan finished.\n")
