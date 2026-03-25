import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# SETUP LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN   = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# ============================================================
# CONSTANTS & PARAMETERS
# ============================================================
TOP_COINS_BY_VOLUME = 40    
RANKED_CANDIDATES   = 12    
CANDLES_REQUIRED    = 100   

ATR_PERIOD   = 14
RSI_PERIOD   = 14
EMA_FAST     = 20
EMA_SLOW     = 50
EMA_TREND    = 200
SWING_LOOKBACK = 10         

SEMAPHORE_PHASE1 = 5   
SEMAPHORE_PHASE2 = 3   
PHASE2_DELAY     = 0.3 

MAX_RETRIES  = 3
RETRY_DELAY  = 5

# ============================================================
# UTILITY: INDIKATOR
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, prev_close = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low  - prev_close).abs()], axis=1).max(axis=1)
    tr_median = tr.rolling(50, min_periods=1).median()
    tr_clean  = pd.Series(np.where(tr > (tr_median * 4), tr_median, tr), index=df.index)
    return tr_clean.rolling(period).mean()

def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / np.where(loss == 0, 1e-9, loss)
    return 100 - (100 / (1 + rs))

def calc_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()

def calc_macd(close: pd.Series):
    ema12, ema26 = calc_ema(close, 12), calc_ema(close, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def calc_squeeze(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    close  = df['close']
    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    bb_upper, bb_lower = sma20 + (2 * std20), sma20 - (2 * std20)
    kc_upper, kc_lower = sma20 + (1.5 * atr), sma20 - (1.5 * atr)
    return (bb_upper < kc_upper) & (bb_lower > kc_lower)

def _find_swings(highs, lows, lookback: int):
    n = len(highs)
    sh, sl = [], []
    for i in range(lookback, n - lookback):
        lh, rh = highs[i - lookback: i], highs[i + 1: i + lookback + 1]
        if len(lh) == lookback and len(rh) == lookback:
            if highs[i] >= max(lh) and highs[i] >= max(rh): sh.append((i, float(highs[i])))
        ll, rl = lows[i - lookback: i], lows[i + 1: i + lookback + 1]
        if len(ll) == lookback and len(rl) == lookback:
            if lows[i] <= min(ll) and lows[i] <= min(rl): sl.append((i, float(lows[i])))
    return sh, sl

def calc_swing_levels(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> dict:
    highs, lows = df['high'].values, df['low'].values
    sh_list, sl_list = [], []
    for lb in range(lookback, 2, -1):
        sh_list, sl_list = _find_swings(highs, lows, lb)
        if sh_list and sl_list: break
    if not sh_list: sh_list = [(-1, float(df['high'].rolling(20).max().iloc[-1]))]
    if not sl_list: sl_list = [(-1, float(df['low'].rolling(20).min().iloc[-1]))]
    return {'swing_high': sh_list[-1][1], 'swing_low': sl_list[-1][1]}

def calc_structure_bias(price: float, swing_high: float, swing_low: float) -> str:
    if not swing_high or not swing_low: return "MID RANGE"
    range_size = swing_high - swing_low
    if range_size == 0: return "MID RANGE"
    pos = (price - swing_low) / range_size
    if price > swing_high: return "BREAKOUT"
    elif price < swing_low: return "BREAKDOWN"
    elif pos > 0.65: return "NEAR RESISTANCE"
    elif pos < 0.35: return "NEAR SUPPORT"
    return "MID RANGE"

# ============================================================
# DATA PIPELINE
# ============================================================

async def fetch_ohlcv_safe(exchange, symbol: str, timeframe: str, limit: int):
    try:
        ohlcv = await asyncio.wait_for(exchange.fetch_ohlcv(symbol, timeframe, limit=limit), timeout=10.0)
        return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except: return None

def build_indicators(df: pd.DataFrame) -> dict:
    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
    atr = calc_atr(df)
    v_range = high - low
    n_delta = pd.Series(np.where(v_range == 0, 0, ((close - low) - (high - close)) / v_range * volume), index=df.index)
    a_delta = n_delta.abs().rolling(20).mean()
    s_flow  = np.clip((n_delta / np.where(a_delta == 0, 1, a_delta)) * 20, -40, 40)
    rsi = calc_rsi(close)
    ps = s_flow + np.clip((rsi - 50) * 1.2, -30, 30)
    swings = calc_swing_levels(df)
    price = float(close.iloc[-1])
    return {
        'close': price, 'power_score': round(float(ps.iloc[-1]), 1), 'rsi': round(float(rsi.iloc[-1]), 1),
        'atr': round(float(atr.iloc[-1]), 6), 'is_squeezing': bool(calc_squeeze(df, atr).iloc[-1]),
        'swing_high': round(swings['swing_high'], 6), 'swing_low': round(swings['swing_low'], 6),
        'struct_bias': calc_structure_bias(price, swings['swing_high'], swings['swing_low']),
        'z_score': round((volume.iloc[-1] - volume.rolling(20).mean().iloc[-1]) / volume.rolling(20).std().iloc[-1], 2),
        'macd_hist': calc_macd(close)[2].iloc[-1], 'ema_f': calc_ema(close, EMA_FAST).iloc[-1], 'ema_s': calc_ema(close, EMA_SLOW).iloc[-1]
    }

async def phase1_scan(exchange, coin: dict) -> dict | None:
    symbol = coin['symbol']
    df = await fetch_ohlcv_safe(exchange, symbol, '1h', CANDLES_REQUIRED)
    if df is None: return None
    ind = build_indicators(df)
    return {
        'symbol': symbol, 'Symbol': symbol.split(':')[0], 'Price': ind['close'],
        'power_1h': ind['power_score'], 'RSI_1H': ind['rsi'], 'ATR_1H': ind['atr'],
        'z_score': ind['z_score'], 'Squeeze_1H': ind['is_squeezing'], 'Struct_1H': ind['struct_bias'],
        'Swing_High_1H': ind['swing_high'], 'Swing_Low_1H': ind['swing_low'],
        'MACD_1H': "BULL" if ind['macd_hist'] > 0 else "BEAR"
    }

async def phase2_enrich(exchange, candidate: dict) -> dict:
    symbol = candidate['symbol']
    await asyncio.sleep(PHASE2_DELAY)
    df = await fetch_ohlcv_safe(exchange, symbol, '4h', CANDLES_REQUIRED)
    if df is None:
        candidate.update({'Trend_4H': 'N/A', 'power_4h': candidate['power_1h']})
        return candidate
    ind = build_indicators(df)
    c4 = ind['close']
    trend = "UPTREND" if c4 > ind['ema_f'] > ind['ema_s'] else "DOWNTREND" if c4 < ind['ema_f'] < ind['ema_s'] else "RANGING"
    candidate.update({'Trend_4H': trend, 'power_4h': ind['power_score'], 'RSI_4H': ind['rsi']})
    return candidate

def finalize_candidate(c: dict) -> dict:
    p1, p4 = c['power_1h'], c.get('power_4h', c['power_1h'])
    comp = (p4 * 0.55) + (p1 * 0.45)
    if c['Squeeze_1H']: comp += 10 if comp > 0 else -10
    c['Composite'] = round(comp, 1)
    c['Matrix_Sync'] = "FULL BULL" if comp >= 40 else "BULLISH" if comp > 10 else "FULL BEAR" if comp <= -40 else "BEARISH" if comp < -10 else "NEUTRAL"
    p, a, sh, sl = c['Price'], c['ATR_1H'], c['Swing_High_1H'], c['Swing_Low_1H']
    c.update({
        'SL_Long': round(max(p - a * 1.5, sl * 0.998), 6), 'TP1_Long': round(p + a * 2.0, 6), 'TP2_Long': round(p + a * 3.5, 6),
        'SL_Short': round(min(p + a * 1.5, sh * 1.002), 6), 'TP1_Short': round(p - a * 2.0, 6), 'TP2_Short': round(p - a * 3.5, 6)
    })
    return c

async def get_high_precision_data():
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        top = sorted([v for v in tickers.values() if v.get('quoteVolume')], key=lambda x: x['quoteVolume'], reverse=True)[:TOP_COINS_BY_VOLUME]
        sem1 = asyncio.Semaphore(SEMAPHORE_PHASE1)
        async def s1(c): 
            async with sem1: return await phase1_scan(exchange, c)
        p1_res = await asyncio.gather(*[s1(c) for c in top])
        top_c = sorted([r for r in p1_res if r], key=lambda x: abs(x['power_1h']), reverse=True)[:RANKED_CANDIDATES]
        sem2 = asyncio.Semaphore(SEMAPHORE_PHASE2)
        async def s2(c):
            async with sem2: return await phase2_enrich(exchange, c)
        final = [finalize_candidate(c) for c in await asyncio.gather(*[s2(c) for c in top_c])]
        return sorted(final, key=lambda x: abs(x['Composite']), reverse=True)
    finally: await exchange.close()

# ============================================================
# AI & TELEGRAM (ANTI-ERROR VERSION)
# ============================================================

async def send_to_telegram(text: str, header: str = ""):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    full_text = header + text
    chunks = [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]
    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            payload = {"chat_id": TG_CHAT_ID, "text": chunk, "parse_mode": "Markdown"}
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    payload.pop("parse_mode") # Fallback: Kirim tanpa format jika Markdown error
                    await session.post(url, json=payload)
            await asyncio.sleep(1)

async def ask_ai_agent(data_list: list) -> str:
    if not data_list: return "⚠️ Scan gagal."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    narrative = "\n".join([f"**{d['Symbol']}** ({d['Price']}) - Matrix: {d['Matrix_Sync']}, Score: {d['Composite']}, Squeeze: {d['Squeeze_1H']}, Struct: {d['Struct_1H']}" for d in data_list])
    prompt = f"""Kamu adalah AI Trader. Berikut 12 koin hasil scan:
{narrative}
Pilih 5 TERBAIK. Berikan Entry, SL, TP, dan Red Flag. 
Wajib: 
1. Jangan pakai karakter garis bawah (_) di luar Markdown. 
2. Pastikan tanda bintang (*) selalu berpasangan.
3. R:R minimal 1:2."""
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}) as resp:
            if resp.status == 200: return (await resp.json())['candidates'][0]['content']['parts'][0]['text']
    return "⚠️ Gemini API Error."

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): return
    hdr = f"🎯 *GOD MODE v3.3 — STABLE*\n🕐 {datetime.now().strftime('%H:%M WIB')} | MTF + Squeeze + Fallback\n{'─' * 40}\n\n"
    try:
        data = await get_high_precision_data()
        await send_to_telegram(await ask_ai_agent(data), header=hdr)
    except Exception as e: await send_to_telegram(f"❌ ERROR: {str(e)[:100]}")

if __name__ == "__main__": asyncio.run(main())
