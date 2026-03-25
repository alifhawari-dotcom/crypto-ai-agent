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
# CONSTANTS & PARAMETERS (INTRADAY BREAKOUT EDITION)
# ============================================================
TOP_COINS_BY_VOLUME = 40    
RANKED_CANDIDATES   = 12    
CANDLES_REQUIRED    = 100   

TF_MACRO  = '1h'    # Trend Utama (Untuk hold berjam-jam)
TF_STRUCT = '15m'   # Micro-Structure & Swing Level
TF_MICRO  = '5m'    # Presisi Volume/Whale Anomaly

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
    # Anomaly filter untuk melindungi SL dari flash crash (jarum panjang)
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

# ============================================================
# DATA PIPELINE (INTRADAY BREAKOUT)
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
    
    vol_avg = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    z_score = (volume.iloc[-1] - vol_avg.iloc[-1]) / np.where(vol_std.iloc[-1] == 0, 1, vol_std.iloc[-1])
    
    return {
        'close': price, 'power_score': round(float(ps.iloc[-1]), 1), 'rsi': round(float(rsi.iloc[-1]), 1),
        'atr': round(float(atr.iloc[-1]), 6), 'is_squeezing': bool(calc_squeeze(df, atr).iloc[-1]),
        'swing_high': round(swings['swing_high'], 6), 'swing_low': round(swings['swing_low'], 6),
        'z_score': round(z_score, 2),
        'macd_hist': calc_macd(close)[2].iloc[-1], 'ema_f': calc_ema(close, EMA_FAST).iloc[-1], 'ema_s': calc_ema(close, EMA_SLOW).iloc[-1]
    }

async def phase1_scan(exchange, coin: dict) -> dict | None:
    symbol = coin['symbol']
    # Fase 1: Memindai TF 15 Menit (Micro Structure & Squeeze Detection)
    df = await fetch_ohlcv_safe(exchange, symbol, TF_STRUCT, CANDLES_REQUIRED)
    if df is None: return None
    ind = build_indicators(df)
    
    return {
        'symbol': symbol, 'Symbol': symbol.split(':')[0], 'Price': ind['close'],
        'power_15m': ind['power_score'], 'RSI_15m': ind['rsi'], 'ATR_15m': ind['atr'],
        'Squeeze_15m': ind['is_squeezing'], 
        'Swing_High_15m': ind['swing_high'], 'Swing_Low_15m': ind['swing_low'],
        'MACD_15m': "BULL" if ind['macd_hist'] > 0 else "BEAR"
    }

async def phase2_enrich(exchange, candidate: dict) -> dict:
    symbol = candidate['symbol']
    await asyncio.sleep(PHASE2_DELAY)
    
    # Fase 2: Paralel ambil Trend 1H dan Presisi/Whale 5m
    task_1h = fetch_ohlcv_safe(exchange, symbol, TF_MACRO, CANDLES_REQUIRED)
    task_5m = fetch_ohlcv_safe(exchange, symbol, TF_MICRO, CANDLES_REQUIRED)
    df_1h, df_5m = await asyncio.gather(task_1h, task_5m)
    
    # Proses 1H (Trend Makro)
    if df_1h is not None:
        ind1h = build_indicators(df_1h)
        c1h = ind1h['close']
        trend = "UPTREND" if c1h > ind1h['ema_f'] > ind1h['ema_s'] else "DOWNTREND" if c1h < ind1h['ema_f'] < ind1h['ema_s'] else "RANGING"
        candidate.update({'Trend_1h': trend, 'power_1h': ind1h['power_score']})
    else:
        candidate.update({'Trend_1h': 'N/A', 'power_1h': candidate['power_15m']})

    # Proses 5m (Whale Anomaly & Entry Presisi)
    if df_5m is not None:
        ind5m = build_indicators(df_5m)
        whale = "NUCLEAR (WHALE)" if ind5m['z_score'] > 3.0 else "ACTIVE" if ind5m['z_score'] > 1.5 else "QUIET"
        candidate.update({'z_score_5m': ind5m['z_score'], 'Whale_5m': whale, 'power_5m': ind5m['power_score']})
    else:
        candidate.update({'z_score_5m': 0, 'Whale_5m': 'QUIET', 'power_5m': candidate['power_15m']})
        
    return candidate

def finalize_candidate(c: dict) -> dict:
    p15 = c['power_15m']
    p1h = c.get('power_1h', p15)
    p5m = c.get('power_5m', p15)
    
    # Bobot: Trend 1H (40%), Structure 15m (40%), Trigger 5m (20%)
    comp = (p1h * 0.40) + (p15 * 0.40) + (p5m * 0.20)
    
    # Penalti counter-trend
    trend = c.get('Trend_1h', 'RANGING')
    td_dir = 1 if trend == 'UPTREND' else (-1 if trend == 'DOWNTREND' else 0)
    if td_dir != 0 and (1 if p15 > 0 else -1) != td_dir: comp -= 15

    # Bonus: Ada Whale 5m + Squeeze 15m = Breakout Imminent!
    if c['Squeeze_15m']: comp += 10 if comp > 0 else -10
    if c['z_score_5m'] >= 2.0: comp += 10 if comp > 0 else -10

    c['Composite'] = round(comp, 1)
    c['Matrix_Sync'] = "FULL BULL" if comp >= 40 else "BULLISH" if comp > 10 else "FULL BEAR" if comp <= -40 else "BEARISH" if comp < -10 else "NEUTRAL"
    
    p, a_15, sh_15, sl_15 = c['Price'], c['ATR_15m'], c['Swing_High_15m'], c['Swing_Low_15m']
    
    # --- LOGIKA BUY STOP / SELL STOP BREAKOUT ---
    # Long: Antre Buy Stop sedikit di atas Resistance (Swing High 15m)
    # Short: Antre Sell Stop sedikit di bawah Support (Swing Low 15m)
    
    buy_stop_price = round(sh_15 * 1.001, 6) if sh_15 > p else round(p + (a_15 * 0.2), 6)
    sell_stop_price = round(sl_15 * 0.999, 6) if sl_15 < p else round(p - (a_15 * 0.2), 6)

    # SL/TP Adaptive Intraday (Hold berjam-jam, mengacu ATR 15m)
    # R:R didesain 1:2 hingga 1:3
    c.update({
        'Buy_Stop': buy_stop_price,
        'SL_Long': round(max(buy_stop_price - a_15 * 1.8, sl_15 * 0.998), 6), 
        'TP1_Long': round(buy_stop_price + a_15 * 3.0, 6), 
        'TP2_Long': round(buy_stop_price + a_15 * 5.0, 6),
        
        'Sell_Stop': sell_stop_price,
        'SL_Short': round(min(sell_stop_price + a_15 * 1.8, sh_15 * 1.002), 6), 
        'TP1_Short': round(sell_stop_price - a_15 * 3.0, 6), 
        'TP2_Short': round(sell_stop_price - a_15 * 5.0, 6)
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
        top_c = sorted([r for r in p1_res if r], key=lambda x: abs(x['power_15m']), reverse=True)[:RANKED_CANDIDATES]
        
        sem2 = asyncio.Semaphore(SEMAPHORE_PHASE2)
        async def s2(c):
            async with sem2: return await phase2_enrich(exchange, c)
        final = [finalize_candidate(c) for c in await asyncio.gather(*[s2(c) for c in top_c])]
        return sorted(final, key=lambda x: abs(x['Composite']), reverse=True)
    finally: await exchange.close()

# ============================================================
# AI & TELEGRAM (FALLBACK SYSTEM)
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
                    payload.pop("parse_mode") 
                    await session.post(url, json=payload)
            await asyncio.sleep(1)

def format_coin_for_prompt(i: int, d: dict) -> str:
    action = "LONG (BUY STOP)" if d['Composite'] > 0 else "SHORT (SELL STOP)"
    
    sqz_txt = "🚨 SQUEEZE DETECTED" if d.get('Squeeze_15m') else "Normal"

    return f"""
**{i}. {d['Symbol']}** (Live Price: {d['Price']})
- Kecenderungan: {action} | Skor: {d['Composite']} ({d['Matrix_Sync']})
- Trend 1H (Macro): {d['Trend_1h']} | RSI 15m: {d['RSI_15m']} 
- Struktur 15m: Support di {d['Swing_Low_15m']}, Resistance di {d['Swing_High_15m']}
- Volatilitas 15m: {sqz_txt}
- Anomali Whale 5m: {d['Whale_5m']} (Volume Z-score: {d['z_score_5m']}) 
- Rencana BREAKOUT LONG → Entry (Buy Stop): {d['Buy_Stop']} | SL: {d['SL_Long']} | TP1: {d['TP1_Long']} | TP2: {d['TP2_Long']}
- Rencana BREAKOUT SHORT → Entry (Sell Stop): {d['Sell_Stop']} | SL: {d['SL_Short']} | TP1: {d['TP1_Short']} | TP2: {d['TP2_Short']}"""

async def ask_ai_agent(data_list: list) -> str:
    if not data_list: return "⚠️ Scan gagal."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    narratives = "\n".join([format_coin_for_prompt(i, d) for i, d in enumerate(data_list, 1)])
    
    prompt = f"""Kamu adalah AI Intraday Breakout Trader Profesional. Berikut 12 koin hasil scan "God Mode v4.0 Intraday Breakout":
{narratives}

TUGASMU: Pilih TEPAT 5 KOIN TERBAIK untuk setup Breakout (Hold intraday berjam-jam hingga 1 hari).

KRITERIA BREAKOUT PRIORITAS:
1. SQUEEZE + WHALE: Cari koin dengan "SQUEEZE DETECTED" (volatilitas mampat) yang diiringi "NUCLEAR (WHALE)". Ini adalah indikasi kuat harga akan segera Breakout.
2. ALIGNMENT: Arah Breakout harus searah dengan Trend 1H (Macro).
3. STRATEGI SET & FORGET: Selalu gunakan rekomendasi harga (Buy Stop / Sell Stop) yang diberikan. Jangan gunakan Limit Order. Kita ingin masuk hanya jika harga berhasil menembus Resistance/Support.

ATURAN FORMAT PENULISAN (PENTING!):
- Jangan gunakan karakter garis bawah (_) di luar format Markdown. 
- Pastikan semua tanda bintang (*) selalu berpasangan.

FORMAT WAJIB:
### [Nomor]. [NAMA KOIN] — [LONG / SHORT]
**Konfluensi:** [Sebutkan alasan kuat Breakout, misal: "Trend 1H UPTREND + Squeeze 15m + Whale 5m masuk"]
**Entry Order:** [Tuliskan "BUY STOP di (Harga)" atau "SELL STOP di (Harga)"] 
**Stop Loss:** [Harga] _(Alasan: misal di bawah Swing Low)_
**TP1:** [Harga] | **TP2:** [Harga]
**Risk/Reward:** [Rasio X:Y]
**🚩 Red Flag Invalidasi:** [Kondisi yang membuat setup dibatalkan sebelum tersentuh]
"""
    async with aiohttp.ClientSession() as session:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60) as resp:
                    if resp.status == 200: return (await resp.json())['candidates'][0]['content']['parts'][0]['text']
            except: pass
            await asyncio.sleep(RETRY_DELAY)
    return "⚠️ Gemini API Error."

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): return
    hdr = f"🌪️ *GOD MODE v4.0 — INTRADAY BREAKOUT*\n🕐 {datetime.now().strftime('%H:%M WIB')} | TF: 1H+15m+5m | Buy/Sell Stop Method\n{'─' * 46}\n\n"
    try:
        data = await get_high_precision_data()
        await send_to_telegram(await ask_ai_agent(data), header=hdr)
    except Exception as e: await send_to_telegram(f"❌ ERROR: {str(e)[:100]}")

if __name__ == "__main__": asyncio.run(main())
