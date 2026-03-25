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
# CONSTANTS & PARAMETERS (V4.2 FINAL EDITION)
# ============================================================
TOP_COINS_BY_VOLUME = 40    
RANKED_CANDIDATES   = 12    
CANDLES_REQUIRED    = 100   

# KUNCI MANAJEMEN RISIKO
FIXED_RISK_USD = 1.50   

TF_MACRO  = '1h'    # Trend Utama
TF_STRUCT = '15m'   # Micro-Structure & Swing Level
TF_MICRO  = '5m'    # Presisi Volume/Whale Anomaly

ATR_PERIOD     = 14
RSI_PERIOD     = 14
EMA_FAST       = 20
EMA_SLOW       = 50
EMA_TREND      = 200
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

# FIX: Fungsi Deteksi Smart Money Terstruktur
def detect_smart_money(z_score: float, squeeze: bool) -> dict:
    if z_score > 3.0:
        level, bonus = "NUCLEAR", 15
    elif z_score > 1.5:
        level, bonus = "ACTIVE", 7
    else:
        level, bonus = "QUIET", 0
    
    squeeze_bonus = 8 if squeeze else 0
    return {'whale_level': level, 'sm_bonus': bonus + squeeze_bonus}

# ============================================================
# DATA PIPELINE
# ============================================================

async def fetch_ohlcv_safe(exchange, symbol: str, timeframe: str, limit: int):
    try:
        ohlcv = await asyncio.wait_for(exchange.fetch_ohlcv(symbol, timeframe, limit=limit), timeout=10.0)
        # BUG FIX: Pastikan jumlah candle cukup sebelum dikonversi ke DataFrame
        if ohlcv and len(ohlcv) >= CANDLES_REQUIRED:
            return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except asyncio.TimeoutError:
        logging.debug(f"fetch_ohlcv {symbol} {timeframe}: Timeout 10s")
    except Exception as e:
        logging.debug(f"fetch_ohlcv {symbol} {timeframe}: {e}")
    return None

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
    
    # Fase 2: Tarik 1H (Trend) dan 5m (Whale/Trigger) secara paralel
    task_1h = fetch_ohlcv_safe(exchange, symbol, TF_MACRO, CANDLES_REQUIRED)
    task_5m = fetch_ohlcv_safe(exchange, symbol, TF_MICRO, CANDLES_REQUIRED)
    df_1h, df_5m = await asyncio.gather(task_1h, task_5m)
    
    if df_1h is not None:
        ind1h = build_indicators(df_1h)
        c1h = ind1h['close']
        trend = "UPTREND" if c1h > ind1h['ema_f'] > ind1h['ema_s'] else "DOWNTREND" if c1h < ind1h['ema_f'] < ind1h['ema_s'] else "RANGING"
        candidate.update({'Trend_1h': trend, 'power_1h': ind1h['power_score']})
    else:
        candidate.update({'Trend_1h': 'N/A', 'power_1h': candidate['power_15m']})

    if df_5m is not None:
        ind5m = build_indicators(df_5m)
        sm5 = detect_smart_money(ind5m['z_score'], ind5m['is_squeezing'])
        candidate.update({'z_score_5m': ind5m['z_score'], 'Whale_5m': sm5['whale_level'], 'SM_Bonus': sm5['sm_bonus'], 'power_5m': ind5m['power_score']})
    else:
        candidate.update({'z_score_5m': 0, 'Whale_5m': 'QUIET', 'SM_Bonus': 0, 'power_5m': candidate['power_15m']})
        
    return candidate

def format_qty(qty: float) -> float:
    """Format kuantitas: Koin murah jadikan int, koin mahal (BTC) beri desimal."""
    if qty > 100: return int(round(qty, 0))
    elif qty > 10: return round(qty, 1)
    elif qty > 1: return round(qty, 2)
    return round(qty, 4)

def finalize_candidate(c: dict) -> dict:
    p15 = c['power_15m']
    p1h = c.get('power_1h', p15)
    p5m = c.get('power_5m', p15)
    
    comp = (p1h * 0.40) + (p15 * 0.40) + (p5m * 0.20)
    
    # 1. Penalti Counter Trend
    trend = c.get('Trend_1h', 'RANGING')
    td_dir = 1 if trend == 'UPTREND' else (-1 if trend == 'DOWNTREND' else 0)
    if td_dir != 0 and (1 if p15 > 0 else -1) != td_dir: comp -= 15

    # 2. Smart Money & Squeeze Bonus (Dari v4.0)
    sm_bonus = c.get('SM_Bonus', 0)
    if sm_bonus > 0: comp += sm_bonus if comp > 0 else -sm_bonus
    
    # 3. 3TF Alignment Bonus (Dari v4.0) - Sinyal paling bersih
    all_aligned = ((p5m > 0 and p15 > 0 and p1h > 0) or (p5m < 0 and p15 < 0 and p1h < 0))
    if all_aligned: comp += 5 if comp > 0 else -5

    c['Composite'] = round(comp, 1)
    c['Matrix_Sync'] = "FULL BULL" if comp >= 40 else "BULLISH" if comp > 10 else "FULL BEAR" if comp <= -40 else "BEARISH" if comp < -10 else "NEUTRAL"
    
    p, a_15, sh_15, sl_15 = c['Price'], c['ATR_15m'], c['Swing_High_15m'], c['Swing_Low_15m']
    
    buy_stop_price = round(sh_15 * 1.001, 6) if sh_15 > p else round(p + (a_15 * 0.2), 6)
    sell_stop_price = round(sl_15 * 0.999, 6) if sl_15 < p else round(p - (a_15 * 0.2), 6)

    sl_long = round(max(buy_stop_price - a_15 * 1.8, sl_15 * 0.998), 6)
    sl_short = round(min(sell_stop_price + a_15 * 1.8, sh_15 * 1.002), 6)
    
    # KALKULATOR KUANTITAS BERDASARKAN RISK USD
    risk_dist_long = max(abs(buy_stop_price - sl_long), 1e-9)
    qty_long = format_qty(FIXED_RISK_USD / risk_dist_long)
    
    risk_dist_short = max(abs(sell_stop_price - sl_short), 1e-9)
    qty_short = format_qty(FIXED_RISK_USD / risk_dist_short)

    c.update({
        'Buy_Stop': buy_stop_price, 'SL_Long': sl_long, 'Qty_Long': qty_long,
        'TP1_Long': round(buy_stop_price + a_15 * 3.0, 6), 'TP2_Long': round(buy_stop_price + a_15 * 5.0, 6),
        'Sell_Stop': sell_stop_price, 'SL_Short': sl_short, 'Qty_Short': qty_short,
        'TP1_Short': round(sell_stop_price - a_15 * 3.0, 6), 'TP2_Short': round(sell_stop_price - a_15 * 5.0, 6)
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
# AI & TELEGRAM (ANTI-ERROR & RETRY FIX)
# ============================================================

async def send_to_telegram(text: str, header: str = ""):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    full_text = header + text
    chunks = [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]
    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            payload = {"chat_id": TG_CHAT_ID, "text": chunk, "parse_mode": "Markdown"}
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    async with session.post(url, json=payload, timeout=15) as resp:
                        if resp.status == 200: break
                        if resp.status == 400: # Telegram format error
                            payload.pop("parse_mode")
                            await session.post(url, json=payload)
                            break
                except: pass
                await asyncio.sleep(RETRY_DELAY)

def format_coin_for_prompt(i: int, d: dict) -> str:
    action = "LONG (BUY STOP)" if d['Composite'] > 0 else "SHORT (SELL STOP)"
    sqz_txt = "🚨 SQUEEZE DETECTED" if d.get('Squeeze_15m') else "Normal"

    return f"""
**{i}. {d['Symbol']}** (Live: {d['Price']})
- Kecenderungan: {action} | Skor: {d['Composite']} ({d['Matrix_Sync']})
- Trend 1H (Macro): {d['Trend_1h']} | Volatilitas 15m: {sqz_txt}
- Anomali Whale 5m: {d['Whale_5m']} (Volume Z-score: {d['z_score_5m']}) 
- Rencana LONG (Buy Stop): {d['Buy_Stop']} | SL: {d['SL_Long']} | Qty/Amount: {d['Qty_Long']} koin
- Rencana SHORT (Sell Stop): {d['Sell_Stop']} | SL: {d['SL_Short']} | Qty/Amount: {d['Qty_Short']} koin
(Data Qty dihitung untuk meresikokan tepat ${FIXED_RISK_USD})"""

async def ask_ai_agent(data_list: list) -> str:
    if not data_list: return "⚠️ Scan gagal."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    narratives = "\n".join([format_coin_for_prompt(i, d) for i, d in enumerate(data_list, 1)])
    
    prompt = f"""Kamu adalah AI Intraday Breakout Trader Profesional. Berikut 12 koin hasil scan "God Mode v4.2 Final":
{narratives}

TUGASMU: Pilih TEPAT 5 KOIN TERBAIK untuk setup Breakout (Hold intraday).

KRITERIA PRIORITAS:
1. SQUEEZE + WHALE: Cari "SQUEEZE DETECTED" yang diiringi "NUCLEAR".
2. ALIGNMENT: Arah Breakout harus searah Trend 1H.

ATURAN FORMAT PENULISAN:
- Jangan gunakan karakter garis bawah (_) di luar format Markdown. 
- Pastikan semua tanda bintang (*) selalu berpasangan.

FORMAT WAJIB:
### [Nomor]. [NAMA KOIN] — [LONG / SHORT]
**Konfluensi:** [Alasan kuat Breakout]
**Entry Order:** [Tulis "BUY STOP di (Harga)" atau "SELL STOP di (Harga)"] 
**Amount (Kuantitas):** [Isi dengan data Qty/Amount koin] _(Risk Fixed ${FIXED_RISK_USD})_
**Stop Loss:** [Harga] 
**TP1:** [Harga] | **TP2:** [Harga]
**🚩 Red Flag Invalidasi:** [Kondisi setup dibatalkan]
"""
    async with aiohttp.ClientSession() as session:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # BUG FIX: Ensure the timeout parameter is respected and we catch the response properly
                async with session.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60) as resp:
                    if resp.status == 200: return (await resp.json())['candidates'][0]['content']['parts'][0]['text']
            except Exception as e: logging.warning(f"Gemini attempt {attempt} failed: {e}")
            await asyncio.sleep(RETRY_DELAY)
    return "⚠️ Gemini API Error."

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): return
    hdr = f"👑 *GOD MODE v4.2 — ULTIMATE FINAL*\n🕐 {datetime.now().strftime('%H:%M WIB')} | Breakout & Fixed Risk ${FIXED_RISK_USD}\n{'─' * 46}\n\n"
    try:
        data = await get_high_precision_data()
        await send_to_telegram(await ask_ai_agent(data), header=hdr)
    except Exception as e: await send_to_telegram(f"❌ ERROR: {str(e)[:100]}")

if __name__ == "__main__": asyncio.run(main())
