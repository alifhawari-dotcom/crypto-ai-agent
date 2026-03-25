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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN   = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# ============================================================
# CONSTANTS
# ============================================================
TOP_COINS_BY_VOLUME = 40   # Pool awal dari bursa
RANKED_CANDIDATES   = 12   # Kandidat masuk fase 2 & 3
CANDLES_REQUIRED    = 100  # Minimal candle agar indikator akurat

ATR_PERIOD     = 14
RSI_PERIOD     = 14
EMA_FAST       = 20
EMA_SLOW       = 50
EMA_TREND      = 200
SWING_LOOKBACK = 10        # Adaptive, turun sampai 3 jika gagal

# ============================================================
# SCALPER TIMEFRAME ARCHITECTURE
# ─────────────────────────────────────────────────────────
# FASE 1  │ 15m  │ Scan 40 koin — momentum & smart money
#         │      │ 40 request, paralel semaphore=5
# ─────────────────────────────────────────────────────────
# FASE 2  │ 4H   │ Trend anchor — filter 12 kandidat terkuat
#         │      │ 12 request, semaphore=3, delay 0.3s
# ─────────────────────────────────────────────────────────
# FASE 3  │ 5m   │ Entry trigger — deteksi micro-momentum
#         │      │ 12 request, semaphore=3, delay 0.2s
# ─────────────────────────────────────────────────────────
# Total: 40 + 12 + 12 = 64 request (aman, tidak kena ban)
# ─────────────────────────────────────────────────────────
# Composite Score (scalping bobot):
#   4H = 40% (trend bias, patokan utama)
#   15m = 35% (momentum konfirmasi)
#   5m  = 25% (entry trigger, momentum real-time)
# ============================================================
SEMAPHORE_P1 = 5    # Fase 1: 40 koin @ 15m
SEMAPHORE_P2 = 3    # Fase 2: 12 koin @ 4H
SEMAPHORE_P3 = 3    # Fase 3: 12 koin @ 5m
DELAY_P2     = 0.3  # Detik jeda fase 2
DELAY_P3     = 0.2  # Detik jeda fase 3

MAX_RETRIES  = 3
RETRY_DELAY  = 5

# ============================================================
# INDIKATOR — SAMA PERSIS DENGAN v3.2
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    ATR dengan anomaly filter (v3.2 carry-over).
    TR spike >4x median rolling-50 diganti median — melindungi SL
    dari distorsi flash crash / candle spike palsu.
    """
    high, low, prev_close = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    tr_median = tr.rolling(50, min_periods=1).median()
    tr_clean  = pd.Series(
        np.where(tr > (tr_median * 4), tr_median, tr),
        index=df.index
    )
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
    ema12  = calc_ema(close, 12)
    ema26  = calc_ema(close, 26)
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal


def calc_squeeze(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    """
    Squeeze Detection (v3.2 carry-over).
    BB masuk ke KC = volatilitas dimampatkan → breakout imminent.
    Di scalping, squeeze di 15m/5m = sinyal sangat kuat untuk entry cepat.
    """
    close    = df['close']
    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)
    kc_upper = sma20 + (1.5 * atr)
    kc_lower = sma20 - (1.5 * atr)
    return (bb_upper < kc_upper) & (bb_lower > kc_lower)


# ============================================================
# MARKET STRUCTURE — ADAPTIVE SWING (v3.2 carry-over)
# ============================================================

def _find_swings(highs, lows, lookback: int):
    n = len(highs)
    sh, sl = [], []
    for i in range(lookback, n - lookback):
        lh = highs[i - lookback: i]
        rh = highs[i + 1: i + lookback + 1]
        if len(lh) == lookback and len(rh) == lookback:
            if highs[i] >= max(lh) and highs[i] >= max(rh):
                sh.append((i, float(highs[i])))
        ll = lows[i - lookback: i]
        rl = lows[i + 1: i + lookback + 1]
        if len(ll) == lookback and len(rl) == lookback:
            if lows[i] <= min(ll) and lows[i] <= min(rl):
                sl.append((i, float(lows[i])))
    return sh, sl


def calc_swing_levels(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> dict:
    """
    Adaptive fallback: lookback 10→3, lalu rolling max/min.
    Tidak pernah return None — tidak pernah UNKNOWN.
    """
    highs = df['high'].values
    lows  = df['low'].values
    sh_list, sl_list = [], []
    for lb in range(lookback, 2, -1):
        sh_list, sl_list = _find_swings(highs, lows, lb)
        if sh_list and sl_list:
            break
    if not sh_list:
        w = min(20, len(highs) // 4)
        fb = df['high'].rolling(w).max().dropna()
        if not fb.empty:
            sh_list = [(-1, float(fb.iloc[-1]))]
    if not sl_list:
        w = min(20, len(lows) // 4)
        fb = df['low'].rolling(w).min().dropna()
        if not fb.empty:
            sl_list = [(-1, float(fb.iloc[-1]))]
    return {
        'swing_high': sh_list[-1][1] if sh_list else None,
        'swing_low':  sl_list[-1][1] if sl_list else None,
        'prev_sh':    sh_list[-2][1] if len(sh_list) >= 2 else (sh_list[-1][1] if sh_list else None),
        'prev_sl':    sl_list[-2][1] if len(sl_list) >= 2 else (sl_list[-1][1] if sl_list else None),
    }


def calc_structure_bias(price: float, swing_high, swing_low) -> str:
    if swing_high is None or swing_low is None:
        return "UNKNOWN"
    r = swing_high - swing_low
    if r == 0:
        return "UNKNOWN"
    pos = (price - swing_low) / r
    if price > swing_high:   return "BREAKOUT"
    elif price < swing_low:  return "BREAKDOWN"
    elif pos > 0.65:         return "NEAR_RESISTANCE"
    elif pos < 0.35:         return "NEAR_SUPPORT"
    return "MID_RANGE"


# ============================================================
# SMART MONEY ANOMALY DETECTOR
# Deteksi aktivitas big player dari pola volume abnormal.
# Dua sinyal terkuat dari track record:
#   1. Volume Z-score (Whale NUCLEAR) — volume candle >> rata-rata
#   2. Squeeze (BB vs KC)             — dimampatkan sebelum meledak
# ============================================================

def detect_smart_money(ind: dict) -> dict:
    """
    Klasifikasi sinyal smart money berdasarkan Z-score + Squeeze.
    Return dict berisi level, label, dan bonus skor.
    """
    z       = ind['z_score']
    squeeze = ind['is_squeezing']

    # Whale level dari Z-score volume
    if z > 3.0:
        whale_level = "NUCLEAR"   # Big player masuk besar-besaran
        whale_bonus = 15          # Bonus skor tinggi — sinyal paling win
    elif z > 1.5:
        whale_level = "ACTIVE"    # Volume di atas normal
        whale_bonus = 7
    else:
        whale_level = "QUIET"
        whale_bonus = 0

    # Squeeze bonus tambahan jika bersamaan dengan whale
    squeeze_bonus = 8 if squeeze else 0

    # Combined smart money signal
    if whale_level == "NUCLEAR" and squeeze:
        sm_signal = "💥 NUCLEAR + SQUEEZE"   # Sinyal terkuat
    elif whale_level == "NUCLEAR":
        sm_signal = "🐳 NUCLEAR WHALE"
    elif squeeze:
        sm_signal = "🔥 SQUEEZE AKTIF"
    elif whale_level == "ACTIVE":
        sm_signal = "👀 WHALE ACTIVE"
    else:
        sm_signal = "😴 Quiet"

    return {
        'whale_level':   whale_level,
        'sm_signal':     sm_signal,
        'sm_bonus':      whale_bonus + squeeze_bonus,
    }


# ============================================================
# CORE INDICATOR BUILDER — SAMA UNTUK SEMUA TF
# ============================================================

async def fetch_ohlcv_safe(exchange, symbol: str, timeframe: str, limit: int):
    """Fetch dengan hard timeout 10s — tidak pernah hang."""
    try:
        ohlcv = await asyncio.wait_for(
            exchange.fetch_ohlcv(symbol, timeframe, limit=limit),
            timeout=10.0
        )
        if ohlcv and len(ohlcv) >= CANDLES_REQUIRED:
            return pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
    except asyncio.TimeoutError:
        logging.debug(f"fetch_ohlcv {symbol} {timeframe}: Timeout 10s")
    except Exception as e:
        logging.debug(f"fetch_ohlcv {symbol} {timeframe}: {e}")
    return None


def build_indicators(df: pd.DataFrame) -> dict:
    """Hitung semua indikator — matrix sama persis dengan v3.2."""
    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']

    # Volume Flow (Normalized Delta) — sinyal smart money
    v_range = high - low
    n_delta = pd.Series(
        np.where(v_range == 0, 0, ((close - low) - (high - close)) / v_range * volume),
        index=df.index
    )
    a_delta = n_delta.abs().rolling(20).mean()
    s_flow  = np.clip((n_delta / np.where(a_delta == 0, 1, a_delta)) * 20, -40, 40)

    rsi   = calc_rsi(close)
    s_mom = np.clip((rsi - 50) * 1.2, -30, 30)
    ps    = s_flow + s_mom

    ema_f, ema_s, ema_t = calc_ema(close, EMA_FAST), calc_ema(close, EMA_SLOW), calc_ema(close, EMA_TREND)
    macd, sig, hist     = calc_macd(close)
    atr                 = calc_atr(df)

    vol_avg = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    z_score = (volume - vol_avg) / np.where(vol_std == 0, 1, vol_std)

    swings  = calc_swing_levels(df)
    squeeze = calc_squeeze(df, atr)
    price   = float(df.iloc[-1]['close'])

    return {
        'close':        price,
        'power_score':  round(float(ps.iloc[-1]), 1),
        'rsi':          round(float(rsi.iloc[-1]), 1),
        'z_score':      round(float(z_score.iloc[-1]), 2),
        'ema_fast':     float(ema_f.iloc[-1]),
        'ema_slow':     float(ema_s.iloc[-1]),
        'ema_trend':    float(ema_t.iloc[-1]),
        'macd_hist':    round(float(hist.iloc[-1]), 6),
        'atr':          round(float(atr.iloc[-1]), 6),
        'swing_high':   round(swings['swing_high'], 6) if swings['swing_high'] else None,
        'swing_low':    round(swings['swing_low'], 6) if swings['swing_low'] else None,
        'struct_bias':  calc_structure_bias(price, swings['swing_high'], swings['swing_low']),
        'is_squeezing': bool(squeeze.iloc[-1]),
    }


# ============================================================
# FASE 1: SCAN 15m — Filter momentum awal
# 15m = TF terpendek yang masih punya sinyal bermakna untuk
# ranking. Lebih cepat daripada 1H, lebih stabil dari 5m.
# ============================================================

async def phase1_scan(exchange, coin: dict) -> dict | None:
    symbol = coin['symbol']
    df_15m = await fetch_ohlcv_safe(exchange, symbol, '15m', CANDLES_REQUIRED)
    if df_15m is None:
        return None

    ind = build_indicators(df_15m)
    sm  = detect_smart_money(ind)

    return {
        'symbol':         symbol,
        'Symbol':         symbol.split(':')[0],
        'Price':          ind['close'],
        'power_15m':      ind['power_score'],
        'RSI_15m':        ind['rsi'],
        'ATR_15m':        ind['atr'],
        'MACD_15m':       "BULL" if ind['macd_hist'] > 0 else "BEAR",
        'Swing_High_15m': ind['swing_high'],
        'Swing_Low_15m':  ind['swing_low'],
        'Struct_15m':     ind['struct_bias'],
        'Squeeze_15m':    ind['is_squeezing'],
        'z_score_15m':    ind['z_score'],
        # Smart money dari 15m
        'Whale':          sm['whale_level'],
        'SM_Signal':      sm['sm_signal'],
        'SM_Bonus':       sm['sm_bonus'],
        # Placeholder — diisi fase 2 & 3
        'Trend_4H':       'PENDING',
        'RSI_4H':         None,
        'power_4h':       None,
        'Swing_High_4H':  None,
        'Swing_Low_4H':   None,
        'Struct_4H':      'PENDING',
        'power_5m':       None,
        'RSI_5m':         None,
        'MACD_5m':        'PENDING',
        'Squeeze_5m':     False,
        'Struct_5m':      'PENDING',
        'Swing_High_5m':  None,
        'Swing_Low_5m':   None,
        'z_score_5m':     0.0,
        'ATR_5m':         ind['atr'],  # fallback ke 15m
    }


# ============================================================
# FASE 2: ENRICHMENT 4H — Trend anchor
# Patokan utama arah besar. Koin counter-trend 4H kena penalti.
# Hanya 12 kandidat terkuat dari fase 1 yang di-fetch.
# ============================================================

async def phase2_enrich_4h(exchange, candidate: dict) -> dict:
    symbol = candidate['symbol']
    await asyncio.sleep(DELAY_P2)
    df_4h = await fetch_ohlcv_safe(exchange, symbol, '4h', CANDLES_REQUIRED)

    if df_4h is None:
        candidate['Trend_4H']      = 'N/A'
        candidate['RSI_4H']        = candidate['RSI_15m']
        candidate['power_4h']      = candidate['power_15m']
        candidate['Swing_High_4H'] = candidate['Swing_High_15m']
        candidate['Swing_Low_4H']  = candidate['Swing_Low_15m']
        candidate['Struct_4H']     = candidate['Struct_15m']
        return candidate

    ind4 = build_indicators(df_4h)
    c4   = ind4['close']

    if c4 > ind4['ema_fast'] > ind4['ema_slow']:
        trend = "UPTREND"
    elif c4 < ind4['ema_fast'] < ind4['ema_slow']:
        trend = "DOWNTREND"
    else:
        trend = "RANGING"

    candidate['Trend_4H']      = trend
    candidate['RSI_4H']        = ind4['rsi']
    candidate['power_4h']      = ind4['power_score']
    candidate['Swing_High_4H'] = ind4['swing_high']
    candidate['Swing_Low_4H']  = ind4['swing_low']
    candidate['Struct_4H']     = ind4['struct_bias']
    return candidate


# ============================================================
# FASE 3: ENTRY TRIGGER 5m — Micro-momentum real-time
# 5m = filter akhir sebelum sinyal dikirim. Hanya koin yang
# momentumnya align di 5m yang layak untuk entry scalping.
# Juga tangkap whale anomaly di 5m — sering lebih dini dari 15m.
# ============================================================

async def phase3_entry_trigger(exchange, candidate: dict) -> dict:
    symbol = candidate['symbol']
    await asyncio.sleep(DELAY_P3)
    df_5m = await fetch_ohlcv_safe(exchange, symbol, '5m', CANDLES_REQUIRED)

    if df_5m is None:
        # Fallback: gunakan 15m sebagai proxy 5m
        candidate['power_5m']      = candidate['power_15m']
        candidate['RSI_5m']        = candidate['RSI_15m']
        candidate['MACD_5m']       = candidate['MACD_15m']
        candidate['Squeeze_5m']    = candidate['Squeeze_15m']
        candidate['Struct_5m']     = candidate['Struct_15m']
        candidate['Swing_High_5m'] = candidate['Swing_High_15m']
        candidate['Swing_Low_5m']  = candidate['Swing_Low_15m']
        return candidate

    ind5 = build_indicators(df_5m)
    sm5  = detect_smart_money(ind5)

    candidate['power_5m']      = ind5['power_score']
    candidate['RSI_5m']        = ind5['rsi']
    candidate['MACD_5m']       = "BULL" if ind5['macd_hist'] > 0 else "BEAR"
    candidate['Squeeze_5m']    = ind5['is_squeezing']
    candidate['Struct_5m']     = ind5['struct_bias']
    candidate['Swing_High_5m'] = ind5['swing_high']
    candidate['Swing_Low_5m']  = ind5['swing_low']
    candidate['z_score_5m']    = ind5['z_score']
    candidate['ATR_5m']        = ind5['atr']

    # Upgrade SM signal jika 5m juga deteksi whale
    if sm5['whale_level'] == "NUCLEAR" and candidate['Whale'] != "NUCLEAR":
        candidate['Whale']     = "NUCLEAR"
        candidate['SM_Signal'] = "💥 NUCLEAR (5m spike)"
        candidate['SM_Bonus']  = max(candidate['SM_Bonus'], sm5['sm_bonus'])

    return candidate


# ============================================================
# SCORING — COMPOSITE SCORE SCALPING
# Bobot disesuaikan untuk scalping: entry speed > trend depth
# 4H=40% (arah besar) | 15m=35% (konfirmasi) | 5m=25% (trigger)
# ============================================================

def finalize_candidate(c: dict) -> dict:
    p15m = c['power_15m']
    p4h  = c['power_4h']  if c['power_4h']  is not None else p15m
    p5m  = c['power_5m']  if c['power_5m']  is not None else p15m

    # Composite Score — bobot scalping
    composite = (p4h * 0.40) + (p15m * 0.35) + (p5m * 0.25)

    # Penalti counter-trend 4H (patokan utama tetap 4H)
    trend  = c.get('Trend_4H', 'RANGING')
    ps_dir = 1 if p15m > 0 else -1
    td_dir = 1 if trend == 'UPTREND' else (-1 if trend == 'DOWNTREND' else 0)
    if td_dir != 0 and ps_dir != td_dir:
        composite -= 15

    # Smart Money bonus (Whale NUCLEAR + Squeeze)
    sm_bonus = c.get('SM_Bonus', 0)
    if sm_bonus > 0:
        composite += sm_bonus if composite > 0 else -sm_bonus

    # ENTRY ALIGNMENT BONUS: 5m + 15m + 4H semua searah = sinyal paling bersih
    all_aligned = (
        (p5m > 0 and p15m > 0 and p4h > 0) or
        (p5m < 0 and p15m < 0 and p4h < 0)
    )
    if all_aligned:
        composite += 5 if composite > 0 else -5

    composite = round(composite, 1)

    # Matrix label — sama persis v3.2
    if composite >= 40:    matrix = "FULL BULL"
    elif composite > 10:   matrix = "BULLISH"
    elif composite <= -40: matrix = "FULL BEAR"
    elif composite < -10:  matrix = "BEARISH"
    else:                  matrix = "NEUTRAL"

    # ---- ATR-based SL/TP untuk SCALPING ----
    # Gunakan ATR 5m sebagai dasar SL (lebih ketat, sesuai scalping)
    # TP menggunakan ATR 15m untuk target yang lebih bermakna
    price    = c['Price']
    atr_5m   = c['ATR_5m']
    atr_15m  = c['ATR_15m']
    sh_5m    = c['Swing_High_5m']
    sl_5m    = c['Swing_Low_5m']
    sh_15m   = c['Swing_High_15m']
    sl_15m   = c['Swing_Low_15m']

    # SL: ATR 5m × 1.2 (lebih ketat dari swing) vs struktur 5m, pilih yang lebih konservatif
    atr_sl_long    = price - atr_5m * 1.2
    struct_sl_long = (sl_5m * 0.9995) if sl_5m else atr_sl_long
    sl_long        = round(max(atr_sl_long, struct_sl_long), 6)

    atr_sl_short    = price + atr_5m * 1.2
    struct_sl_short = (sh_5m * 1.0005) if sh_5m else atr_sl_short
    sl_short        = round(min(atr_sl_short, struct_sl_short), 6)

    # TP1: ATR 15m × 1.5 (target cepat, cocok scalping 15-30 menit)
    # TP2: Swing High/Low 15m (natural structure target)
    tp1_long  = round(price + atr_15m * 1.5, 6)
    tp2_long  = round(sh_15m * 0.9995, 6) if sh_15m and sh_15m > price else round(price + atr_15m * 2.5, 6)
    tp1_short = round(price - atr_15m * 1.5, 6)
    tp2_short = round(sl_15m * 1.0005, 6) if sl_15m and sl_15m < price else round(price - atr_15m * 2.5, 6)

    c.update({
        'Matrix_Sync': matrix,
        'Composite':   composite,
        'SL_Long':     sl_long,
        'TP1_Long':    tp1_long,
        'TP2_Long':    tp2_long,
        'SL_Short':    sl_short,
        'TP1_Short':   tp1_short,
        'TP2_Short':   tp2_short,
    })
    return c


# ============================================================
# DATA PIPELINE — 3 FASE
# ============================================================

async def get_high_precision_data():
    exchange = ccxt.gate({
        'options':         {'defaultType': 'swap'},
        'enableRateLimit': True,
    })
    try:
        tickers = await exchange.fetch_tickers()
        top_coins = sorted(
            [v for v in tickers.values() if v.get('quoteVolume')],
            key=lambda x: x['quoteVolume'],
            reverse=True
        )[:TOP_COINS_BY_VOLUME]

        # ── FASE 1: Scan 15m ──────────────────────────────────
        logging.info(f"Fase 1: Scanning {len(top_coins)} koin @ 15m...")
        sem1 = asyncio.Semaphore(SEMAPHORE_P1)
        async def safe_p1(coin):
            async with sem1:
                return await phase1_scan(exchange, coin)

        p1_raw    = await asyncio.gather(*[safe_p1(c) for c in top_coins])
        valid_p1  = [r for r in p1_raw if r is not None]

        # Ranking: power_15m absolut terkuat → 12 kandidat masuk fase 2
        # Smart money signal juga diperhitungkan di ranking awal
        def p1_rank_key(x):
            return abs(x['power_15m']) + x['SM_Bonus']

        top_cands = sorted(valid_p1, key=p1_rank_key, reverse=True)[:RANKED_CANDIDATES]
        logging.info(f"Fase 1 selesai: {len(top_cands)} kandidat (top power_15m + smart money).")

        # ── FASE 2: Enrichment 4H ────────────────────────────
        logging.info("Fase 2: Enrichment 4H (trend anchor)...")
        sem2 = asyncio.Semaphore(SEMAPHORE_P2)
        async def safe_p2(c):
            async with sem2:
                return await phase2_enrich_4h(exchange, c)

        enriched_4h = await asyncio.gather(*[safe_p2(c) for c in top_cands])
        logging.info("Fase 2 selesai.")

        # ── FASE 3: Entry Trigger 5m ─────────────────────────
        logging.info("Fase 3: Entry trigger 5m (micro-momentum)...")
        sem3 = asyncio.Semaphore(SEMAPHORE_P3)
        async def safe_p3(c):
            async with sem3:
                return await phase3_entry_trigger(exchange, c)

        enriched_5m = await asyncio.gather(*[safe_p3(c) for c in enriched_4h])
        logging.info("Fase 3 selesai.")

        # Finalisasi composite score + SL/TP
        final = [finalize_candidate(c) for c in enriched_5m]

        # Re-rank final berdasarkan Composite Score absolut
        final_ranked = sorted(final, key=lambda x: abs(x['Composite']), reverse=True)
        return final_ranked

    finally:
        await exchange.close()


# ============================================================
# PROMPT FORMAT — SCALPING EDITION
# Data disusun untuk konteks scalping: durasi pendek, entry
# presisi, SL ketat. AI diberitahu konteks 3 TF per koin.
# ============================================================

def format_coin_for_prompt(i: int, d: dict) -> str:
    action     = "LONG" if d['Composite'] > 0 else "SHORT"
    risk_long  = abs(d['Price'] - d['SL_Long'])  / d['Price'] * 100
    risk_short = abs(d['Price'] - d['SL_Short']) / d['Price'] * 100
    rr_long    = abs(d['TP1_Long']  - d['Price']) / max(abs(d['Price'] - d['SL_Long']),  1e-9)
    rr_short   = abs(d['TP1_Short'] - d['Price']) / max(abs(d['Price'] - d['SL_Short']), 1e-9)

    # 5m + 15m alignment check
    p5  = d.get('power_5m')  or 0
    p15 = d['power_15m']
    p4  = d.get('power_4h')  or 0
    aligned = (p5 > 0 and p15 > 0 and p4 > 0) or (p5 < 0 and p15 < 0 and p4 < 0)
    align_txt = "✅ 3TF ALIGNED" if aligned else "⚠️ Partial Align"

    # Squeeze status per TF
    sq_txt = []
    if d.get('Squeeze_15m'): sq_txt.append("15m")
    if d.get('Squeeze_5m'):  sq_txt.append("5m")
    sq_display = f"🔥 SQUEEZE di {'+'.join(sq_txt)}" if sq_txt else "Normal"

    return f"""
**{i}. {d['Symbol']}** | Price: {d['Price']} | {align_txt}
- Smart Money: {d['SM_Signal']} | Z-score 15m: {d['z_score_15m']} | Z-score 5m: {d.get('z_score_5m', 'N/A')}
- Squeeze: {sq_display}
- [4H] Trend: {d['Trend_4H']} | RSI: {d.get('RSI_4H', 'N/A')} | Struktur: {d['Struct_4H']} (S:{d['Swing_Low_4H']} R:{d['Swing_High_4H']})
- [15m] Power: {d['power_15m']} | RSI: {d['RSI_15m']} | MACD: {d['MACD_15m']} | Struktur: {d['Struct_15m']} (S:{d['Swing_Low_15m']} R:{d['Swing_High_15m']})
- [5m]  Power: {d.get('power_5m','N/A')} | RSI: {d.get('RSI_5m','N/A')} | MACD: {d.get('MACD_5m','N/A')} | Struktur: {d.get('Struct_5m','N/A')} (S:{d.get('Swing_Low_5m','N/A')} R:{d.get('Swing_High_5m','N/A')})
- Matrix: {d['Matrix_Sync']} | Composite: {d['Composite']}
- Rencana {action}: SL={d['SL_Long'] if action=='LONG' else d['SL_Short']} (-{risk_long:.1f}% / -{risk_short:.1f}%) | TP1={d['TP1_Long'] if action=='LONG' else d['TP1_Short']} | TP2={d['TP2_Long'] if action=='LONG' else d['TP2_Short']} | R:R≈1:{rr_long:.1f}"""


async def ask_ai_agent(data_list: list) -> str:
    if not data_list:
        return "⚠️ Tidak ada data valid."

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    )

    coin_narratives = "\n".join([format_coin_for_prompt(i, d) for i, d in enumerate(data_list, 1)])

    prompt = f"""Kamu adalah AI Scalp Trader profesional. Fokus: trade cepat, presisi tinggi, durasi 15 menit hingga 3 jam.

Berikut {len(data_list)} koin teratas hasil scan "God Mode Scalper" — data 3 timeframe per koin (4H, 15m, 5m):

{coin_narratives}

---
**KONTEKS SCALPING:**
- 4H = patokan arah besar (wajib searah, tidak boleh dilawan)
- 15m = konfirmasi momentum
- 5m = timing entry presisi

**KRITERIA SELEKSI (urut prioritas):**
1. **3TF ALIGNED** wajib diprioritaskan — 4H + 15m + 5m searah adalah setup terbersih
2. **Smart Money NUCLEAR atau NUCLEAR+SQUEEZE** = sinyal anomali big player, masuk daftar teratas
3. Harga di dekat Swing Level 15m atau 5m (NEAR_SUPPORT/NEAR_RESISTANCE) — bukan MID_RANGE
4. R:R minimal 1:1.5 di TP1 (standar scalping, lebih longgar dari swing)
5. Jangan pilih koin yang 4H RANGING + 15m MID_RANGE — tidak ada momentum jelas

**ATURAN ENTRY:**
- RSI 15m > 72: wajib Entry Limit di pullback (0.3×ATR_15m dari harga sekarang)
- RSI 15m < 28: wajib Entry Limit di retest
- RSI 15m 35–65: Entry Market diperbolehkan
- Jika Squeeze aktif di 5m: entry setelah candle 5m konfirmasi arah (jangan entry di tengah candle squeeze)

**FORMAT WAJIB untuk setiap koin:**

### [N]. [KOIN] — [LONG/SHORT] | ⏱ Est. {'{'}durasi{'}'}
**Signal:** [Ringkas: contoh "3TF Aligned BULL + NUCLEAR Whale 15m + Squeeze 5m"]
**Entry:** [Harga] _(Market/Limit — alasan singkat)_
**Stop Loss:** [Harga] _(di bawah/atas Swing Level 5m atau 15m di angka X)_
**TP1:** [Harga] _(~15-30 menit)_ | **TP2:** [Harga] _(~1-2 jam jika momentum lanjut)_
**R:R:** [X:Y]
**🚩 Invalidasi:** [Kondisi spesifik dengan angka — candle close di bawah X, atau RSI 5m tembus Y]

---
Tulis langsung 5 pilihan. Singkat, presisi, actionable. Tidak perlu pembukaan.
"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(1, MAX_RETRIES + 1):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data['candidates'][0]['content']['parts'][0]['text']
                    err = await resp.text()
                    logging.warning(f"Gemini attempt {attempt}: HTTP {resp.status} — {err[:200]}")
            except asyncio.TimeoutError:
                logging.warning(f"Gemini attempt {attempt}: Timeout")
            except Exception as e:
                logging.warning(f"Gemini attempt {attempt}: {e}")
        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY)

    return "⚠️ Gemini API gagal setelah 3x percobaan."


# ============================================================
# TELEGRAM SENDER — v3.2 carry-over (chunking + retry)
# ============================================================

async def send_to_telegram(text: str, header: str = ""):
    url       = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    full_text = header + text
    chunks    = [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]

    async with aiohttp.ClientSession() as session:
        for idx, chunk in enumerate(chunks):
            payload = {
                "chat_id":    TG_CHAT_ID,
                "text":       chunk,
                "parse_mode": "Markdown",
            }
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    async with session.post(
                        url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as resp:
                        if resp.status == 200:
                            if len(chunks) > 1:
                                logging.info(f"Telegram chunk {idx+1}/{len(chunks)} terkirim.")
                            break
                        err = await resp.text()
                        logging.warning(f"Telegram attempt {attempt}: HTTP {resp.status} — {err[:200]}")
                except Exception as e:
                    logging.warning(f"Telegram attempt {attempt}: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
            else:
                logging.error(f"Gagal kirim chunk {idx+1} ke Telegram.")
            if len(chunks) > 1:
                await asyncio.sleep(1)


# ============================================================
# MAIN
# ============================================================

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]):
        logging.error("❌ API Keys belum lengkap!")
        return

    timestamp = datetime.now().strftime("%d %b %Y, %H:%M WIB")
    header = (
        f"⚡ *GOD MODE v4 SCALPER — TOP 5 SETUPS*\n"
        f"🕐 {timestamp} | 4H Anchor + 15m Momentum + 5m Entry\n"
        f"🐳 Smart Money: Whale Z-score + Squeeze Detection\n"
        f"{'─' * 40}\n\n"
    )

    logging.info("🚀 God Mode v4 Scalper dimulai (4H+15m+5m)...")

    try:
        data = await get_high_precision_data()

        if not data:
            logging.error("Tidak ada data valid.")
            await send_to_telegram("⚠️ Scan gagal: tidak ada data valid.")
            return

        logging.info(f"✅ {len(data)} koin final. Mengirim ke Gemini...")
        analysis = await ask_ai_agent(data)
        await send_to_telegram(analysis, header=header)
        logging.info("✅ Laporan scalper dikirim ke Telegram.")

    except Exception as e:
        logging.exception(f"❌ Fatal error: {e}")
        try:
            await send_to_telegram(f"❌ *SYSTEM ERROR*\n`{str(e)[:300]}`")
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
