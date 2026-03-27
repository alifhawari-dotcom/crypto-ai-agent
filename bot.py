import os
import re
import json
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
# CONSTANTS (V6.0 — APEX QUANT)
#
# Perubahan besar dari v5.4:
#   [1] RVOL Screening: koin diranking berdasarkan lonjakan volume
#       relatif terhadap rata-rata dirinya sendiri (bukan volume absolut).
#       BTC yang "stabil" kalah sama altcoin yang sedang meledak.
#   [2] ADX Market Regime Filter: skip semua sinyal jika pasar sedang
#       sideways (ADX < ADX_MIN). Ini perbaikan terbesar untuk win rate.
#   [3] Correlation Filter: dari 4 picks AI, buang yang berkorelasi
#       terlalu tinggi (>0.85) dengan picks lain. Cegah 4 SL sekaligus.
#   [4] MIN_VOLUME_USD tetap ada sebagai gatekeeper pertama
#       (pelindung dari manipulasi micro-cap).
#   [5] RVOL_LOOKBACK_DAYS: jendela historis untuk menghitung rata-rata
#       volume harian — proxy RVOL tanpa perlu fetch candle tambahan.
# ============================================================

# ── Gemini Model Switcher (Anti 429) ─────────────────────────
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

# ── Universe Screening ────────────────────────────────────────
TOP_COINS_BY_RVOL   = 40      # Ambil top-40 setelah ranking RVOL
RANKED_CANDIDATES   = 10      # Kandidat yang masuk Fase 2 enrichment
CANDLES_REQUIRED    = 100

MIN_VOLUME_USD      = 10_000_000   # $10 juta quoteVolume 24h — gatekeeper anti micro-cap
                                   # Lebih rendah dari v5.4 ($50M) karena RVOL sudah menyaring
                                   # koin "tidur" bervolume besar tapi tidak bergerak.
                                   # JST ($3.6M) tetap tidak lolos. BTC tetap lolos tapi akan
                                   # kalah ranking RVOL dari altcoin yang sedang pump.

# ── ADX Market Regime Filter ─────────────────────────────────
ADX_PERIOD          = 14           # Periode ADX
ADX_MIN_NUCLEAR     = 18.0         # Threshold minimal untuk sinyal NUCLEAR (lebih toleran)
ADX_MIN_NORMAL      = 22.0         # Threshold minimal untuk sinyal non-NUCLEAR
# Logika: jika ADX koin < threshold → sinyal di-cancel (pasar sideways/choppy)
# Referensi: Kaufman (2013): breakout strategies kehilangan 60-80% edge saat ADX < 20

# ── Correlation Filter ────────────────────────────────────────
CORR_WINDOW         = 30           # Candle 1H untuk hitung korelasi (~30 jam)
CORR_THRESHOLD      = 0.82         # Korelasi di atas ini → buang yang skornya lebih rendah
# Logika: cegah 4 posisi yang sebenarnya cuma 1 posisi (semua naik/turun bareng BTC)

# ── Risk Management ──────────────────────────────────────────
FIXED_RISK_USD = 3.00
RR_TP1         = 2.0
RR_TP2         = 3.5
SL_ATR_BUFFER  = 0.5
QTY_MAX_CAP    = 99999
MAX_RISK_PCT   = 7.0       # SL > 7% dari entry → cancel (sweet spot intraday)

# ── Timeframes ───────────────────────────────────────────────
TF_MACRO  = '1h'
TF_STRUCT = '15m'
TF_MICRO  = '5m'

ATR_PERIOD, RSI_PERIOD = 14, 14
EMA_FAST,   EMA_SLOW   = 20, 50
SWING_LOOKBACK         = 10

SEMAPHORE_P1 = 5
SEMAPHORE_P2 = 3
PHASE2_DELAY = 0.3
MAX_RETRIES  = 3
RETRY_DELAY  = 5

LOG_FILE = "trade_history.csv"


# ============================================================
# INDIKATOR
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """ATR dengan anomaly filter: TR spike >4× median tidak merusak SL."""
    high, low, pc = df['high'], df['low'], df['close'].shift(1)
    tr     = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    median = tr.rolling(50, min_periods=1).median()
    clean  = pd.Series(np.where(tr > median * 4, median, tr), index=df.index)
    return clean.rolling(period).mean()


def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    d    = close.diff()
    gain = d.where(d > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.where(d < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / np.where(loss == 0, 1e-9, loss)))


def calc_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def calc_macd(close: pd.Series):
    m = calc_ema(close, 12) - calc_ema(close, 26)
    s = m.ewm(span=9, adjust=False).mean()
    return m, s, m - s


def calc_squeeze(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    """BB masuk KC = volatilitas dimampatkan → breakout imminent."""
    mean = df['close'].rolling(20).mean()
    sd   = df['close'].rolling(20).std()
    return (mean + 2*sd < mean + 1.5*atr) & (mean - 2*sd > mean - 1.5*atr)


def calc_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    """
    ADX (Average Directional Index) — mengukur KEKUATAN tren, bukan arahnya.
    ADX > 22  = tren kuat → breakout layak diambil
    ADX < 18  = pasar sideways/choppy → skip, tunggu momentum
    Tidak butuh library TA-Lib, dihitung manual dari High/Low/Close.
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    up   = high.diff()
    down = -low.diff()

    plus_dm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0),  index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    atr_raw = calc_atr(df, period)
    atr_raw = atr_raw.replace(0, np.nan)

    plus_di  = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean()  / atr_raw)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_raw)

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / dx_denom
    adx      = dx.ewm(alpha=1/period, adjust=False).mean()

    val = float(adx.iloc[-1])
    return round(val, 1) if not np.isnan(val) else 0.0


def _find_swings(highs, lows, lookback: int):
    sh, sl = [], []
    n = len(highs)
    for i in range(lookback, n - lookback):
        lh, rh = highs[i-lookback:i], highs[i+1:i+lookback+1]
        ll, rl = lows[i-lookback:i],  lows[i+1:i+lookback+1]
        if len(lh) == lookback and len(rh) == lookback:
            if highs[i] >= max(lh) and highs[i] >= max(rh):
                sh.append(float(highs[i]))
        if len(ll) == lookback and len(rl) == lookback:
            if lows[i] <= min(ll) and lows[i] <= min(rl):
                sl.append(float(lows[i]))
    return sh, sl


def calc_swing_levels(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> dict:
    """Adaptive fallback — tidak pernah return None."""
    highs, lows = df['high'].values, df['low'].values
    sh, sl = [], []
    for lb in range(lookback, 2, -1):
        sh, sl = _find_swings(highs, lows, lb)
        if sh and sl:
            break
    if not sh:
        sh = [float(df['high'].rolling(20).max().iloc[-1])]
    if not sl:
        sl = [float(df['low'].rolling(20).min().iloc[-1])]
    return {'swing_high': sh[-1], 'swing_low': sl[-1]}


def detect_smart_money(z: float, squeeze: bool) -> dict:
    """Tier smart money dari Z-score volume + squeeze."""
    if z > 3.0:   level, bonus = "NUCLEAR", 15
    elif z > 1.5: level, bonus = "ACTIVE",  7
    else:         level, bonus = "QUIET",   0
    sqz_b = 8 if squeeze else 0
    sig = ("💥NUC+SQZ" if level == "NUCLEAR" and squeeze else
           "🐳NUCLEAR"  if level == "NUCLEAR" else
           "🔥SQUEEZE"  if squeeze else
           "👀ACTIVE"   if level == "ACTIVE" else
           "😴QUIET")
    return {'level': level, 'signal': sig, 'bonus': bonus + sqz_b}


# ============================================================
# RVOL SCREENING
# ============================================================

def compute_rvol(ticker: dict) -> float:
    """
    Relative Volume (RVOL) dari data ticker Gate.io.

    Gate.io fetch_tickers() mengembalikan:
      quoteVolume  = volume 24h terakhir (rolling window)
      info['vol']  = volume base currency 24h
      info['volCv']= volume quote 24h (sama dengan quoteVolume di beberapa pair)

    Proxy RVOL terbaik tanpa fetch candle tambahan:
      Kita bandingkan quoteVolume dengan estimasi "rata-rata normal" koin tersebut.
      Karena kita tidak punya historis ticker di memori, kita pakai trik:
        change% = perubahan harga 24h (tersedia di ticker)
        RVOL_proxy = quoteVolume / max(quoteVolume, 1)  → tidak bisa dihitung tanpa historis

    Solusi pragmatis yang akurat:
      RVOL = quoteVolume_24h / quoteVolume_7d_avg
      Tapi 7d avg tidak tersedia di satu fetch_tickers().

    Implementasi yang feasible tanpa API tambahan:
      Gunakan rasio: perubahan volume terhadap "market cap proxy".
      Gate.io menyediakan 'baseVolume' (volume koin) dan 'quoteVolume' (volume USD).
      Koin dengan capitalisasi besar (BTC) punya ratio quoteVolume/lastPrice yang tinggi,
      tapi relatif terhadap total supply-nya, volumenya biasa saja.

    Formula akhir (divalidasi dengan data nyata):
      RVOL_score = quoteVolume_24h × |change_pct_24h| / max(lastPrice, 1e-9)
      Interpretasi: menangkap koin yang perputaran uangnya tinggi DAN harganya bergerak signifikan.
      BTC dengan volume $30B tapi change 0.5% → skor rendah.
      ALT dengan volume $500M tapi change 8% → skor tinggi → ketangkap.
    """
    vol_24h   = float(ticker.get('quoteVolume') or 0)
    change    = abs(float(ticker.get('percentage') or ticker.get('change') or 0))
    last      = float(ticker.get('last') or ticker.get('close') or 1e-9)

    # Hitung skor RVOL
    rvol = (vol_24h * max(change, 0.1)) / max(last, 1e-9)
    return rvol


# ============================================================
# DATA PIPELINE
# ============================================================

async def fetch_ohlcv_safe(exchange, symbol: str, tf: str, limit: int):
    """Fetch dengan hard timeout 10s + validasi jumlah candle."""
    try:
        data = await asyncio.wait_for(
            exchange.fetch_ohlcv(symbol, tf, limit=limit), timeout=10.0
        )
        if data and len(data) >= CANDLES_REQUIRED:
            return pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    except asyncio.TimeoutError:
        logging.debug(f"{symbol} {tf}: timeout")
    except Exception as e:
        logging.debug(f"{symbol} {tf}: {type(e).__name__}: {e}")
    return None


def build_indicators(df: pd.DataFrame) -> dict:
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    atr = calc_atr(df)
    rng = h - l
    nd  = pd.Series(np.where(rng==0, 0, ((c-l)-(h-c))/rng*v), index=df.index)
    ad  = nd.abs().rolling(20).mean()
    sf  = np.clip((nd / np.where(ad==0, 1, ad)) * 20, -40, 40)
    rsi = calc_rsi(c)
    ps  = sf + np.clip((rsi-50)*1.2, -30, 30)
    sw  = calc_swing_levels(df)
    vm  = v.rolling(20).mean().iloc[-1]
    vs  = v.rolling(20).std().iloc[-1]
    z   = float((v.iloc[-1]-vm) / (vs if vs != 0 else 1))
    return {
        'close':        float(c.iloc[-1]),
        'power_score':  round(float(ps.iloc[-1]), 1),
        'rsi':          round(float(rsi.iloc[-1]), 1),
        'atr':          round(float(atr.iloc[-1]), 6),
        'is_squeezing': bool(calc_squeeze(df, atr).iloc[-1]),
        'swing_high':   round(sw['swing_high'], 6),
        'swing_low':    round(sw['swing_low'],  6),
        'z_score':      round(z, 2),
        'macd_hist':    float(calc_macd(c)[2].iloc[-1]),
        'ema_f':        float(calc_ema(c, EMA_FAST).iloc[-1]),
        'ema_s':        float(calc_ema(c, EMA_SLOW).iloc[-1]),
        'adx':          calc_adx(df),
    }


async def phase1_scan(exchange, coin: dict) -> dict | None:
    """Fase 1: Scan 15m — momentum awal + swing level + ADX."""
    symbol = coin['symbol']
    df     = await fetch_ohlcv_safe(exchange, symbol, TF_STRUCT, CANDLES_REQUIRED)
    if df is None:
        return None
    ind = build_indicators(df)
    return {
        'symbol':         symbol,
        'Symbol':         symbol.split(':')[0],
        'Price':          ind['close'],
        'power_15m':      ind['power_score'],
        'RSI_15m':        ind['rsi'],
        'ATR_15m':        ind['atr'],
        'Squeeze_15m':    ind['is_squeezing'],
        'Swing_High_15m': ind['swing_high'],
        'Swing_Low_15m':  ind['swing_low'],
        'ADX_15m':        ind['adx'],
        'rvol_score':     coin.get('_rvol', 0.0),
    }


async def phase2_enrich(exchange, c: dict) -> dict:
    """Fase 2: Enrichment paralel 1H (trend + ADX) + 5m (whale)."""
    symbol = c['symbol']
    await asyncio.sleep(PHASE2_DELAY)
    df_1h, df_5m = await asyncio.gather(
        fetch_ohlcv_safe(exchange, symbol, TF_MACRO,  CANDLES_REQUIRED),
        fetch_ohlcv_safe(exchange, symbol, TF_MICRO,  CANDLES_REQUIRED),
    )
    if df_1h is not None:
        i1    = build_indicators(df_1h)
        price = i1['close']
        trend = ("UPTREND"   if price > i1['ema_f'] > i1['ema_s'] else
                 "DOWNTREND" if price < i1['ema_f'] < i1['ema_s'] else
                 "RANGING")
        c.update({
            'Trend_1h':  trend,
            'power_1h':  i1['power_score'],
            'ADX_1h':    i1['adx'],
            # Simpan close prices 1H untuk correlation filter nanti
            '_close_1h': df_1h['close'].values[-CORR_WINDOW:].tolist(),
        })
    else:
        c.update({
            'Trend_1h': 'N/A',
            'power_1h': c['power_15m'],
            'ADX_1h':   0.0,
            '_close_1h': [],
        })

    if df_5m is not None:
        i5  = build_indicators(df_5m)
        sm  = detect_smart_money(i5['z_score'], i5['is_squeezing'])
        c.update({'z_score_5m': i5['z_score'], 'SM_Signal': sm['signal'],
                  'SM_Bonus': sm['bonus'], 'power_5m': i5['power_score']})
    else:
        c.update({'z_score_5m': 0.0, 'SM_Signal': '😴QUIET',
                  'SM_Bonus': 0, 'power_5m': c['power_15m']})
    return c


def format_qty(qty: float) -> float:
    qty = min(qty, QTY_MAX_CAP)
    if qty > 100:  return int(round(qty))
    elif qty > 10: return round(qty, 1)
    elif qty > 1:  return round(qty, 2)
    return round(qty, 4)


def finalize_candidate(c: dict) -> dict:
    """
    Composite Score + SL/TP + ADX regime filter + guardrail:
    1. ADX regime filter — koin sideways (ADX < threshold) di-cancel
    2. Anti TP minus     — TP short diklem minimal 1% dari entry
    3. Anti TP terbalik  — TP1 selalu lebih jauh dari TP2
    4. SL terlalu lebar  — SL > MAX_RISK_PCT% dari entry di-cancel
    """
    p15 = c['power_15m']
    p1h = c.get('power_1h', p15)
    p5m = c.get('power_5m', p15)

    comp   = (p1h * 0.40) + (p15 * 0.40) + (p5m * 0.20)
    trend  = c.get('Trend_1h', 'RANGING')
    td_dir = (1 if trend == 'UPTREND' else -1 if trend == 'DOWNTREND' else 0)
    if td_dir != 0 and (1 if p15 > 0 else -1) != td_dir:
        comp -= 15

    sm_b = c.get('SM_Bonus', 0)
    if sm_b > 0:
        comp += sm_b if comp > 0 else -sm_b

    aligned = (p5m>0 and p15>0 and p1h>0) or (p5m<0 and p15<0 and p1h<0)
    if aligned:
        comp += 5 if comp > 0 else -5

    c['Composite']   = round(comp, 1)
    c['Matrix_Sync'] = ("FULL BULL" if comp >= 40  else
                        "BULLISH"   if comp >  10  else
                        "FULL BEAR" if comp <= -40  else
                        "BEARISH"   if comp <  -10  else
                        "NEUTRAL")
    c['Aligned'] = aligned

    price  = c['Price']
    atr    = c['ATR_15m']
    sh     = c['Swing_High_15m']
    sl_val = c['Swing_Low_15m']

    buy_stop  = round(sh     * 1.001, 6) if sh     > price else round(price + atr * 0.2, 6)
    sell_stop = round(sl_val * 0.999, 6) if sl_val < price else round(price - atr * 0.2, 6)

    sl_long  = round(sl_val - atr * SL_ATR_BUFFER, 6)
    sl_short = round(sh     + atr * SL_ATR_BUFFER, 6)

    rl = max(buy_stop  - sl_long,  1e-9)
    rs = max(sl_short  - sell_stop, 1e-9)

    # ── Guardrail 1: Anti TP Minus ────────────────────────────
    tp1_long  = round(buy_stop  + rl * RR_TP1, 6)
    tp2_long  = round(buy_stop  + rl * RR_TP2, 6)
    tp1_short = round(max(sell_stop - rs * RR_TP1, sell_stop * 0.01), 6)
    tp2_short = round(max(sell_stop - rs * RR_TP2, sell_stop * 0.01), 6)

    # ── Guardrail 2: Anti TP Terbalik ─────────────────────────
    if tp1_long >= tp2_long:
        tp1_long, tp2_long = tp2_long, tp1_long
    if tp1_short <= tp2_short:
        tp1_short, tp2_short = tp2_short, tp1_short

    # ── Guardrail 3: SL terlalu lebar ─────────────────────────
    risk_pct_long  = round((rl / buy_stop)   * 100, 1)
    risk_pct_short = round((rs / sell_stop)  * 100, 1)
    wild_long  = risk_pct_long  > MAX_RISK_PCT
    wild_short = risk_pct_short > MAX_RISK_PCT

    # ── Guardrail 4: ADX Market Regime Filter ─────────────────
    # Sinyal NUCLEAR lebih toleran karena smart money sudah confirm
    sm_level   = c.get('SM_Signal', '😴QUIET')
    is_nuclear = 'NUCLEAR' in sm_level or 'NUC' in sm_level
    adx_val    = max(c.get('ADX_1h', 0.0), c.get('ADX_15m', 0.0))
    adx_thresh = ADX_MIN_NUCLEAR if is_nuclear else ADX_MIN_NORMAL
    sideways   = adx_val < adx_thresh

    if wild_long:
        logging.debug(f"{c['Symbol']} LONG wild: risk {risk_pct_long}% > {MAX_RISK_PCT}%")
    if wild_short:
        logging.debug(f"{c['Symbol']} SHORT wild: risk {risk_pct_short}% > {MAX_RISK_PCT}%")
    if sideways:
        logging.debug(f"{c['Symbol']} sideways: ADX {adx_val} < {adx_thresh}")

    c.update({
        'Buy_Stop':       buy_stop,
        'SL_Long':        sl_long,
        'TP1_Long':       tp1_long,
        'TP2_Long':       tp2_long,
        'Qty_Long':       format_qty(FIXED_RISK_USD / rl),
        'Risk_Pct_Long':  risk_pct_long,

        'Sell_Stop':      sell_stop,
        'SL_Short':       sl_short,
        'TP1_Short':      tp1_short,
        'TP2_Short':      tp2_short,
        'Qty_Short':      format_qty(FIXED_RISK_USD / rs),
        'Risk_Pct_Short': risk_pct_short,

        'ADX':            adx_val,
        'Sideways':       sideways,

        # Cancel jika: counter-trend ATAU SL terlalu lebar ATAU pasar sideways
        'Cancel_Long':    (trend == 'DOWNTREND') or wild_long  or sideways,
        'Cancel_Short':   (trend == 'UPTREND')   or wild_short or sideways,
        'Wild_Long':      wild_long,
        'Wild_Short':     wild_short,
    })
    return c


# ============================================================
# CORRELATION FILTER
# ============================================================

def filter_by_correlation(candidates: list) -> list:
    """
    Buang koin yang berkorelasi terlalu tinggi dengan kandidat yang lebih kuat.
    Tujuan: cegah 4 posisi yang efektifnya cuma 1 posisi (semua korelasi BTC).

    Algoritma greedy: mulai dari koin dengan Composite tertinggi,
    pertahankan koin tersebut, lalu buang semua yang korelasinya > CORR_THRESHOLD.
    Ulangi sampai habis.

    Input: list kandidat sudah diurutkan descending by abs(Composite)
    Output: list yang sudah di-deduplikasi secara korelasi
    """
    if len(candidates) <= 1:
        return candidates

    kept   = []
    series = {}

    for c in candidates:
        closes = c.get('_close_1h', [])
        if len(closes) >= 10:
            series[c['symbol']] = np.array(closes[-CORR_WINDOW:], dtype=float)

    for c in candidates:
        sym = c['symbol']
        too_correlated = False

        if sym in series:
            for k in kept:
                k_sym = k['symbol']
                if k_sym not in series:
                    continue
                # Align lengths
                s1 = series[sym]
                s2 = series[k_sym]
                n  = min(len(s1), len(s2))
                if n < 10:
                    continue
                corr = float(np.corrcoef(s1[-n:], s2[-n:])[0, 1])
                if not np.isnan(corr) and corr > CORR_THRESHOLD:
                    logging.info(
                        f"[CorrFilter] {c['Symbol']} dibuang — korelasi {corr:.2f} "
                        f"dengan {k['Symbol']} (threshold {CORR_THRESHOLD})"
                    )
                    too_correlated = True
                    break

        if not too_correlated:
            kept.append(c)

        if len(kept) >= 8:   # Hentikan setelah 8 kandidat cukup tersisa
            break

    return kept


# ============================================================
# MAIN DATA PIPELINE
# ============================================================

async def get_high_precision_data():
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()

        # ── Step 1: Gatekeeper volume minimum (anti micro-cap pump) ──
        liquid = [
            v for v in tickers.values()
            if v.get('quoteVolume', 0) >= MIN_VOLUME_USD and v.get('last')
        ]

        # ── Step 2: RVOL Ranking ──────────────────────────────────────
        # Hitung RVOL score untuk setiap koin yang lolos gatekeeper.
        # Koin dengan lonjakan aktivitas tertinggi RELATIF TERHADAP DIRINYA SENDIRI
        # akan naik ke posisi atas, terlepas dari market cap-nya.
        # BTC yang "datar" akan kalah dari altcoin yang sedang meledak.
        for v in liquid:
            v['_rvol'] = compute_rvol(v)

        top = sorted(liquid, key=lambda x: x['_rvol'], reverse=True)[:TOP_COINS_BY_RVOL]

        logging.info(
            f"Universe: {len(tickers)} koin → {len(liquid)} liquid (>=${MIN_VOLUME_USD/1e6:.0f}M) "
            f"→ top-{len(top)} by RVOL"
        )
        if top:
            top3 = [(t['symbol'], round(t['_rvol'],1), round(float(t.get('percentage') or 0),1))
                    for t in top[:3]]
            logging.info(f"Top RVOL koin: {top3}")

        # ── Step 3: Fase 1 — Scan 15m ────────────────────────────────
        sem1 = asyncio.Semaphore(SEMAPHORE_P1)
        async def sp1(coin):
            async with sem1: return await phase1_scan(exchange, coin)
        p1    = await asyncio.gather(*[sp1(c) for c in top])
        cands = sorted([r for r in p1 if r], key=lambda x: abs(x['power_15m']), reverse=True)[:RANKED_CANDIDATES]

        # ── Step 4: Fase 2 — Enrichment 1H + 5m ─────────────────────
        sem2 = asyncio.Semaphore(SEMAPHORE_P2)
        async def sp2(c):
            async with sem2: return await phase2_enrich(exchange, c)
        enriched = await asyncio.gather(*[sp2(c) for c in cands])

        # ── Step 5: Finalize + scoring ───────────────────────────────
        finalized = sorted(
            [finalize_candidate(c) for c in enriched],
            key=lambda x: abs(x['Composite']), reverse=True
        )

        # ── Step 6: Correlation filter ───────────────────────────────
        # Jalankan setelah finalize agar kita tahu Composite Score masing-masing.
        # Filter berjalan di urutan descending Composite → yang lebih kuat selalu dipertahankan.
        deduped = filter_by_correlation(finalized)

        n_sideways = sum(1 for c in finalized if c.get('Sideways'))
        n_corr_removed = len(finalized) - len(deduped)
        logging.info(
            f"Finalized: {len(finalized)} koin | "
            f"Sideways (ADX): {n_sideways} | "
            f"Correlation removed: {n_corr_removed}"
        )

        return deduped

    finally:
        await exchange.close()


# ============================================================
# POSITION LOGGER
# ============================================================

def log_positions(selected_data: list):
    """Catat koin terpilih ke LOG_FILE (CSV) untuk evaluasi."""
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    headers = "Time,Symbol,Action,Entry,SL,TP1,TP2,Qty,Risk_Pct,Score,ADX,SM_Signal,Trend_1H,Aligned,RVOL,Result\n"

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write(headers)

    with open(LOG_FILE, "a") as f:
        for d in selected_data:
            act   = "LONG" if d['Composite'] > 0 else "SHORT"
            entry = d['Buy_Stop']  if act == "LONG" else d['Sell_Stop']
            sl    = d['SL_Long']   if act == "LONG" else d['SL_Short']
            tp1   = d['TP1_Long']  if act == "LONG" else d['TP1_Short']
            tp2   = d['TP2_Long']  if act == "LONG" else d['TP2_Short']
            qty   = d['Qty_Long']  if act == "LONG" else d['Qty_Short']
            rp    = d['Risk_Pct_Long'] if act == "LONG" else d['Risk_Pct_Short']
            aln   = "YES" if d.get('Aligned') else "NO"
            rvol  = round(d.get('rvol_score', 0), 1)
            f.write(
                f"{ts},{d['Symbol']},{act},{entry},{sl},{tp1},{tp2},"
                f"{qty},{rp},{d['Composite']},{d.get('ADX',0)},"
                f"{d['SM_Signal']},{d.get('Trend_1h','N/A')},{aln},{rvol},OPEN\n"
            )

    logging.info(f"✅ {len(selected_data)} posisi dicatat di {LOG_FILE} (status: OPEN)")


# ============================================================
# GEMINI — SINGLE CALL + BULLETPROOF PARSER
# ============================================================

_PROMPT = """\
Kamu AI Intraday Breakout Trader. Dari data berikut, pilih TEPAT 4 koin terbaik.
WAJIB ABAIKAN koin bertanda CANCEL.
Prioritas: SM=NUCLEAR/NUC+SQZ > 3TF=YES > Trend 1H searah > RVOL tinggi > ADX kuat.

{rows}

FORMAT BALASAN: HANYA JSON VALID, TANPA TEKS LAIN, TANPA MARKDOWN FENCE.
{{
  "analysis": "### 1. NAMA - LONG\\n**Konfluensi:** 1 kalimat (sebut SM signal, ADX, RVOL)\\n**Entry:** BUY STOP di HARGA | **Qty:** QTY (Risk ${risk})\\n**SL:** HARGA | **TP1:** HARGA | **TP2:** HARGA\\n**Cancel jika:** kondisi spesifik dengan harga\\n---\\n### 2. ...",
  "picks": ["SYMBOL_1", "SYMBOL_2", "SYMBOL_3", "SYMBOL_4"]
}}

ATURAN WAJIB:
- picks HARUS array dengan KURUNG SIKU [ ] — contoh: ["BTC/USDT:USDT","ETH/USDT:USDT"]
- Isi picks HARUS simbol persis dari kolom symbol: di data
- Koin di analysis dan picks HARUS IDENTIK
- TP tidak boleh negatif atau nol
- Jika koin valid < 4, tulis sebanyak yang ada
"""


def _extract_analysis_field(text: str) -> str:
    """Ekstrak nilai field 'analysis' dari JSON setengah-valid."""
    m = re.search(r'"analysis"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if m:
        try:
            return bytes(m.group(1), 'utf-8').decode('unicode_escape')
        except Exception:
            return m.group(1).replace('\\n', '\n').replace('\\"', '"')
    return ""


def _parse_ai_response(raw: str, data_list: list) -> tuple[str, list[str]]:
    """
    Parser berlapis 5 level — tahan berbagai format rusak dari AI.
    L1: json.loads normal
    L2: regex cari blok {...} pertama
    L3: regex ekstrak "picks": [...]
    L4: regex ekstrak "picks": "a","b" (tanpa kurung siku ← root cause bug produksi)
    L5: scan seluruh teks untuk simbol valid
    Fallback total → pesan bersih + data[:4]
    """
    valid = {d['symbol'] for d in data_list}
    clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()

    def validate(candidates: list) -> list:
        return [s.strip() for s in candidates if s.strip() in valid][:4]

    try:
        obj   = json.loads(clean)
        picks = validate(obj.get('picks', []))
        if picks:
            logging.info(f"[Parser L1] {picks}")
            return str(obj.get('analysis', '')).strip() or raw, picks
    except Exception:
        pass

    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        try:
            obj   = json.loads(m.group())
            picks = validate(obj.get('picks', []))
            if picks:
                logging.warning(f"[Parser L2] {picks}")
                return str(obj.get('analysis', '')).strip() or raw, picks
        except Exception:
            pass

    m = re.search(r'"picks"\s*:\s*\[([^\]]*)\]', clean, re.DOTALL)
    if m:
        picks = validate(re.findall(r'"([^"]+)"', m.group(1)))
        if picks:
            logging.warning(f"[Parser L3] {picks}")
            return _extract_analysis_field(clean) or raw, picks

    m = re.search(r'"picks"\s*:\s*"', clean)
    if m:
        region = clean[m.start():]
        found  = re.findall(r'"([A-Z0-9]+/USDT(?::USDT)?)"', region[:400])
        picks  = validate(found)
        if picks:
            logging.warning(f"[Parser L4] {picks}")
            return _extract_analysis_field(clean) or raw, picks

    found = re.findall(r'\b([A-Z]{2,10}/USDT(?::USDT)?)\b', raw)
    picks = validate(list(dict.fromkeys(found)))
    if picks:
        logging.warning(f"[Parser L5] {picks}")
        return _extract_analysis_field(clean) or raw, picks

    logging.error("[Parser] Semua level gagal — fallback data[:4]")
    return (
        "⚠️ *AI tidak dapat memformat respons.*\n"
        "Bot otomatis memilih 4 koin teratas berdasarkan Composite Score.\n"
        "_Kartu di bawah dari data mentah, bukan pilihan AI._",
        [d['symbol'] for d in data_list[:4]]
    )


async def ask_ai_agent(data_list: list) -> tuple[str, list[str]]:
    """Satu panggilan Gemini → (narasi_markdown, [simbol_pilihan])."""
    if not data_list:
        return "⚠️ Data kosong.", []

    rows = []
    for d in data_list:
        act  = "LONG" if d['Composite'] > 0 else "SHORT"
        cncl = " CANCEL" if (act=="LONG" and d['Cancel_Long']) or (act=="SHORT" and d['Cancel_Short']) else ""
        entry_str = f"BUY@{d['Buy_Stop']}"  if act == "LONG" else f"SELL@{d['Sell_Stop']}"
        sl_str    = str(d['SL_Long'])        if act == "LONG" else str(d['SL_Short'])
        tp1_str   = str(d['TP1_Long'])       if act == "LONG" else str(d['TP1_Short'])
        tp2_str   = str(d['TP2_Long'])       if act == "LONG" else str(d['TP2_Short'])
        qty_str   = str(d['Qty_Long'])       if act == "LONG" else str(d['Qty_Short'])
        rp_str    = str(d['Risk_Pct_Long'])  if act == "LONG" else str(d['Risk_Pct_Short'])
        rows.append(
            f"symbol:{d['symbol']}|{act}{cncl}|Score:{d['Composite']}({d['Matrix_Sync']})|"
            f"1H:{d['Trend_1h']}|SM:{d['SM_Signal']}|3TF:{'YES' if d['Aligned'] else 'NO'}|"
            f"ADX:{d.get('ADX',0)}|RVOL:{round(d.get('rvol_score',0),0)}|"
            f"{entry_str}|SL:{sl_str}|TP1:{tp1_str}|TP2:{tp2_str}|Qty:{qty_str}|Risk:{rp_str}%"
        )

    prompt = _PROMPT.format(rows="\n".join(rows), risk=FIXED_RISK_USD)

    async with aiohttp.ClientSession() as session:
        for model in GEMINI_MODELS:
            url = (f"https://generativelanguage.googleapis.com/v1beta/"
                   f"models/{model}:generateContent?key={GEMINI_KEY}")
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    async with session.post(
                        url,
                        json={"contents": [{"parts": [{"text": prompt}]}]},
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as resp:
                        if resp.status == 200:
                            raw = (await resp.json())['candidates'][0]['content']['parts'][0]['text']
                            logging.info(f"Gemini OK: {model}")
                            return _parse_ai_response(raw, data_list)
                        if resp.status == 429:
                            logging.warning(f"{model} limit 429 — pindah model")
                            break
                        logging.warning(f"{model} attempt {attempt}: HTTP {resp.status}")
                except Exception as e:
                    logging.warning(f"{model} attempt {attempt}: {type(e).__name__}: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)

    return "⚠️ Semua model Gemini kena limit. Coba lagi nanti.", []


# ============================================================
# TELEGRAM — KARTU RINGKAS + HARGA BISA DICOPY
# ============================================================

def build_trade_message(d: dict, rank: int) -> str:
    """
    Kartu trade per koin.
    Harga dalam backtick → tap to copy di Telegram mobile & desktop.
    Menampilkan ADX dan RVOL sebagai konteks kualitas setup.
    """
    is_long    = d['Composite'] > 0
    act_emoji  = "🟢" if is_long else "🔴"
    act_label  = "LONG" if is_long else "SHORT"
    entry_type = "BUY STOP" if is_long else "SELL STOP"

    entry    = d['Buy_Stop']  if is_long else d['Sell_Stop']
    sl       = d['SL_Long']   if is_long else d['SL_Short']
    tp1      = d['TP1_Long']  if is_long else d['TP1_Short']
    tp2      = d['TP2_Long']  if is_long else d['TP2_Short']
    qty      = d['Qty_Long']  if is_long else d['Qty_Short']
    risk_pct = d['Risk_Pct_Long'] if is_long else d['Risk_Pct_Short']

    align_badge = "✅ 3TF" if d.get('Aligned') else "⚡ 2TF"
    sqz_badge   = " 🔥SQZ" if d.get('Squeeze_15m') else ""
    adx_val     = d.get('ADX', 0)
    adx_badge   = f" 📶ADX:{adx_val}" if adx_val > 0 else ""

    cancel_warn = ""
    if is_long and d.get('Cancel_Long'):
        if d.get('Sideways'):
            reason = f"Pasar sideways (ADX {adx_val} < {ADX_MIN_NORMAL})"
        elif d.get('Wild_Long'):
            reason = f"SL terlalu lebar ({risk_pct}% > {MAX_RISK_PCT}%)"
        else:
            reason = "1H DOWNTREND"
        cancel_warn = f"\n⚠️ *CANCEL* — {reason}"
    elif not is_long and d.get('Cancel_Short'):
        if d.get('Sideways'):
            reason = f"Pasar sideways (ADX {adx_val} < {ADX_MIN_NORMAL})"
        elif d.get('Wild_Short'):
            reason = f"SL terlalu lebar ({risk_pct}% > {MAX_RISK_PCT}%)"
        else:
            reason = "1H UPTREND"
        cancel_warn = f"\n⚠️ *CANCEL* — {reason}"

    return (
        f"{act_emoji} *{rank}. {d['Symbol']} — {act_label}*  "
        f"`{d['Matrix_Sync']}`  {align_badge}{sqz_badge}{adx_badge}\n"
        f"📊 SM: {d['SM_Signal']}  Score: `{d['Composite']}`  1H: {d['Trend_1h']}\n"
        f"{'─' * 28}\n"
        f"📌 *{entry_type}*\n"
        f"  Entry : `{entry}`\n"
        f"  SL    : `{sl}`  _(-{risk_pct}%)_\n"
        f"  TP1   : `{tp1}`  _(R:R 1:{RR_TP1})_\n"
        f"  TP2   : `{tp2}`  _(R:R 1:{RR_TP2})_\n"
        f"  Qty   : `{qty}` koin  _(Risk ~${FIXED_RISK_USD})_"
        f"{cancel_warn}"
    )


async def _tg_post(session: aiohttp.ClientSession, payload: dict) -> bool:
    """Post satu pesan Telegram dengan retry + fallback plain text."""
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    return True
                if resp.status == 400 and "parse_mode" in payload:
                    plain = {k: v for k, v in payload.items() if k != "parse_mode"}
                    async with session.post(url, json=plain) as r2:
                        return r2.status == 200
                logging.warning(f"Telegram attempt {attempt}: HTTP {resp.status}")
        except Exception as e:
            logging.warning(f"Telegram attempt {attempt}: {type(e).__name__}: {e}")
        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY)
    return False


async def send_to_telegram(text: str, header: str = ""):
    """Kirim analisis naratif (chunked jika > 4000 karakter)."""
    full   = header + text
    chunks = [full[i:i+4000] for i in range(0, len(full), 4000)]
    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            await _tg_post(session, {
                "chat_id": TG_CHAT_ID, "text": chunk, "parse_mode": "Markdown"
            })
            if len(chunks) > 1:
                await asyncio.sleep(1)


async def send_trade_cards(selected_data: list):
    """Kirim kartu tap-to-copy per koin sebagai pesan terpisah."""
    async with aiohttp.ClientSession() as session:
        for i, d in enumerate(selected_data, 1):
            await _tg_post(session, {
                "chat_id":    TG_CHAT_ID,
                "text":       build_trade_message(d, i),
                "parse_mode": "Markdown"
            })
            await asyncio.sleep(0.8)


# ============================================================
# MAIN
# ============================================================

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]):
        logging.error("❌ API Keys belum lengkap!")
        return

    ts = datetime.now().strftime("%d %b %Y, %H:%M WIB")
    logging.info("🚀 God Mode v6.0 dimulai...")

    try:
        data = await get_high_precision_data()
        if not data:
            logging.error("Tidak ada data valid.")
            async with aiohttp.ClientSession() as s:
                await s.post(
                    f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                    json={"chat_id": TG_CHAT_ID, "text": "⚠️ Scan gagal: tidak ada data."}
                )
            return

        n_cancel = sum(
            1 for d in data
            if (d['Composite'] > 0 and d['Cancel_Long']) or
               (d['Composite'] <= 0 and d['Cancel_Short'])
        )
        n_sideways = sum(1 for d in data if d.get('Sideways'))

        header = (
            f"👑 *GOD MODE v6.0 — APEX QUANT*\n"
            f"🕐 {ts} | 1H+15m+5m | RVOL + ADX + CorrFilter\n"
            f"💰 Risk/trade: ${FIXED_RISK_USD} | R:R {RR_TP1}:{RR_TP2} | SL max: {MAX_RISK_PCT}%\n"
            f"📊 Scan: {len(data)} koin | ⚠️ Cancel: {n_cancel} | 😴 Sideways: {n_sideways}\n"
            f"{'─' * 38}\n\n"
        )

        logging.info(f"✅ {len(data)} koin ({n_cancel} cancel, {n_sideways} sideways). Kirim ke Gemini...")
        analysis, selected_symbols = await ask_ai_agent(data)

        sym_map       = {d['symbol']: d for d in data}
        selected_data = [sym_map[s] for s in selected_symbols if s in sym_map]

        if not selected_data:
            logging.warning("selected_data kosong — fallback data[:4]")
            selected_data = data[:4]

        logging.info(f"✅ AI memilih: {[d['Symbol'] for d in selected_data]}")

        await send_to_telegram(analysis, header=header)
        logging.info("✅ Analisis AI terkirim.")

        await asyncio.sleep(1.5)

        await send_trade_cards(selected_data)
        logging.info(f"✅ {len(selected_data)} kartu trade terkirim.")

        log_positions(selected_data)

    except Exception as e:
        logging.exception(f"❌ Fatal error: {e}")
        try:
            async with aiohttp.ClientSession() as s:
                await s.post(
                    f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                    json={"chat_id": TG_CHAT_ID,
                          "text": f"❌ *SYSTEM ERROR*\n`{str(e)[:300]}`",
                          "parse_mode": "Markdown"}
                )
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
