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
TOP_COINS_BY_VOLUME = 40    # Pool awal (sama seperti v1, aman)
RANKED_CANDIDATES   = 12    # Kandidat ke AI — cukup tanpa spam token
CANDLES_REQUIRED    = 100   # Minimal candle untuk indikator akurat

ATR_PERIOD   = 14
RSI_PERIOD   = 14
EMA_FAST     = 20
EMA_SLOW     = 50
EMA_TREND    = 200
SWING_LOOKBACK = 20         # Lookback candle untuk deteksi Swing High/Low

# ============================================================
# FIX #1 — RATE LIMIT STRATEGY
# Solusi: Fetch 1H saja untuk scoring awal (ringan),
# lalu baru fetch 4H HANYA untuk Top-N kandidat terkuat.
# Total request jauh berkurang: 40 (1H) + 12 (4H) = 52 request
# vs v2 yang 150 request (50 koin × 3 TF).
# ============================================================
SEMAPHORE_PHASE1 = 5   # Fase 1: scan 40 koin @ 1H — ringan
SEMAPHORE_PHASE2 = 3   # Fase 2: enrichment 12 koin @ 4H — lebih berhati-hati
PHASE2_DELAY     = 0.3 # Detik jeda antar request fase 2 (hindari burst)

MAX_RETRIES  = 3
RETRY_DELAY  = 5

# ============================================================
# UTILITY: INDIKATOR
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, prev_close = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


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


# ============================================================
# FIX #3 — MARKET STRUCTURE: SWING HIGH / SWING LOW
# Deteksi level harga nyata tempat price cenderung reaction.
# Lebih akurat dari indikator lagging untuk entry/SL/TP.
# ============================================================

def calc_swing_levels(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> dict:
    """
    Deteksi Swing High dan Swing Low terbaru dari price action.
    Swing High = candle dengan high tertinggi dalam window lookback di kiri & kanan.
    Swing Low  = candle dengan low terendah dalam window lookback di kiri & kanan.
    Return level harga terbaru yang valid.
    """
    highs = df['high'].values
    lows  = df['low'].values
    n     = len(df)

    swing_highs = []
    swing_lows  = []

    # Scan dari candle ke-lookback sampai candle ke-(n-lookback-1)
    # Ini memastikan ada cukup candle di kiri dan kanan untuk konfirmasi
    for i in range(lookback, n - lookback):
        left_h  = highs[i - lookback : i]
        right_h = highs[i + 1 : i + lookback + 1]
        if highs[i] == max(left_h) and highs[i] == max(right_h):
            swing_highs.append((i, highs[i]))

        left_l  = lows[i - lookback : i]
        right_l = lows[i + 1 : i + lookback + 1]
        if lows[i] == min(left_l) and lows[i] == min(right_l):
            swing_lows.append((i, lows[i]))

    # Ambil Swing High & Low terbaru (paling dekat ke harga sekarang)
    latest_sh = swing_highs[-1][1] if swing_highs else None
    latest_sl = swing_lows[-1][1]  if swing_lows  else None

    # Ambil juga yang kedua terbaru sebagai struktur tambahan
    prev_sh = swing_highs[-2][1] if len(swing_highs) >= 2 else latest_sh
    prev_sl = swing_lows[-2][1]  if len(swing_lows)  >= 2 else latest_sl

    return {
        'swing_high':      latest_sh,
        'swing_low':       latest_sl,
        'prev_swing_high': prev_sh,
        'prev_swing_low':  prev_sl,
    }


def calc_structure_bias(price: float, swing_high: float, swing_low: float) -> str:
    """
    Tentukan posisi harga relatif terhadap struktur.
    Breakout di atas Swing High = bullish structure.
    Breakdown di bawah Swing Low = bearish structure.
    """
    if swing_high is None or swing_low is None:
        return "UNKNOWN"
    range_size = swing_high - swing_low
    if range_size == 0:
        return "UNKNOWN"
    position = (price - swing_low) / range_size  # 0.0 = di bawah SL, 1.0 = di atas SH
    if price > swing_high:
        return "BREAKOUT"       # Di atas resistance — bullish kuat
    elif price < swing_low:
        return "BREAKDOWN"      # Di bawah support — bearish kuat
    elif position > 0.65:
        return "NEAR_RESISTANCE"  # Mendekati resistance, hati-hati long
    elif position < 0.35:
        return "NEAR_SUPPORT"     # Mendekati support, potensi bounce
    else:
        return "MID_RANGE"        # Di tengah, tunggu konfirmasi


# ============================================================
# CORE: FASE 1 — SCAN RINGAN (1H ONLY)
# Tujuan: Ranking awal untuk filter, belum full MTF
# ============================================================

async def fetch_ohlcv_safe(exchange, symbol: str, timeframe: str, limit: int):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if ohlcv and len(ohlcv) >= CANDLES_REQUIRED:
            return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        logging.debug(f"fetch_ohlcv {symbol} {timeframe}: {e}")
    return None


def build_indicators(df: pd.DataFrame) -> dict:
    """Hitung semua indikator dari satu DataFrame OHLCV."""
    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']

    # Volume Flow (Normalized Delta)
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

    # Market Structure
    swings = calc_swing_levels(df)

    latest = df.iloc[-1]
    price  = float(latest['close'])

    struct_bias = calc_structure_bias(
        price,
        swings['swing_high'],
        swings['swing_low']
    )

    return {
        'close':        price,
        'power_score':  round(float(ps.iloc[-1]), 1),
        'rsi':          round(float(rsi.iloc[-1]), 1),
        'z_score':      round(float(z_score.iloc[-1]), 2),
        'ema_fast':     float(ema_f.iloc[-1]),
        'ema_slow':     float(ema_s.iloc[-1]),
        'ema_trend':    float(ema_t.iloc[-1]),
        'macd_hist':    round(float(hist.iloc[-1]), 6),
        'macd':         round(float(macd.iloc[-1]), 6),
        'macd_signal':  round(float(sig.iloc[-1]), 6),
        'atr':          round(float(atr.iloc[-1]), 6),
        'swing_high':   round(swings['swing_high'], 6) if swings['swing_high'] else None,
        'swing_low':    round(swings['swing_low'], 6) if swings['swing_low'] else None,
        'prev_sh':      round(swings['prev_swing_high'], 6) if swings['prev_swing_high'] else None,
        'prev_sl':      round(swings['prev_swing_low'], 6) if swings['prev_swing_low'] else None,
        'struct_bias':  struct_bias,
    }


async def phase1_scan(exchange, coin: dict) -> dict | None:
    """Fase 1: Hanya tarik 1H, hitung skor awal."""
    symbol = coin['symbol']
    df_1h  = await fetch_ohlcv_safe(exchange, symbol, '1h', CANDLES_REQUIRED)
    if df_1h is None:
        return None

    ind = build_indicators(df_1h)
    z   = ind['z_score']

    # Volume Z-score
    vol_avg = pd.Series([ind['z_score']]) # placeholder, real z_score sudah di ind
    whale   = "NUCLEAR" if z > 3.0 else "ACTIVE" if z > 1.2 else "QUIET"
    macd_ok = ind['macd_hist'] > 0
    macd_str = "BULL" if macd_ok else "BEAR"

    return {
        'symbol':       symbol,
        'Symbol':       symbol.split(':')[0],
        'Price':        ind['close'],
        'power_1h':     ind['power_score'],
        'RSI_1H':       ind['rsi'],
        'Whale':        whale,
        'z_score':      z,
        'MACD_1H':      macd_str,
        'ATR_1H':       ind['atr'],
        'EMA_Fast_1H':  ind['ema_fast'],
        'EMA_Slow_1H':  ind['ema_slow'],
        'Swing_High_1H': ind['swing_high'],
        'Swing_Low_1H':  ind['swing_low'],
        'Struct_1H':    ind['struct_bias'],
        # Placeholder, diisi fase 2
        'Trend_4H':     'PENDING',
        'RSI_4H':       None,
        'power_4h':     None,
        'Swing_High_4H': None,
        'Swing_Low_4H':  None,
        'Struct_4H':    'PENDING',
    }


# ============================================================
# CORE: FASE 2 — ENRICHMENT 4H (HANYA TOP KANDIDAT)
# ============================================================

async def phase2_enrich(exchange, candidate: dict) -> dict:
    """Fase 2: Tambahkan data 4H ke kandidat terpilih dari fase 1."""
    symbol = candidate['symbol']
    await asyncio.sleep(PHASE2_DELAY)  # Throttle ringan untuk hindari burst
    df_4h  = await fetch_ohlcv_safe(exchange, symbol, '4h', CANDLES_REQUIRED)

    if df_4h is None:
        # Fallback: gunakan data 1H sebagai estimasi 4H
        candidate['Trend_4H']      = 'N/A'
        candidate['RSI_4H']        = candidate['RSI_1H']
        candidate['power_4h']      = candidate['power_1h']
        candidate['Swing_High_4H'] = candidate['Swing_High_1H']
        candidate['Swing_Low_4H']  = candidate['Swing_Low_1H']
        candidate['Struct_4H']     = candidate['Struct_1H']
        return candidate

    ind4 = build_indicators(df_4h)
    c4   = ind4['close']

    # Trend filter via EMA alignment
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
# SCORING & RISK MANAGEMENT
# ============================================================

def finalize_candidate(c: dict) -> dict:
    """
    Hitung Composite Score gabungan MTF + ATR-based SL/TP
    yang diselaraskan dengan Swing Level (Price Action).
    """
    p1h = c['power_1h']
    p4h = c['power_4h'] if c['power_4h'] is not None else p1h

    # Bobot: 4H lebih dominan untuk swing
    composite = (p4h * 0.55) + (p1h * 0.45)

    # Penalti jika counter-trend 4H
    trend = c.get('Trend_4H', 'RANGING')
    ps_dir = 1 if p1h > 0 else -1
    td_dir = 1 if trend == 'UPTREND' else (-1 if trend == 'DOWNTREND' else 0)
    if td_dir != 0 and ps_dir != td_dir:
        composite -= 15

    composite = round(composite, 1)

    # Matrix label
    if composite >= 40:    matrix = "FULL BULL"
    elif composite > 10:   matrix = "BULLISH"
    elif composite <= -40: matrix = "FULL BEAR"
    elif composite < -10:  matrix = "BEARISH"
    else:                  matrix = "NEUTRAL"

    # ---- ATR-based SL/TP, diselaraskan dengan Swing Level ----
    price   = c['Price']
    atr     = c['ATR_1H']
    sh_1h   = c['Swing_High_1H']
    sl_1h   = c['Swing_Low_1H']

    # SL: gunakan yang LEBIH KONSERVATIF antara ATR-based vs Swing Level
    # LONG: SL di bawah Swing Low atau 1.5×ATR, mana yang lebih dekat (lebih protektif)
    atr_sl_long   = round(price - atr * 1.5, 6)
    struct_sl_long = round(sl_1h * 0.998, 6) if sl_1h else atr_sl_long
    sl_long_final  = max(atr_sl_long, struct_sl_long)  # Lebih tinggi = lebih ketat

    # SHORT: SL di atas Swing High atau 1.5×ATR
    atr_sl_short   = round(price + atr * 1.5, 6)
    struct_sl_short = round(sh_1h * 1.002, 6) if sh_1h else atr_sl_short
    sl_short_final  = min(atr_sl_short, struct_sl_short)  # Lebih rendah = lebih ketat

    # TP: ATR-based dengan bonus jika ada swing level berikutnya
    tp1_long  = round(price + atr * 2.0, 6)
    tp2_long  = round(price + atr * 3.5, 6)
    tp1_short = round(price - atr * 2.0, 6)
    tp2_short = round(price - atr * 3.5, 6)

    # Jika ada Prev Swing High lebih tinggi, gunakan sebagai TP2 Long
    # (target price action lebih natural)

    c.update({
        'Matrix_Sync':    matrix,
        'Composite':      composite,
        'SL_Long':        sl_long_final,
        'TP1_Long':       tp1_long,
        'TP2_Long':       tp2_long,
        'SL_Short':       sl_short_final,
        'TP1_Short':      tp1_short,
        'TP2_Short':      tp2_short,
    })
    return c


# ============================================================
# DATA PIPELINE (2 FASE)
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

        logging.info(f"Fase 1: Scanning {len(top_coins)} koin @ 1H...")

        # FASE 1 — Scan ringan paralel
        sem1 = asyncio.Semaphore(SEMAPHORE_PHASE1)
        async def safe_p1(coin):
            async with sem1:
                return await phase1_scan(exchange, coin)

        p1_results = await asyncio.gather(*[safe_p1(c) for c in top_coins])
        valid_p1   = [r for r in p1_results if r is not None]

        # Ranking awal berdasarkan Power Score 1H absolut
        top_candidates = sorted(valid_p1, key=lambda x: abs(x['power_1h']), reverse=True)[:RANKED_CANDIDATES]
        logging.info(f"Fase 1 selesai: {len(top_candidates)} kandidat terpilih. Mulai Fase 2 (4H enrichment)...")

        # FASE 2 — Enrichment 4H hanya untuk top kandidat
        sem2 = asyncio.Semaphore(SEMAPHORE_PHASE2)
        async def safe_p2(c):
            async with sem2:
                return await phase2_enrich(exchange, c)

        enriched = await asyncio.gather(*[safe_p2(c) for c in top_candidates])
        logging.info("Fase 2 selesai. Menghitung final score...")

        # Finalisasi: composite score + SL/TP
        final = [finalize_candidate(c) for c in enriched]

        # Re-rank berdasarkan Composite Score
        final_ranked = sorted(final, key=lambda x: abs(x['Composite']), reverse=True)
        return final_ranked

    finally:
        await exchange.close()


# ============================================================
# FIX #2 — PROMPT BERKUALITAS TINGGI
# Data ditulis dalam bahasa natural per-koin (bukan kode mesin),
# AI diminta 5 opsi + Red Flag yang kritis (seperti versi asli).
# ============================================================

def format_coin_for_prompt(i: int, d: dict) -> str:
    """
    Format data koin menjadi narasi ringkas yang mudah dibaca AI.
    Hindari format pipe-separated seperti kode mesin.
    """
    action = "LONG" if d['Composite'] > 0 else "SHORT"

    # Hitung % risiko untuk referensi AI
    risk_long  = abs(d['Price'] - d['SL_Long'])  / d['Price'] * 100
    risk_short = abs(d['Price'] - d['SL_Short']) / d['Price'] * 100
    rr_long    = abs(d['TP1_Long']  - d['Price']) / max(abs(d['Price'] - d['SL_Long']), 1e-9)
    rr_short   = abs(d['TP1_Short'] - d['Price']) / max(abs(d['Price'] - d['SL_Short']), 1e-9)

    return f"""
**{i}. {d['Symbol']}** (Harga: {d['Price']})
- Kecenderungan: {action} | Matrix: {d['Matrix_Sync']} | Composite Score: {d['Composite']}
- Trend 4H: {d['Trend_4H']} | RSI 1H: {d['RSI_1H']} | RSI 4H: {d.get('RSI_4H', 'N/A')}
- Struktur Harga 1H: {d['Struct_1H']} (Support: {d['Swing_Low_1H']}, Resistance: {d['Swing_High_1H']})
- Struktur Harga 4H: {d['Struct_4H']} (Support: {d['Swing_Low_4H']}, Resistance: {d['Swing_High_4H']})
- Aktivitas Whale: {d['Whale']} (Volume Z-score: {d['z_score']}) | MACD 1H: {d['MACD_1H']}
- Rencana LONG → Entry: {d['Price']} | SL: {d['SL_Long']} ({risk_long:.1f}% risiko) | TP1: {d['TP1_Long']} | TP2: {d['TP2_Long']} | R:R TP1 ≈ 1:{rr_long:.1f}
- Rencana SHORT → Entry: {d['Price']} | SL: {d['SL_Short']} ({risk_short:.1f}% risiko) | TP1: {d['TP1_Short']} | TP2: {d['TP2_Short']} | R:R TP1 ≈ 1:{rr_short:.1f}"""


async def ask_ai_agent(data_list: list) -> str:
    if not data_list:
        return "⚠️ Tidak ada data valid untuk dianalisis."

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    )

    coin_narratives = "\n".join([format_coin_for_prompt(i, d) for i, d in enumerate(data_list, 1)])

    prompt = f"""Kamu adalah AI Swing Trader profesional dengan keahlian Price Action dan analisis multi-timeframe.

Berikut adalah {len(data_list)} koin teratas hasil scan algoritma "God Mode Matrix":

{coin_narratives}

---
**TUGASMU:**

Pilih **TEPAT 5 KOIN TERBAIK** dari daftar di atas untuk setup swing trading (hold beberapa jam).

**KRITERIA SELEKSI (urut prioritas):**
1. Konfluensi Kuat: Trend 4H sejalan dengan Matrix_Sync dan Struktur Harga
2. Price Action Valid: Harga di dekat Support/Resistance nyata (Swing Level), bukan di tengah range
3. Whale + MACD sebagai konfirmasi tambahan
4. R:R minimal 1:2 di TP1

**FORMAT WAJIB untuk setiap koin:**

### [Nomor]. [NAMA KOIN] — [LONG / SHORT]
**Konfluensi:** [Sebutkan 3 faktor terkuat secara spesifik, contoh: "Trend 4H UPTREND + harga bounce dari Swing Low 1H di 0.524 + Whale NUCLEAR"]
**Entry:** [Harga] _(area/level konkret)_
**Stop Loss:** [Harga] _(sebutkan alasan: "di bawah Swing Low" atau "di atas Swing High")_
**TP1:** [Harga] | **TP2:** [Harga]
**Risk/Reward:** [X:Y]
**🚩 Red Flag / Invalidasi:** [Kondisi spesifik yang membatalkan sinyal ini — WAJIB diisi]

---
Tulis langsung 5 pilihan tanpa pembukaan atau penutupan. Prioritas kejelasan dan akurasi.
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

    return "⚠️ Gemini API gagal setelah 3x percobaan. Coba lagi nanti."


# ============================================================
# TELEGRAM SENDER
# ============================================================

async def send_to_telegram(text: str, header: str = ""):
    """Kirim pesan ke Telegram dengan auto-split jika > 4096 karakter."""
    url       = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    full_text = header + text

    # Telegram batas 4096 karakter per pesan
    chunks = [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]

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
                await asyncio.sleep(1)  # Jeda antar chunk


# ============================================================
# MAIN
# ============================================================

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]):
        logging.error("❌ API Keys belum lengkap! Pastikan env vars sudah diset.")
        return

    timestamp = datetime.now().strftime("%d %b %Y, %H:%M WIB")
    header = (
        f"🎯 *GOD MODE v3 — TOP 5 SWING SETUPS*\n"
        f"🕐 {timestamp} | MTF: 4H + 1H | Price Action + Indikator\n"
        f"{'─' * 34}\n\n"
    )

    logging.info("🚀 God Mode v3 dimulai...")

    try:
        data = await get_high_precision_data()

        if not data:
            logging.error("Tidak ada data valid.")
            await send_to_telegram("⚠️ Scan gagal: tidak ada data valid. Cek koneksi / API bursa.")
            return

        logging.info(f"✅ {len(data)} koin final. Mengirim ke Gemini...")
        analysis = await ask_ai_agent(data)
        await send_to_telegram(analysis, header=header)
        logging.info("✅ Laporan dikirim ke Telegram.")

    except Exception as e:
        logging.exception(f"❌ Fatal error: {e}")
        try:
            await send_to_telegram(f"❌ *SYSTEM ERROR*\n`{str(e)[:300]}`")
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
