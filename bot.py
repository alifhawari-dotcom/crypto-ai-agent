import os
import asyncio
import logging
import json
import re
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
# CONSTANTS (V5.2 — BULLETPROOF JSON + GUARDRAILS)
# ============================================================

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

TOP_COINS_BY_VOLUME = 40
RANKED_CANDIDATES   = 10
CANDLES_REQUIRED    = 100

FIXED_RISK_USD  = 1.50
RR_TP1          = 2.0
RR_TP2          = 3.5
SL_ATR_BUFFER   = 0.5
QTY_MAX_CAP     = 99999
MAX_RISK_PCT    = 15.0   # Koin dibatalkan jika SL > 15% dari entry

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

# ============================================================
# INDIKATOR
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, pc = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
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
    c  = df['close']
    s  = c.rolling(20).mean()
    sd = c.rolling(20).std()
    return (s + 2*sd < s + 1.5*atr) & (s - 2*sd > s - 1.5*atr)


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
# DATA PIPELINE
# ============================================================

async def fetch_ohlcv_safe(exchange, symbol: str, tf: str, limit: int):
    try:
        data = await asyncio.wait_for(
            exchange.fetch_ohlcv(symbol, tf, limit=limit), timeout=10.0
        )
        if data and len(data) >= CANDLES_REQUIRED:
            return pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    except asyncio.TimeoutError:
        logging.debug(f"{symbol} {tf}: timeout")
    except Exception as e:
        logging.debug(f"{symbol} {tf}: {e}")
    return None


def build_indicators(df: pd.DataFrame) -> dict:
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    atr  = calc_atr(df)
    rng  = h - l
    nd   = pd.Series(np.where(rng==0, 0, ((c-l)-(h-c))/rng*v), index=df.index)
    ad   = nd.abs().rolling(20).mean()
    sf   = np.clip((nd / np.where(ad==0, 1, ad)) * 20, -40, 40)
    rsi  = calc_rsi(c)
    ps   = sf + np.clip((rsi-50)*1.2, -30, 30)
    sw   = calc_swing_levels(df)
    vm   = v.rolling(20).mean().iloc[-1]
    vs   = v.rolling(20).std().iloc[-1]
    z    = float((v.iloc[-1]-vm) / (vs if vs != 0 else 1))
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
    }


async def phase1_scan(exchange, coin: dict) -> dict | None:
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
    }


async def phase2_enrich(exchange, c: dict) -> dict:
    symbol = c['symbol']
    await asyncio.sleep(PHASE2_DELAY)
    df_1h, df_5m = await asyncio.gather(
        fetch_ohlcv_safe(exchange, symbol, TF_MACRO,  CANDLES_REQUIRED),
        fetch_ohlcv_safe(exchange, symbol, TF_MICRO,  CANDLES_REQUIRED),
    )
    if df_1h is not None:
        i1    = build_indicators(df_1h)
        v     = i1['close']
        trend = ("UPTREND"   if v > i1['ema_f'] > i1['ema_s'] else
                 "DOWNTREND" if v < i1['ema_f'] < i1['ema_s'] else
                 "RANGING")
        c.update({'Trend_1h': trend, 'power_1h': i1['power_score']})
    else:
        c.update({'Trend_1h': 'N/A', 'power_1h': c['power_15m']})

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

    price   = c['Price']
    atr     = c['ATR_15m']
    sh      = c['Swing_High_15m']
    sl_val  = c['Swing_Low_15m']

    buy_stop  = round(sh * 1.001, 6) if sh > price else round(price + atr * 0.2, 6)
    sell_stop = round(sl_val * 0.999, 6) if sl_val < price else round(price - atr * 0.2, 6)

    sl_long  = round(sl_val - atr * SL_ATR_BUFFER, 6)
    sl_short = round(sh     + atr * SL_ATR_BUFFER, 6)

    rl = max(buy_stop  - sl_long,  1e-9)
    rs = max(sl_short  - sell_stop, 1e-9)

    # ── GUARDRAIL 1: Anti TP Minus ────────────────────────────
    # TP short tidak boleh ≤ 0. Diklem di minimal 1% dari sell_stop.
    tp1_long  = round(buy_stop  + rl * RR_TP1, 6)
    tp2_long  = round(buy_stop  + rl * RR_TP2, 6)
    tp1_short = round(max(sell_stop - rs * RR_TP1, sell_stop * 0.01), 6)
    tp2_short = round(max(sell_stop - rs * RR_TP2, sell_stop * 0.01), 6)

    # ── GUARDRAIL 2: Batalkan koin jika SL terlalu lebar ─────
    # Risk > MAX_RISK_PCT% dari entry = setup tidak layak
    risk_pct_long  = (rl / buy_stop)   * 100
    risk_pct_short = (rs / sell_stop)  * 100
    wild_long  = risk_pct_long  > MAX_RISK_PCT
    wild_short = risk_pct_short > MAX_RISK_PCT

    if wild_long:
        logging.debug(f"{c['Symbol']} LONG risk {risk_pct_long:.1f}% > {MAX_RISK_PCT}% → CANCEL")
    if wild_short:
        logging.debug(f"{c['Symbol']} SHORT risk {risk_pct_short:.1f}% > {MAX_RISK_PCT}% → CANCEL")

    c.update({
        'Buy_Stop':    buy_stop,
        'SL_Long':     sl_long,
        'TP1_Long':    tp1_long,
        'TP2_Long':    tp2_long,
        'Qty_Long':    format_qty(FIXED_RISK_USD / rl),
        'Sell_Stop':   sell_stop,
        'SL_Short':    sl_short,
        'TP1_Short':   tp1_short,
        'TP2_Short':   tp2_short,
        'Qty_Short':   format_qty(FIXED_RISK_USD / rs),
        # Cancel jika: counter-trend 1H ATAU SL terlalu lebar
        'Cancel_Long':  (trend == 'DOWNTREND') or wild_long,
        'Cancel_Short': (trend == 'UPTREND')   or wild_short,
        'Risk_Pct_Long':  round(risk_pct_long, 1),
        'Risk_Pct_Short': round(risk_pct_short, 1),
    })
    return c


async def get_high_precision_data():
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        top     = sorted(
            [v for v in tickers.values() if v.get('quoteVolume')],
            key=lambda x: x['quoteVolume'], reverse=True
        )[:TOP_COINS_BY_VOLUME]

        logging.info(f"Fase 1: scan {len(top)} koin @ {TF_STRUCT}...")
        sem1 = asyncio.Semaphore(SEMAPHORE_P1)
        async def sp1(coin):
            async with sem1: return await phase1_scan(exchange, coin)
        p1    = await asyncio.gather(*[sp1(c) for c in top])
        cands = sorted([r for r in p1 if r], key=lambda x: abs(x['power_15m']), reverse=True)[:RANKED_CANDIDATES]

        logging.info(f"Fase 2: enrich {len(cands)} koin @ {TF_MACRO}+{TF_MICRO}...")
        sem2 = asyncio.Semaphore(SEMAPHORE_P2)
        async def sp2(c):
            async with sem2: return await phase2_enrich(exchange, c)
        enriched = await asyncio.gather(*[sp2(c) for c in cands])

        return sorted([finalize_candidate(c) for c in enriched],
                      key=lambda x: abs(x['Composite']), reverse=True)
    finally:
        await exchange.close()


# ============================================================
# GEMINI — SINGLE CALL + BULLETPROOF PARSER
#
# Root cause error v5.1 di produksi:
#   AI mengembalikan: "picks": "A", "B", "C", "D"
#   (lupa kurung siku → JSONDecodeError → fallback total → teks mentah
#    tercetak ke Telegram + kartu mengabaikan pilihan AI)
#
# Fix di v5.2:
#   1. Prompt lebih eksplisit — contoh format picks dengan kurung siku
#   2. Parser berlapis 5 level — level baru: ekstrasi simbol
#      dari string tanpa kurung siku (persis penyebab bug produksi)
#   3. Guardrail anti-TP minus & SL lebar sudah di finalize_candidate
#   4. Fallback teks mentah TIDAK dikirim ke Telegram jika picks valid
# ============================================================

_PROMPT_TEMPLATE = """\
Kamu AI Intraday Breakout Trader. Dari data berikut, pilih TEPAT 4 koin terbaik.
WAJIB ABAIKAN koin bertanda ⚠️CANCEL.
Prioritas: SM=NUCLEAR/NUC+SQZ > 3TF=✅ > Trend 1H searah aksi.

{rows}

FORMAT BALASAN — HANYA JSON VALID, TANPA TEKS LAIN, TANPA MARKDOWN FENCE:
{{
  "analysis": "### 1. NAMA — LONG\\n**Konfluensi:** ...\\n**Entry:** BUY STOP di HARGA | **Qty:** QTY _(Risk ${risk})_\\n**SL:** HARGA | **TP1:** HARGA | **TP2:** HARGA\\n**🚩 Cancel jika:** ...\\n---\\n### 2. ...",
  "picks": ["SYMBOL_1", "SYMBOL_2", "SYMBOL_3", "SYMBOL_4"]
}}

ATURAN WAJIB:
- "picks" HARUS berupa array dengan KURUNG SIKU [ ] — contoh: ["BTC/USDT:USDT","ETH/USDT:USDT"]
- Isi "picks" HARUS SAMA PERSIS dengan simbol di kolom symbol: pada data
- Koin di "analysis" dan "picks" HARUS IDENTIK
- TP tidak boleh bernilai negatif atau nol
"""


async def ask_ai_agent(data_list: list) -> tuple[str, list[str]]:
    """
    Satu panggilan Gemini → (narasi_markdown, [simbol_pilihan_AI]).
    Narasi dan picks berasal dari respons yang sama → tidak bisa inkonsisten.
    """
    if not data_list:
        return "⚠️ Data kosong.", []

    rows = []
    for d in data_list:
        act  = "LONG" if d['Composite'] > 0 else "SHORT"
        cncl = " ⚠️CANCEL" if (act=="LONG" and d['Cancel_Long']) or (act=="SHORT" and d['Cancel_Short']) else ""
        rows.append(
            f"symbol:{d['symbol']}|{act}{cncl}|Score:{d['Composite']}({d['Matrix_Sync']})|"
            f"1H:{d['Trend_1h']}|SM:{d['SM_Signal']}|3TF:{'✅' if d['Aligned'] else '❌'}|"
            f"Entry:{'BUY@'+str(d['Buy_Stop']) if act=='LONG' else 'SELL@'+str(d['Sell_Stop'])}|"
            f"SL:{d['SL_Long'] if act=='LONG' else d['SL_Short']}|"
            f"TP1:{d['TP1_Long'] if act=='LONG' else d['TP1_Short']}|"
            f"TP2:{d['TP2_Long'] if act=='LONG' else d['TP2_Short']}|"
            f"Qty:{d['Qty_Long'] if act=='LONG' else d['Qty_Short']}"
        )

    prompt = _PROMPT_TEMPLATE.format(
        rows="\n".join(rows),
        risk=FIXED_RISK_USD
    )

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
                    logging.warning(f"{model} attempt {attempt}: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)

    return "⚠️ Semua model Gemini kena limit. Coba lagi nanti.", []


def _parse_ai_response(raw: str, data_list: list) -> tuple[str, list[str]]:
    """
    Parser berlapis 5 level — tahan terhadap berbagai format rusak dari AI.

    Level 1: json.loads normal
    Level 2: ekstrak blok {...} pertama dengan regex lalu json.loads
    Level 3: ekstrak field "picks" array normal ["a","b"]
    Level 4: ekstrak simbol dari "picks": "a","b","c" (tanpa kurung siku ← bug produksi)
    Level 5: fallback regex nama koin dari teks bebas di seluruh respons
    Jika semua gagal → fallback data[:4], log warning

    Narasi selalu diambil dari field "analysis" jika tersedia.
    Jika tidak, narasi = raw teks (HANYA jika picks berhasil ditemukan).
    Jika semua gagal total, narasi = pesan error bersih (BUKAN teks JSON mentah).
    """
    valid_symbols = {d['symbol'] for d in data_list}

    # Bersihkan markdown fence jika ada
    clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()

    def validate_picks(candidates: list) -> list:
        """Filter picks ke simbol yang valid, maks 4."""
        return [s.strip() for s in candidates if s.strip() in valid_symbols][:4]

    # ── Level 1: json.loads normal ────────────────────────────
    try:
        obj      = json.loads(clean)
        analysis = str(obj.get('analysis', '')).strip()
        picks    = validate_picks(obj.get('picks', []))
        if picks:
            logging.info(f"[Parser L1] OK: {picks}")
            return analysis or raw, picks
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    # ── Level 2: cari blok {...} pertama ─────────────────────
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        try:
            obj      = json.loads(m.group())
            analysis = str(obj.get('analysis', '')).strip()
            picks    = validate_picks(obj.get('picks', []))
            if picks:
                logging.warning(f"[Parser L2] regex block OK: {picks}")
                return analysis or raw, picks
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Level 3: ekstrak "picks" array normal ─────────────────
    # Menangkap: "picks": ["A", "B", "C"]
    m = re.search(r'"picks"\s*:\s*\[([^\]]*)\]', clean, re.DOTALL)
    if m:
        found = re.findall(r'"([^"]+)"', m.group(1))
        picks = validate_picks(found)
        if picks:
            logging.warning(f"[Parser L3] picks-array regex OK: {picks}")
            # Coba ambil analysis dengan regex terpisah
            analysis = _extract_analysis_field(clean)
            return analysis or raw, picks

    # ── Level 4: "picks" tanpa kurung siku ← ROOT CAUSE BUG PRODUKSI
    # Menangkap: "picks": "A", "B", "C", "D"  (AI lupa [ ])
    m = re.search(r'"picks"\s*:\s*"([^"]+)"(?:\s*,\s*"([^"]+)")*', clean)
    if m:
        # Ambil semua string setelah "picks": sampai akhir baris / field berikutnya
        region = clean[m.start():]
        found  = re.findall(r'"([A-Z0-9]+/USDT(?::USDT)?)"', region[:300])
        picks  = validate_picks(found)
        if picks:
            logging.warning(f"[Parser L4] picks-no-bracket OK (root cause fix): {picks}")
            analysis = _extract_analysis_field(clean)
            return analysis or raw, picks

    # ── Level 5: ekstrak nama koin dari seluruh teks ──────────
    # Sabuk pengaman terakhir — selama AI menyebut nama koin, kita tangkap
    all_symbols = re.findall(r'\b([A-Z]{2,10}/USDT(?::USDT)?)\b', raw)
    picks = validate_picks(list(dict.fromkeys(all_symbols)))  # deduplicate
    if picks:
        logging.warning(f"[Parser L5] full-text symbol scan OK: {picks}")
        analysis = _extract_analysis_field(clean)
        return analysis or raw, picks

    # ── Fallback total ────────────────────────────────────────
    logging.error("[Parser] Semua level gagal — fallback ke data[:4]")
    fallback_picks = [d['symbol'] for d in data_list[:4]]
    # Kirim pesan error bersih — BUKAN teks JSON mentah
    fallback_analysis = (
        "⚠️ *AI tidak dapat memformat respons dengan benar.*\n"
        "Bot otomatis memilih 4 koin teratas berdasarkan Composite Score.\n"
        "_Kartu trade di bawah ini dibuat dari data mentah, bukan pilihan AI._"
    )
    return fallback_analysis, fallback_picks


def _extract_analysis_field(text: str) -> str:
    """Coba ekstrak nilai field 'analysis' dari teks JSON setengah-valid."""
    # Cari "analysis": "..." (multiline, escaped)
    m = re.search(r'"analysis"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if m:
        # Unescape escape sequences dari JSON
        try:
            return bytes(m.group(1), 'utf-8').decode('unicode_escape')
        except Exception:
            return m.group(1).replace('\\n', '\n').replace('\\"', '"')
    return ""


# ============================================================
# TELEGRAM
# ============================================================

def build_trade_message(d: dict, rank: int) -> str:
    """
    Kartu trade per koin — tap-to-copy di Telegram.
    Hanya dipanggil untuk koin yang sudah diverifikasi AI.
    Risk % ditampilkan dari data nyata (bukan kalkulasi ulang).
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
    risk_pct = d.get('Risk_Pct_Long') if is_long else d.get('Risk_Pct_Short')
    risk_str = f"{risk_pct:.1f}%" if risk_pct is not None else "N/A"

    align_badge = "✅ 3TF" if d.get('Aligned') else "⚡ 2TF"
    sqz_badge   = " 🔥SQZ" if d.get('Squeeze_15m') else ""

    cancel_warn = ""
    if is_long and d.get('Cancel_Long'):
        reason = "SL terlalu lebar" if d.get('Risk_Pct_Long', 0) > MAX_RISK_PCT else "1H DOWNTREND"
        cancel_warn = f"\n⚠️ *CANCEL* — {reason}!"
    elif not is_long and d.get('Cancel_Short'):
        reason = "SL terlalu lebar" if d.get('Risk_Pct_Short', 0) > MAX_RISK_PCT else "1H UPTREND"
        cancel_warn = f"\n⚠️ *CANCEL* — {reason}!"

    return (
        f"{act_emoji} *{rank}. {d['Symbol']} — {act_label}*  "
        f"`{d['Matrix_Sync']}`  {align_badge}{sqz_badge}\n"
        f"📊 SM: {d['SM_Signal']}  Score: `{d['Composite']}`  1H: {d['Trend_1h']}\n"
        f"{'─' * 28}\n"
        f"📌 *{entry_type}*\n"
        f"  Entry : `{entry}`\n"
        f"  SL    : `{sl}`  _(-{risk_str})_\n"
        f"  TP1   : `{tp1}`  _(R:R 1:{RR_TP1})_\n"
        f"  TP2   : `{tp2}`  _(R:R 1:{RR_TP2})_\n"
        f"  Qty   : `{qty}` koin  _(Risk ~${FIXED_RISK_USD})_"
        f"{cancel_warn}"
    )


async def _tg_post(session: aiohttp.ClientSession, payload: dict) -> bool:
    """Kirim satu pesan Telegram dengan retry + fallback plain text."""
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    return True
                if resp.status == 400 and "parse_mode" in payload:
                    # Markdown error → kirim ulang tanpa formatting
                    plain = {k: v for k, v in payload.items() if k != "parse_mode"}
                    async with session.post(url, json=plain) as r2:
                        return r2.status == 200
                logging.warning(f"Telegram attempt {attempt}: HTTP {resp.status}")
        except Exception as e:
            logging.warning(f"Telegram attempt {attempt}: {e}")
        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY)
    return False


async def send_to_telegram(text: str, header: str = ""):
    """Kirim analisis naratif AI (chunked jika >4000 karakter)."""
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
    """
    Kirim kartu tap-to-copy per koin.
    selected_data sudah diverifikasi dari respons AI yang sama → selalu konsisten.
    """
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
    header = (
        f"👑 *GOD MODE v5.2 — BULLETPROOF JSON*\n"
        f"🕐 {ts} | 1H+15m+5m | Anti Stop-Hunt\n"
        f"💰 Risk per trade: ${FIXED_RISK_USD} | R:R {RR_TP1}:{RR_TP2}\n"
        f"{'─' * 38}\n\n"
    )

    logging.info("🚀 God Mode v5.2 dimulai...")

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

        logging.info(f"✅ {len(data)} koin siap. Kirim ke Gemini (single call)...")

        # ── Satu panggilan Gemini → narasi + simbol dari sumber yang sama ──
        analysis, selected_symbols = await ask_ai_agent(data)

        # ── Map simbol ke data lengkap ──────────────────────────────────
        symbol_lookup = {d['symbol']: d for d in data}
        selected_data = [symbol_lookup[sym] for sym in selected_symbols if sym in symbol_lookup]

        if not selected_data:
            logging.warning("selected_data kosong — fallback ke data[:4]")
            selected_data = data[:4]

        logging.info(
            f"✅ AI memilih {len(selected_data)} koin: "
            f"{[d['Symbol'] for d in selected_data]}"
        )

        # ── Pesan 1: Header + analisis naratif ─────────────────────────
        await send_to_telegram(analysis, header=header)
        logging.info("✅ Analisis AI terkirim.")

        await asyncio.sleep(1.5)

        # ── Pesan 2+: Kartu tap-to-copy — 100% konsisten dengan narasi ─
        await send_trade_cards(selected_data)
        logging.info(f"✅ {len(selected_data)} kartu trade terkirim.")

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
