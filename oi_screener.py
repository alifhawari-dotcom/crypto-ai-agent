"""
oi_screener.py — Screener OI × Harga (4-Quadrant) + Funding + Long/Short Ratio
Sumber data: Gate.io v4 (tidak kena geo-block GitHub Actions)
Verifikasi silang: Bybit instruments-info (hanya tampilkan yang tradable di Bybit)

Modul TERPISAH dari screener.py (Apex Quant v7.1). Additive-only.

=============================================================================
KENAPA ARSITEKTURNYA BEGINI
=============================================================================
- Bybit /v5/market/tickers TERBUKTI 403 dari runner GitHub Actions (tes 14 Sep
  2026). Jadi Bybit tidak bisa dipakai sebagai sumber universe/OI/funding.
- Gate.io TERBUKTI jalan dari runner yang sama (screener.py sudah 5 bulan).
- TAPI Gate.io punya 1.700+ perpetual vs Bybit yang jauh lebih sedikit →
  mayoritas coin Gate.io TIDAK ADA di Bybit. Karena eksekusi trading kamu di
  Bybit, kandidat yang tidak tradable di sana = sampah.
  → Solusi: ambil data dari Gate.io, lalu SARING pakai daftar simbol Bybit.

=============================================================================
DASAR KONSEPTUAL
=============================================================================
Framework 4-kuadran OI × Harga:
    Harga ↑ + OI ↑ = LONG BUILDUP    → uang baru masuk long
    Harga ↓ + OI ↑ = SHORT BUILDUP   → uang baru masuk short
    Harga ↑ + OI ↓ = SHORT COVERING  → short keluar (bullish, kurang tahan lama)
    Harga ↓ + OI ↓ = LONG UNWINDING  → long keluar (lemah)

OI TANPA konteks harga tidak punya arah. Funding rate = sinyal CROWDING,
bukan sinyal ARAH: ekstrem di satu sisi = sisi itu rapuh terhadap cascade.

⚠️ TIDAK ADA KLAIM EDGE/PROFIT. Semua threshold BELUM dikalibrasi backtest.
   Framework ini DESKRIPTIF (di mana uang terposisi sekarang), bukan
   PREDIKTIF (ke mana harga akan bergerak).

⚠️ CATATAN VERIFIKASI: nama field Gate.io contract_stats (open_interest,
   lsr_account, dll) disusun dari dokumentasi publik tapi BELUM diverifikasi
   langsung terhadap respons API riil. Script ini punya mode diagnostik
   (DIAGNOSTIC_MODE=True) yang mencetak struktur respons mentah — jalankan
   sekali dengan mode itu kalau ada field yang ternyata beda nama.
=============================================================================
"""

import os
import asyncio
import logging
import json
import aiohttp
import numpy as np
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TG_TOKEN   = (os.getenv('OI_TELEGRAM_TOKEN')   or '').strip()
TG_CHAT_ID = (os.getenv('OI_TELEGRAM_CHAT_ID') or '').strip()

# Set True untuk mencetak struktur respons API mentah (cek nama field).
# Jalankan sekali dengan ini True kalau ada data yang kosong terus.
DIAGNOSTIC_MODE = os.getenv('OI_DIAGNOSTIC', '').strip().lower() == 'true'

# ── THRESHOLD (BELUM DIKALIBRASI — ubah setelah lihat hasil nyata) ──────
# DIKALIBRASI 14 Sep 2026 dari data riil: perubahan OI per-candle untuk coin
# likuid umumnya 0.5-2%, bukan >2%. Threshold lama (2.0) membuang hampir semua
# kandidat. Angka di bawah masih PERKIRAAN — longgarkan/ketatkan setelah
# melihat berapa kandidat yang lolos per siklus.
OI_CHANGE_MIN_PCT    = 0.8     # % perubahan OI minimum agar dianggap bergerak
PRICE_CHANGE_MIN_PCT = 0.3     # % perubahan harga minimum agar dianggap bergerak
RVOL_MIN             = 1.1     # volume relatif minimum (validasi sinyal)
TOP_LSR_EXTREME_HI   = 1.5     # rasio akun trader TOP: > ini = top trader long
TOP_LSR_EXTREME_LO   = 0.67    # < ini = top trader short
FUNDING_EXTREME      = 0.0005  # 0.05%/8h — batas "crowded"
FUNDING_VERY_EXTREME = 0.0010  # 0.10%/8h — batas "sangat crowded"
LSR_EXTREME_HI       = 2.0     # long/short account ratio > ini = long ramai
LSR_EXTREME_LO       = 0.5     # < ini = short ramai
MIN_TURNOVER_24H     = 3_000_000   # USD — buang pair terlalu tipis
MIN_TF_CONSISTENCY   = 2       # kuadran sama harus muncul di >= N timeframe
MAX_SYMBOLS_SCANNED  = 300     # batas atas agar runtime & rate limit terkendali

# Timeframe: (label, interval_gate_kline, detik_per_candle)
TIMEFRAMES = [
    ('15m', '15m', 900),
    ('1h',  '1h',  3600),
    ('4h',  '4h',  14400),
]

SEMAPHORE_LIMIT = 6
MAX_RETRIES     = 3
RETRY_DELAY     = 3
TIMEOUT_SEC     = 15
TOP_N_REPORT    = 8

GATE_BASE  = "https://api.gateio.ws/api/v4/futures/usdt"
BYBIT_BASE = "https://api.bybit.com/v5/market"


# ── HTTP ────────────────────────────────────────────────────────────────
async def _get(session, url, params=None, label=""):
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC)) as r:
                if r.status == 200:
                    return await r.json()
                if r.status in (403, 451):
                    logging.warning(f"HTTP {r.status} dari {label or url} — geo-block.")
                    return None
                if r.status == 429:
                    await asyncio.sleep(RETRY_DELAY * 2)
                    continue
        except Exception as e:
            logging.debug(f"fetch error {label or url}: {e}")
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY)
    return None


# ── LAPIS 1: daftar simbol yang TRADABLE DI BYBIT (verifikasi silang) ───
async def fetch_bybit_symbols(session):
    """Ambil daftar simbol USDT perp Bybit. Kalau endpoint ini juga kena 403,
    script tetap jalan tapi TANPA filter Bybit — dan itu dilaporkan jelas
    di pesan Telegram, bukan disembunyikan."""
    data = await _get(session, f"{BYBIT_BASE}/instruments-info",
                      {"category": "linear", "limit": 1000},
                      label="Bybit instruments-info")
    if not data or data.get('retCode') != 0:
        logging.warning("Daftar simbol Bybit TIDAK bisa diambil — "
                        "filter ketersediaan Bybit DINONAKTIFKAN untuk run ini.")
        return None
    syms = {
        it['symbol'] for it in data.get('result', {}).get('list', [])
        if it.get('symbol', '').endswith('USDT') and it.get('status') == 'Trading'
    }
    logging.info(f"Bybit: {len(syms)} simbol USDT perp tradable")
    return syms


def gate_to_bybit_symbol(contract):
    """BTC_USDT (Gate) → BTCUSDT (Bybit)"""
    return contract.replace('_', '')


# ── LAPIS 2: universe dari Gate.io ──────────────────────────────────────
async def fetch_gate_universe(session, bybit_symbols):
    data = await _get(session, f"{GATE_BASE}/tickers", label="Gate tickers")
    if not data:
        return []

    if DIAGNOSTIC_MODE and data:
        logging.info(f"[DIAG] Struktur tickers Gate: {json.dumps(data[0], indent=2)[:600]}")

    out, skipped_not_on_bybit = [], 0
    for it in data:
        contract = it.get('contract', '')
        if not contract.endswith('_USDT'):
            continue
        bybit_sym = gate_to_bybit_symbol(contract)

        # Filter ketersediaan Bybit — inti dari verifikasi silang
        if bybit_symbols is not None and bybit_sym not in bybit_symbols:
            skipped_not_on_bybit += 1
            continue

        try:
            turnover = float(it.get('volume_24h_quote') or 0)
            last     = float(it.get('last') or 0)
            funding  = float(it.get('funding_rate') or 0)
        except (TypeError, ValueError):
            continue
        if turnover < MIN_TURNOVER_24H or last <= 0:
            continue

        out.append({
            'contract':    contract,
            'bybit_symbol': bybit_sym,
            'turnover24h': turnover,
            'funding':     funding,   # Gate tickers sudah bawa funding_rate
        })

    out.sort(key=lambda x: x['turnover24h'], reverse=True)
    out = out[:MAX_SYMBOLS_SCANNED]
    logging.info(f"Universe: {len(out)} pair (dibuang karena tidak ada di Bybit: "
                 f"{skipped_not_on_bybit})")
    return out


# ── LAPIS 3: OI + long/short ratio dari Gate contract_stats ─────────────
async def fetch_contract_stats(session, contract, interval, limit=3):
    """Gate contract_stats: OI, long/short ratio, dll per interval."""
    data = await _get(session, f"{GATE_BASE}/contract_stats",
                      {"contract": contract, "interval": interval, "limit": limit},
                      label="Gate contract_stats")
    if not data or not isinstance(data, list) or len(data) < 2:
        return None, None, None

    if DIAGNOSTIC_MODE:
        logging.info(f"[DIAG] contract_stats {contract}: "
                     f"{json.dumps(data[0], indent=2)[:600]}")

    # JANGAN asumsikan urutan — sort eksplisit berdasarkan field 'time'.
    # (Bug versi sebelumnya: mengasumsikan lama→baru, padahal Gate bisa
    #  mengembalikan baru→lama, yang membalik TANDA perubahan OI.)
    try:
        data = sorted(data, key=lambda d: d.get('time', 0))
    except Exception:
        pass
    latest, prev = data[-1], data[-2]

    def _f(d, *keys):
        """Ambil nilai dari key pertama yang ada — defensif terhadap
        perbedaan nama field antar versi API."""
        for k in keys:
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    oi_now  = _f(latest, 'open_interest_usd', 'open_interest')
    oi_prev = _f(prev,   'open_interest_usd', 'open_interest')
    oi_chg  = None
    if oi_now is not None and oi_prev and oi_prev > 0:
        oi_chg = ((oi_now - oi_prev) / oi_prev) * 100.0

    lsr     = _f(latest, 'lsr_account', 'long_short_ratio')
    top_lsr = _f(latest, 'top_lsr_account')
    return oi_chg, lsr, top_lsr


# ── LAPIS 4: kline dari Gate (harga + volume) ──────────────────────────
async def fetch_gate_kline(session, contract, interval, limit=25):
    data = await _get(session, f"{GATE_BASE}/candlesticks",
                      {"contract": contract, "interval": interval, "limit": limit},
                      label="Gate candlesticks")
    if not data or not isinstance(data, list) or len(data) < 21:
        return None, None

    if DIAGNOSTIC_MODE:
        logging.info(f"[DIAG] candlestick {contract}: "
                     f"{json.dumps(data[0], indent=2)[:400]}")

    try:
        closes = np.array([float(r['c']) for r in data])
        vols   = np.array([float(r['v']) for r in data])
    except (KeyError, TypeError, ValueError):
        return None, None

    if closes[-2] <= 0:
        return None, None
    price_chg = ((closes[-1] - closes[-2]) / closes[-2]) * 100.0
    vol_avg   = vols[-21:-1].mean()
    rvol      = float(vols[-1] / vol_avg) if vol_avg > 0 else 0.0
    return price_chg, rvol


# ── KLASIFIKASI ─────────────────────────────────────────────────────────
def classify_quadrant(price_chg, oi_chg):
    if price_chg is None or oi_chg is None:
        return None
    if abs(price_chg) < PRICE_CHANGE_MIN_PCT or abs(oi_chg) < OI_CHANGE_MIN_PCT:
        return 'NEUTRAL'
    if price_chg > 0 and oi_chg > 0:  return 'LONG_BUILDUP'
    if price_chg < 0 and oi_chg > 0:  return 'SHORT_BUILDUP'
    if price_chg > 0 and oi_chg < 0:  return 'SHORT_COVERING'
    return 'LONG_UNWINDING'


def funding_state(fr):
    if fr is None:
        return 'UNKNOWN', '⚪'
    if fr >= FUNDING_VERY_EXTREME:   return 'LONG_VERY_CROWDED',  '🔴'
    if fr >= FUNDING_EXTREME:        return 'LONG_CROWDED',       '🟠'
    if fr <= -FUNDING_VERY_EXTREME:  return 'SHORT_VERY_CROWDED', '🔴'
    if fr <= -FUNDING_EXTREME:       return 'SHORT_CROWDED',      '🟠'
    return 'BALANCED', '🟢'


def lsr_state(lsr):
    if lsr is None:               return 'UNKNOWN'
    if lsr >= LSR_EXTREME_HI:     return 'LONG_HEAVY'
    if lsr <= LSR_EXTREME_LO:     return 'SHORT_HEAVY'
    return 'BALANCED'


# ── SCREENING PER SIMBOL ────────────────────────────────────────────────
async def screen_symbol(session, item, sem, stats):
    """stats = dict penghitung alasan gugur, supaya kita tahu PERSIS di tahap
    mana kandidat tersaring — bukan menebak kenapa hasilnya 0."""
    contract = item['contract']
    async with sem:
        tf_data = {}
        for label, interval, _ in TIMEFRAMES:
            oi_chg, lsr, top_lsr = await fetch_contract_stats(session, contract, interval)
            price_chg, rv        = await fetch_gate_kline(session, contract, interval)
            tf_data[label] = {
                'oi_chg':    oi_chg,
                'price_chg': price_chg,
                'rvol':      rv,
                'lsr':       lsr,
                'top_lsr':   top_lsr,
                'quadrant':  classify_quadrant(price_chg, oi_chg),
            }

    if all(v['oi_chg'] is None for v in tf_data.values()):
        stats['no_oi_data'] += 1
        return None

    quads = [v['quadrant'] for v in tf_data.values()
             if v['quadrant'] and v['quadrant'] != 'NEUTRAL']
    if not quads:
        stats['all_neutral'] += 1
        return None
    dominant    = max(set(quads), key=quads.count)
    consistency = quads.count(dominant)

    if consistency < MIN_TF_CONSISTENCY:
        stats['low_consistency'] += 1
        return None
    if dominant not in ('LONG_BUILDUP', 'SHORT_BUILDUP'):
        stats['not_buildup'] += 1
        return None

    rvol_1h = tf_data.get('1h', {}).get('rvol') or 0.0
    if rvol_1h < RVOL_MIN:
        stats['low_rvol'] += 1
        return None

    stats['passed'] += 1
    funding = item.get('funding')
    fstate, femoji = funding_state(funding)
    lsr     = tf_data.get('1h', {}).get('lsr')
    top_lsr = tf_data.get('1h', {}).get('top_lsr')
    lstate  = lsr_state(lsr)

    # SQUEEZE WATCH: buildup yang sisi ramainya SEARAH posisi itu sendiri.
    # Long buildup + long crowded  → rally bergantung leverage, rawan cascade.
    # Short buildup + short crowded → short ramai, rawan short squeeze.
    squeeze = (
        (dominant == 'LONG_BUILDUP'  and (fstate in ('LONG_CROWDED', 'LONG_VERY_CROWDED')
                                          or lstate == 'LONG_HEAVY')) or
        (dominant == 'SHORT_BUILDUP' and (fstate in ('SHORT_CROWDED', 'SHORT_VERY_CROWDED')
                                          or lstate == 'SHORT_HEAVY'))
    )

    # Skor KUALITAS PEMBACAAN (bukan skor profit)
    score = consistency * 15
    score += min(int(rvol_1h * 10), 25)
    ois = [abs(v['oi_chg']) for v in tf_data.values() if v['oi_chg'] is not None]
    if ois:
        score += min(int(np.mean(ois) * 2), 20)
    if fstate == 'BALANCED':
        score += 10
    score = max(0, min(100, score))

    return {
        'contract':     contract,
        'symbol':       item['bybit_symbol'],
        'quadrant':     dominant,
        'consistency':  consistency,
        'tf_total':     len(TIMEFRAMES),
        'funding':      funding,
        'funding_state': fstate,
        'funding_emoji': femoji,
        'lsr':          lsr,
        'top_lsr':      top_lsr,
        'lsr_state':    lstate,
        'squeeze':      squeeze,
        'rvol_1h':      round(rvol_1h, 2),
        'score':        score,
        'tf':           tf_data,
        'turnover24h':  item['turnover24h'],
    }


# ── OUTPUT TELEGRAM ─────────────────────────────────────────────────────
QUAD = {
    'LONG_BUILDUP':  ('🟢 LONG BUILDUP',
                      'Setup LONG — harga naik, OI naik: uang baru masuk posisi long'),
    'SHORT_BUILDUP': ('🔴 SHORT BUILDUP',
                      'Setup SHORT — harga turun, OI naik: uang baru masuk posisi short'),
}
FUND_LBL = {
    'BALANCED': 'seimbang', 'LONG_CROWDED': 'long ramai',
    'LONG_VERY_CROWDED': 'long SANGAT ramai', 'SHORT_CROWDED': 'short ramai',
    'SHORT_VERY_CROWDED': 'short SANGAT ramai', 'UNKNOWN': 'n/a',
}
LSR_LBL = {
    'LONG_HEAVY': 'akun mayoritas long', 'SHORT_HEAVY': 'akun mayoritas short',
    'BALANCED': 'akun seimbang', 'UNKNOWN': '',
}


def _fmt(r, i):
    t = r['tf'].get('1h', {})
    oi, pc = t.get('oi_chg'), t.get('price_chg')
    oi_s = f"{oi:+.2f}%" if oi is not None else "n/a"
    pc_s = f"{pc:+.2f}%" if pc is not None else "n/a"
    fr_s = f"{r['funding']*100:+.4f}%" if r['funding'] is not None else "n/a"
    lsr_s = f" | L/S {r['lsr']:.2f}" if r['lsr'] is not None else ""
    top_s = f" | topL/S {r['top_lsr']:.2f}" if r.get('top_lsr') is not None else ""
    return (
        f"{i}. <b>{r['symbol']}</b> · Skor {r['score']}/100\n"
        f"   1h: OI {oi_s} | Harga {pc_s} | RVOL {r['rvol_1h']}x\n"
        f"   TF konsisten {r['consistency']}/{r['tf_total']} | "
        f"Funding {fr_s} {r['funding_emoji']}{lsr_s}{top_s}"
    )


def build_message(results, bybit_filter_active):
    now = datetime.now(timezone.utc).astimezone()
    longs   = [r for r in results if r['quadrant'] == 'LONG_BUILDUP'  and not r['squeeze']]
    shorts  = [r for r in results if r['quadrant'] == 'SHORT_BUILDUP' and not r['squeeze']]
    squeeze = [r for r in results if r['squeeze']]

    filter_note = ("✅ difilter: hanya coin tradable di Bybit"
                   if bybit_filter_active
                   else "⚠️ filter Bybit TIDAK aktif — sebagian coin mungkin tidak ada di Bybit")

    p = [
        f"📡 <b>OI QUADRANT SCREENER</b>",
        f"🕐 {now.strftime('%d %b %Y, %H:%M')} WIB",
        f"📊 Data: Gate.io | {filter_note}",
        f"🎯 Kandidat: {len(results)} "
        f"(long {len(longs)} · short {len(shorts)} · squeeze {len(squeeze)})",
        "─" * 28,
    ]

    if not results:
        p.append("\nTidak ada kandidat lolos filter siklus ini.")
        return "\n".join(p)

    for bucket, key in ((longs, 'LONG_BUILDUP'), (shorts, 'SHORT_BUILDUP')):
        if not bucket:
            continue
        title, desc = QUAD[key]
        p.append(f"\n<b>{title}</b>")
        p.append(f"<i>{desc}</i>\n")
        for i, r in enumerate(bucket[:TOP_N_REPORT], 1):
            p.append(_fmt(r, i))

    if squeeze:
        p.append("\n<b>⚡ SQUEEZE WATCH</b>")
        p.append("<i>Ada buildup TAPI sisi itu sudah terlalu ramai (funding/LSR "
                 "ekstrem). Posisi rapuh: gerakan kecil melawan bisa memicu "
                 "liquidation cascade. Kandidat PENGAMATAN, bukan sinyal masuk.</i>\n")
        for i, r in enumerate(squeeze[:TOP_N_REPORT], 1):
            side = "long" if r['quadrant'] == 'LONG_BUILDUP' else "short"
            extra = LSR_LBL.get(r['lsr_state'], '')
            p.append(_fmt(r, i) + f"\n   ⚠️ sisi {side} crowded"
                                   + (f" · {extra}" if extra else ""))

    p.append("\n" + "─" * 28)
    p.append("<i>Deskriptif, bukan rekomendasi. Threshold belum dikalibrasi "
             "backtest. Cek chart sendiri sebelum ambil posisi.</i>")
    return "\n".join(p)


async def send_telegram(session, text):
    if not TG_TOKEN or not TG_CHAT_ID:
        logging.warning("OI_TELEGRAM_TOKEN / OI_TELEGRAM_CHAT_ID kosong — "
                        "hasil dicetak ke log saja.")
        print("\n" + text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3800:
            chunks.append(cur); cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    for ch in chunks:
        try:
            async with session.post(url, json={
                "chat_id": TG_CHAT_ID, "text": ch,
                "parse_mode": "HTML", "disable_web_page_preview": True
            }, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status != 200:
                    logging.error(f"Telegram HTTP {r.status}: {await r.text()}")
                else:
                    logging.info("Pesan terkirim ke Telegram.")
        except Exception as e:
            logging.error(f"Telegram error: {e}")
        await asyncio.sleep(0.4)


# ── MAIN ────────────────────────────────────────────────────────────────
async def main():
    async with aiohttp.ClientSession(headers={'Accept': 'application/json'}) as session:
 