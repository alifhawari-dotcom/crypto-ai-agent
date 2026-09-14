"""
oi_screener.py — Screener OI × Harga (4-Quadrant) + Funding Overlay
Modul TERPISAH dari screener.py (Apex Quant v7.1). Additive-only.

TUJUAN: screening MANUAL. Output = daftar kandidat + label setup untuk
kamu cek chart sendiri. BUKAN sinyal entry otomatis, BUKAN rekomendasi.

=============================================================================
DASAR KONSEPTUAL (dan batasnya — baca ini sebelum percaya outputnya)
=============================================================================

Framework 4-kuadran OI × Harga adalah kerangka MEKANIS yang diakui luas di
literatur trading derivatif:

    Harga ↑ + OI ↑ = LONG BUILDUP    → uang baru masuk long
    Harga ↓ + OI ↑ = SHORT BUILDUP   → uang baru masuk short
    Harga ↑ + OI ↓ = SHORT COVERING  → short keluar (bullish, kurang tahan lama)
    Harga ↓ + OI ↓ = LONG UNWINDING  → long keluar (lemah, bukan serangan bear)

Prinsip kunci: OI TANPA konteks harga tidak punya arah. Ini memperbaiki
cacat desain versi pertama file ini, yang hanya mengecek "OI naik" tanpa
melihat harga — sehingga long buildup dan short buildup tercampur jadi satu.

Funding rate adalah sinyal CROWDING, bukan sinyal ARAH. Funding ekstrem
(±0.05%–0.1% per 8 jam) menandakan satu sisi terlalu ramai dan rapuh:
gerakan kecil melawan bisa memicu liquidation cascade.

⚠️ YANG TIDAK DIKLAIM DI SINI:
   - Tidak ada bukti bahwa kombinasi ini menghasilkan edge/profit.
   - Semua THRESHOLD di bawah adalah ANGKA AWAL yang belum dikalibrasi
     dengan backtest. Perlakukan sebagai titik mulai untuk dikalibrasi
     sendiri dari data, bukan angka yang sudah terbukti.
   - Framework ini deskriptif (menggambarkan posisi pasar saat ini),
     bukan prediktif.
=============================================================================
"""

import os
import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Env var TERPISAH dari screener.py (TELEGRAM_TOKEN/TELEGRAM_CHAT_ID)
TG_TOKEN   = (os.getenv('OI_TELEGRAM_TOKEN')   or '').strip()
TG_CHAT_ID = (os.getenv('OI_TELEGRAM_CHAT_ID') or '').strip()

# ── THRESHOLD (BELUM DIKALIBRASI — kalibrasi sendiri dari data) ──────────
OI_CHANGE_MIN_PCT    = 2.0    # % perubahan OI minimum agar dianggap "bergerak"
PRICE_CHANGE_MIN_PCT = 0.5    # % perubahan harga minimum agar dianggap "bergerak"
RVOL_MIN             = 1.3    # volume relatif minimum (validasi sinyal)
FUNDING_EXTREME      = 0.0005 # 0.05% per 8 jam — batas "crowded"
FUNDING_VERY_EXTREME = 0.0010 # 0.10% per 8 jam — batas "sangat crowded"
MIN_TURNOVER_24H     = 3_000_000   # USD, buang koin terlalu tipis untuk ditradingkan
MIN_TF_CONSISTENCY   = 2      # kuadran sama harus muncul di >= N timeframe

# Timeframe yang dicek. Bybit intervalTime untuk OI: 5min,15min,30min,1h,4h,1d
TIMEFRAMES = [
    # (label, bybit_oi_interval, bybit_kline_interval, jumlah candle)
    ('15m', '15min', '15',  3),
    ('1h',  '1h',    '60',  3),
    ('4h',  '4h',    '240', 3),
]

SEMAPHORE_LIMIT = 6
MAX_RETRIES     = 3
RETRY_DELAY     = 4
TIMEOUT_SEC     = 15
TOP_N_REPORT    = 8     # berapa kandidat per kategori dikirim ke Telegram

BYBIT_BASE = "https://api.bybit.com/v5/market"


# ── HTTP ─────────────────────────────────────────────────────────────────
async def _get(session, path, params):
    url = f"{BYBIT_BASE}/{path}"
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC)) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get('retCode') == 0:
                        return data.get('result', {})
                    return None
                if r.status in (403, 451):
                    logging.error(f"HTTP {r.status} dari {url} — kemungkinan blokir region. "
                                  f"Cek apakah runner GitHub Actions kena geo-block Bybit.")
                    return None
        except Exception as e:
            logging.debug(f"fetch error {url}: {e}")
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY)
    return None


# ── UNIVERSE ─────────────────────────────────────────────────────────────
async def fetch_universe(session):
    """Semua USDT perpetual Bybit + turnover 24h, difilter likuiditas minimum.
    Universe LUAS (tanpa filter market cap) sesuai kebutuhan screening manual,
    tapi tetap buang yang terlalu tipis untuk ditradingkan."""
    res = await _get(session, "tickers", {"category": "linear"})
    if not res:
        return []
    out = []
    for it in res.get('list', []):
        sym = it.get('symbol', '')
        if not sym.endswith('USDT'):
            continue
        try:
            turnover = float(it.get('turnover24h') or 0)
        except (TypeError, ValueError):
            continue
        if turnover < MIN_TURNOVER_24H:
            continue
        out.append({'symbol': sym, 'turnover24h': turnover})
    out.sort(key=lambda x: x['turnover24h'], reverse=True)
    logging.info(f"Universe: {len(out)} pair USDT perp (turnover >= ${MIN_TURNOVER_24H:,.0f})")
    return out


# ── DATA PER TIMEFRAME ───────────────────────────────────────────────────
async def fetch_oi_change(session, symbol, oi_interval):
    """% perubahan OI antara 2 titik terakhir."""
    res = await _get(session, "open-interest", {
        "category": "linear", "symbol": symbol,
        "intervalTime": oi_interval, "limit": 2
    })
    if not res:
        return None
    rows = res.get('list', [])
    if len(rows) < 2:
        return None
    try:
        latest = float(rows[0]['openInterest'])   # Bybit: terbaru dulu
        prev   = float(rows[1]['openInterest'])
    except (KeyError, TypeError, ValueError):
        return None
    if prev <= 0:
        return None
    return ((latest - prev) / prev) * 100.0


async def fetch_kline_metrics(session, symbol, kline_interval, limit=25):
    """% perubahan harga candle terakhir + RVOL terhadap rata-rata 20."""
    res = await _get(session, "kline", {
        "category": "linear", "symbol": symbol,
        "interval": kline_interval, "limit": str(limit)
    })
    if not res:
        return None, None
    rows = res.get('list', [])
    if len(rows) < 21:
        return None, None
    rows = list(reversed(rows))  # jadikan kronologis
    try:
        closes = np.array([float(r[4]) for r in rows])
        vols   = np.array([float(r[5]) for r in rows])
    except (IndexError, TypeError, ValueError):
        return None, None
    if closes[-2] <= 0:
        return None, None
    price_chg = ((closes[-1] - closes[-2]) / closes[-2]) * 100.0
    vol_avg   = vols[-21:-1].mean()
    rvol      = float(vols[-1] / vol_avg) if vol_avg > 0 else 0.0
    return price_chg, rvol


async def fetch_funding(session, symbol):
    res = await _get(session, "funding/history", {
        "category": "linear", "symbol": symbol, "limit": 1
    })
    if not res:
        return None
    rows = res.get('list', [])
    if not rows:
        return None
    try:
        return float(rows[0]['fundingRate'])
    except (KeyError, TypeError, ValueError):
        return None


# ── KLASIFIKASI 4 KUADRAN ────────────────────────────────────────────────
def classify_quadrant(price_chg, oi_chg):
    """Kuadran OI × Harga. NEUTRAL kalau gerakan terlalu kecil untuk dibaca."""
    if price_chg is None or oi_chg is None:
        return None
    if abs(price_chg) < PRICE_CHANGE_MIN_PCT or abs(oi_chg) < OI_CHANGE_MIN_PCT:
        return 'NEUTRAL'
    if price_chg > 0 and oi_chg > 0:
        return 'LONG_BUILDUP'
    if price_chg < 0 and oi_chg > 0:
        return 'SHORT_BUILDUP'
    if price_chg > 0 and oi_chg < 0:
        return 'SHORT_COVERING'
    return 'LONG_UNWINDING'


def funding_state(fr):
    """Funding = sinyal CROWDING, bukan arah."""
    if fr is None:
        return 'UNKNOWN', ''
    if fr >= FUNDING_VERY_EXTREME:
        return 'LONG_VERY_CROWDED', '🔴'
    if fr >= FUNDING_EXTREME:
        return 'LONG_CROWDED', '🟠'
    if fr <= -FUNDING_VERY_EXTREME:
        return 'SHORT_VERY_CROWDED', '🔴'
    if fr <= -FUNDING_EXTREME:
        return 'SHORT_CROWDED', '🟠'
    return 'BALANCED', '🟢'


# ── SCREENING PER SYMBOL ─────────────────────────────────────────────────
async def screen_symbol(session, item, sem):
    symbol = item['symbol']
    async with sem:
        tf_results = {}
        for label, oi_int, kl_int, _ in TIMEFRAMES:
            oi_chg = await fetch_oi_change(session, symbol, oi_int)
            price_chg, rvol = await fetch_kline_metrics(session, symbol, kl_int)
            tf_results[label] = {
                'oi_chg':    oi_chg,
                'price_chg': price_chg,
                'rvol':      rvol,
                'quadrant':  classify_quadrant(price_chg, oi_chg),
            }
        funding = await fetch_funding(session, symbol)

    # Konsistensi: kuadran apa yang paling sering muncul lintas timeframe?
    quads = [v['quadrant'] for v in tf_results.values()
             if v['quadrant'] and v['quadrant'] != 'NEUTRAL']
    if not quads:
        return None
    dominant = max(set(quads), key=quads.count)
    consistency = quads.count(dominant)
    if consistency < MIN_TF_CONSISTENCY:
        return None
    if dominant not in ('LONG_BUILDUP', 'SHORT_BUILDUP'):
        return None   # sesuai keputusan: hanya tampilkan dua kuadran buildup

    # Validasi volume: OI bergerak tanpa volume = sinyal lemah
    # (bisa hasil satu-dua transaksi besar, bukan partisipasi pasar riil)
    rvol_1h = tf_results.get('1h', {}).get('rvol') or 0.0
    if rvol_1h < RVOL_MIN:
        return None

    fstate, femoji = funding_state(funding)

    # Squeeze watch: buildup yang funding-nya crowded SEARAH posisinya.
    # Long buildup + long crowded  → rally bergantung leverage, rawan cascade.
    # Short buildup + short crowded → short ramai, rawan short squeeze.
    squeeze_watch = (
        (dominant == 'LONG_BUILDUP'  and fstate in ('LONG_CROWDED',  'LONG_VERY_CROWDED')) or
        (dominant == 'SHORT_BUILDUP' and fstate in ('SHORT_CROWDED', 'SHORT_VERY_CROWDED'))
    )

    # Skor kualitas SINYAL (bukan skor profit) — seberapa bersih pembacaannya
    score = 0
    score += consistency * 15                      # konsistensi lintas TF
    score += min(int(rvol_1h * 10), 25)            # dukungan volume
    avg_oi = np.mean([abs(v['oi_chg']) for v in tf_results.values()
                      if v['oi_chg'] is not None])
    score += min(int(avg_oi * 2), 20)              # besarnya pergerakan OI
    if fstate == 'BALANCED':
        score += 10                                # tidak crowded = lebih bersih
    score = max(0, min(100, score))

    return {
        'symbol':      symbol,
        'quadrant':    dominant,
        'consistency': f"{consistency}/{len(TIMEFRAMES)}",
        'funding':     funding,
        'funding_state': fstate,
        'funding_emoji': femoji,
        'squeeze_watch': squeeze_watch,
        'rvol_1h':     round(rvol_1h, 2),
        'score':       score,
        'tf':          tf_results,
        'turnover24h': item['turnover24h'],
    }


# ── FORMAT TELEGRAM ──────────────────────────────────────────────────────
QUAD_LABEL = {
    'LONG_BUILDUP':  ('🟢 LONG BUILDUP',  'Setup LONG — uang baru masuk posisi long'),
    'SHORT_BUILDUP': ('🔴 SHORT BUILDUP', 'Setup SHORT — uang baru masuk posisi short'),
}

FUNDING_LABEL = {
    'BALANCED':           'seimbang',
    'LONG_CROWDED':       'long ramai',
    'LONG_VERY_CROWDED':  'long SANGAT ramai',
    'SHORT_CROWDED':      'short ramai',
    'SHORT_VERY_CROWDED': 'short SANGAT ramai',
    'UNKNOWN':            'n/a',
}


def _fmt_coin(r, idx):
    tf1h = r['tf'].get('1h', {})
    oi   = tf1h.get('oi_chg')
    pc   = tf1h.get('price_chg')
    fr_pct = f"{r['funding']*100:+.4f}%" if r['funding'] is not None else "n/a"
    return (
        f"{idx}. <b>{r['symbol']}</b> — Skor {r['score']}/100\n"
        f"   1h: OI {oi:+.2f}% | Harga {pc:+.2f}% | RVOL {r['rvol_1h']}x\n"
        f"   Konsistensi TF: {r['consistency']} | "
        f"Funding {fr_pct} {r['funding_emoji']} ({FUNDING_LABEL[r['funding_state']]})"
    )


def build_message(results):
    now = datetime.now(timezone.utc).astimezone()
    header = (
        f"📡 <b>OI QUADRANT SCREENER</b>\n"
        f"🕐 {now.strftime('%d %b %Y, %H:%M')} | Bybit USDT-Perp\n"
        f"📊 Kandidat: {len(results)}\n"
        f"{'─'*30}"
    )
    if not results:
        return header + "\n\nTidak ada kandidat yang lolos filter siklus ini."

    longs   = [r for r in results if r['quadrant'] == 'LONG_BUILDUP'  and not r['squeeze_watch']]
    shorts  = [r for r in results if r['quadrant'] == 'SHORT_BUILDUP' and not r['squeeze_watch']]
    squeeze = [r for r in results if r['squeeze_watch']]

    parts = [header]

    for bucket, key in ((longs, 'LONG_BUILDUP'), (shorts, 'SHORT_BUILDUP')):
        if not bucket:
            continue
        title, desc = QUAD_LABEL[key]
        parts.append(f"\n<b>{title}</b>\n<i>{desc}</i>\n")
        for i, r in enumerate(bucket[:TOP_N_REPORT], 1):
            parts.append(_fmt_coin(r, i))

    if squeeze:
        parts.append(
            "\n<b>⚡ SQUEEZE WATCH</b>\n"
            "<i>Buildup TAPI funding ekstrem — sisi yang ramai ini rapuh. "
            "Gerakan kecil melawan bisa memicu liquidation cascade. "
            "Bukan sinyal masuk, ini kandidat pengamatan.</i>\n"
        )
        for i, r in enumerate(squeeze[:TOP_N_REPORT], 1):
            side = "long" if r['quadrant'] == 'LONG_BUILDUP' else "short"
            parts.append(_fmt_coin(r, i) + f"\n   ⚠️ sisi {side} crowded")

    parts.append(
        f"\n{'─'*30}\n"
        "<i>Screening deskriptif, bukan rekomendasi. "
        "Threshold belum dikalibrasi backtest — cek chart sendiri sebelum ambil posisi.</i>"
    )
    return "\n".join(parts)


async def send_telegram(session, text):
    if not TG_TOKEN or not TG_CHAT_ID:
        logging.warning("OI_TELEGRAM_TOKEN / OI_TELEGRAM_CHAT_ID kosong — "
                        "hasil dicetak ke log saja.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    # Telegram batas ~4096 char, pecah kalau kepanjangan
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
                    logging.error(f"Telegram gagal: HTTP {r.status} — {await r.text()}")
        except Exception as e:
            logging.error(f"Telegram error: {e}")
        await asyncio.sleep(0.4)


# ── MAIN ─────────────────────────────────────────────────────────────────
async def main():
    async with aiohttp.ClientSession() as session:
        universe = await fetch_universe(session)
        if not universe:
            logging.error("Universe kosong. Kalau ini karena HTTP 403/451, "
                          "artinya IP runner kena geo-block Bybit — "
                          "lihat log di atas.")
            return

        sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
        tasks = [screen_symbol(session, it, sem) for it in universe]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for r in raw:
            if isinstance(r, Exception):
                logging.debug(f"task error: {r}")
                continue
            if r:
                results.append(r)

        results.sort(key=lambda x: x['score'], reverse=True)
        logging.info(f"Kandidat lolos: {len(results)} dari {len(universe)} pair")

        msg = build_message(results)
        await send_telegram(session, msg)


if __name__ == "__main__":
    asyncio.run(main())
