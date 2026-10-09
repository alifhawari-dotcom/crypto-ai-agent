"""Money Flow Screener — deteksi inflow/outflow taker yang TIDAK BIASA per koin.

TUJUAN: deteksi, bukan prediksi arah. Alert berarti "di koin ini barusan ada
aliran agresif yang jauh di atas kebiasaannya sendiri". Dipakai untuk
menemukan koin yang sedang hidup dan berpotensi volatil.

DATA (Gate.io, sama dengan OI screener, lolos dari GitHub Actions):
  - candlesticks 15m  -> 'sum' = volume dalam USDT per bar
  - contract_stats 15m -> 'lsr_taker' = rasio taker long/short,
                          'open_interest_usd', 'long_liq_usd', 'short_liq_usd'

ESTIMASI DELTA TAKER per bar 15m:
  share_buy = r / (1 + r), r = lsr_taker
  delta_usd = volume_usd * (2 * share_buy - 1)
  CATATAN: makna persis lsr_taker (rasio volume vs rasio jumlah) BELUM
  diverifikasi. Jalankan sekali dengan FLOW_DIAGNOSTIC=true untuk melihat
  data mentahnya sebelum alert dipercaya.

ATURAN ALERT (biner, bisa diuji):
  1. delta 1 jam (4 bar 15m tertutup) dalam USD
  2. z-score delta 1 jam vs histori koin itu sendiri (~24 jam sebelumnya)
  3. lolos jika |z| >= Z_MIN DAN |delta| >= MIN_FLOW_USD
     DAN imbalance (|delta|/volume) >= MIN_IMB
  4. FRESH: jam sebelumnya (4 bar sebelumnya) TIDAK lolos -> tanpa file state

Konteks di pesan (bukan filter): arah harga 1 jam, perubahan OI 1 jam,
porsi likuidasi. Kirim ke bot yang SAMA dengan OI screener.
"""
import os, asyncio, logging, json, aiohttp, numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
# Bot tujuan: bot DD (Decision Dashboard). Kalau secret DD kosong,
# otomatis jatuh ke bot OI supaya alert tidak hilang.
TG_TOKEN = (os.getenv('DD_TELEGRAM_TOKEN') or os.getenv('OI_TELEGRAM_TOKEN') or '').strip()
TG_CHAT  = (os.getenv('DD_TELEGRAM_CHAT_ID') or os.getenv('OI_TELEGRAM_CHAT_ID') or '').strip()
DIAG     = os.getenv('FLOW_DIAGNOSTIC', '').strip().lower() == 'true'

# ---------- AMBANG (titik awal, BELUM dioptimasi) ----------
Z_MIN        = 3.0          # seberapa luar biasa dibanding kebiasaan koin itu
MIN_FLOW_USD = 250_000      # nilai minimum agar koin tipis tidak spam
MIN_IMB      = 0.10         # |delta| / volume 1 jam minimal 10%
BARS         = 100          # 15m x 100 = 25 jam histori
WIN          = 4            # 4 bar = 1 jam
MIN_TURNOVER = 1_000_000
MAX_SYM, TOPN = 400, 10
SEM_N, RETRIES, DELAY, TMO = 12, 3, 3, 15

GATE  = "https://api.gateio.ws/api/v4/futures/usdt"
BYBIT = "https://api.bybit.com/v5/market"
BYBIT_SYMBOLS_FILE = "bybit_symbols.json"

# Sama dengan oi_screener.py
NON_CRYPTO = {
    'NVDA','META','MU','SOXL','SOXS','AVGO','ORCL','IBM','AAOI','INTC','AMD',
    'TSLA','AAPL','MSFT','GOOGL','AMZN','NFLX','COIN','MSTR','CRWV','ASML',
    'SNDK','WDC','SKHYNIX','SKHY','SAMSUNG','CXMT','DRAM','4STOCK','LITE',
    'CRCL','OPENAI','ANTHROPIC','SPX','QQQ','ESPORTS','MVLL','MET','RAVE',
    'COHR','QCOM','GLW','SPCX','SNXX','BSP','AKE','TSM','ARM','PLTR','SMCI',
    'DELL','HPQ','STX','KLAC','LRCX','AMAT','NXPI','ADI','TXN','ON','MCHP',
    'SWKS','QRVO','MRVL','ALAB','CRDO','ANET','CIEN','JNPR','ERIC','NOK','ZM',
    'SNOW','DDOG','NET','CRWD','PANW','ZS','OKTA','MDB','TEAM','NOW','WDAY',
    'ADBE','IREN','NBIS','ASTS','KORU','RKLB','BEAT','UB','BZ','SAGA','HEMI',
    'XAU','XAG','XAUT','PAXG','OIL','GOLD','SILVER',
}


async def _get(s, url, p=None, lbl=""):
    for a in range(RETRIES):
        try:
            async with s.get(url, params=p, timeout=aiohttp.ClientTimeout(total=TMO)) as r:
                if r.status == 200:
                    return await r.json()
                if r.status in (403, 451):
                    logging.warning(f"HTTP {r.status} dari {lbl} - geo-block.")
                    return None
                if r.status == 429:
                    await asyncio.sleep(DELAY * 2); continue
        except Exception as e:
            logging.debug(f"err {lbl}: {e}")
        if a < RETRIES - 1:
            await asyncio.sleep(DELAY)
    return None


def load_bybit_symbols_local():
    try:
        with open(BYBIT_SYMBOLS_FILE) as f:
            data = json.load(f)
        syms = set(data.get('symbols', []))
        if syms:
            logging.info(f"Bybit: {len(syms)} simbol dari file lokal")
            return syms
    except FileNotFoundError:
        logging.warning(f"{BYBIT_SYMBOLS_FILE} tidak ditemukan - filter Bybit NONAKTIF.")
    except (json.JSONDecodeError, KeyError) as e:
        logging.warning(f"{BYBIT_SYMBOLS_FILE} rusak: {e}")
    return None


async def universe(s, bsyms):
    d = await _get(s, f"{GATE}/tickers", lbl="Gate tickers")
    if not d:
        return []
    out = []
    for i in d:
        c = i.get('contract', '')
        if not c.endswith('_USDT'):
            continue
        base = c.replace('_USDT', '')
        if base in NON_CRYPTO:
            continue
        bs = c.replace('_', '')
        if bsyms is not None and bs not in bsyms:
            continue
        try:
            tv = float(i.get('volume_24h_quote') or 0)
            lp = float(i.get('last') or 0)
        except (TypeError, ValueError):
            continue
        if tv < MIN_TURNOVER or lp <= 0:
            continue
        out.append({'c': c, 'sym': bs, 'tv': tv})
    out.sort(key=lambda x: x['tv'], reverse=True)
    out = out[:MAX_SYM]
    logging.info(f"Universe: {len(out)} pair")
    return out


def _f(d, k):
    try:
        v = d.get(k)
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


async def fetch_bars(s, c):
    """Gabungkan candle 15m dan contract_stats 15m berdasarkan timestamp.
    Candle terakhir (masih berjalan) dibuang."""
    cd = await _get(s, f"{GATE}/candlesticks",
                    {"contract": c, "interval": "15m", "limit": BARS + 1}, "Gate candles")
    st = await _get(s, f"{GATE}/contract_stats",
                    {"contract": c, "interval": "15m", "limit": BARS + 1}, "Gate stats")
    if not isinstance(cd, list) or not isinstance(st, list) or len(cd) < 30 or len(st) < 30:
        return None
    try:
        cd = sorted(cd, key=lambda x: int(x.get('t', 0)))[:-1]   # buang candle berjalan
    except (TypeError, ValueError):
        return None
    smap = {}
    for x in st:
        try:
            smap[int(x.get('time', 0))] = x
        except (TypeError, ValueError):
            continue

    rows = []
    for k in cd:
        try:
            t = int(k['t']); vol = float(k.get('sum') or 0); cl = float(k['c'])
        except (KeyError, TypeError, ValueError):
            continue
        x = smap.get(t)
        if x is None:
            continue
        r = _f(x, 'lsr_taker')
        if r is None or r <= 0 or vol <= 0:
            continue
        share = r / (1.0 + r)
        rows.append({
            't': t, 'close': cl, 'vol': vol,
            'delta': vol * (2.0 * share - 1.0),
            'oi': _f(x, 'open_interest_usd'),
            'lliq': _f(x, 'long_liq_usd') or 0.0,
            'sliq': _f(x, 'short_liq_usd') or 0.0,
        })
    return rows if len(rows) >= 30 else None


def window_sum(vals, end, n=WIN):
    """Jumlah n nilai berakhir di indeks end (inklusif)."""
    return float(np.sum(vals[end - n + 1:end + 1]))


def evaluate(rows, end):
    """Hitung delta 1 jam yang berakhir di bar 'end' dan z-score-nya
    terhadap jendela 1 jam sebelumnya (tidak tumpang tindih dengan jam ini)."""
    d = np.array([r['delta'] for r in rows])
    v = np.array([r['vol'] for r in rows])
    if end - WIN + 1 < WIN * 4:
        return None
    hist = [window_sum(d, i) for i in range(WIN - 1, end - WIN + 1)]
    sd = float(np.std(hist))
    mu = float(np.mean(hist))
    if sd <= 0:
        return None
    dl = window_sum(d, end)
    vl = window_sum(v, end)
    z = (dl - mu) / sd
    imb = dl / vl if vl > 0 else 0.0
    hit = abs(z) >= Z_MIN and abs(dl) >= MIN_FLOW_USD and abs(imb) >= MIN_IMB
    return {'delta': dl, 'vol': vl, 'z': z, 'imb': imb, 'hit': hit}


async def screen(s, it, sem, st):
    async with sem:
        rows = await fetch_bars(s, it['c'])
    if not rows:
        st['data_kurang'] += 1
        return None
    last = len(rows) - 1
    now = evaluate(rows, last)
    prev = evaluate(rows, last - WIN)
    if now is None:
        st['data_kurang'] += 1
        return None
    st['top'].append((it['sym'], now['z'], now['delta'], now['imb']))
    if not now['hit']:
        st['normal'] += 1
        return None
    if prev is not None and prev['hit'] and np.sign(prev['delta']) == np.sign(now['delta']):
        st['bukan_fresh'] += 1
        return None
    st['lolos'] += 1

    c0, c1 = rows[last - WIN]['close'], rows[last]['close']
    px = (c1 - c0) / c0 * 100.0 if c0 > 0 else 0.0
    o0, o1 = rows[last - WIN]['oi'], rows[last]['oi']
    oi = (o1 - o0) / o0 * 100.0 if o0 and o1 and o0 > 0 else None
    lliq = sum(r['lliq'] for r in rows[last - WIN + 1:last + 1])
    sliq = sum(r['sliq'] for r in rows[last - WIN + 1:last + 1])
    return {'sym': it['sym'], 'tv': it['tv'], 'px': px, 'oi': oi,
            'lliq': lliq, 'sliq': sliq, **now}


def read_flow(r):
    """Tafsiran kombinasi arah flow x OI. Deskriptif, bukan sinyal entry."""
    inflow = r['delta'] > 0
    if r['oi'] is None:
        return 'OI tidak tersedia'
    oi_up = r['oi'] > 0
    if inflow and oi_up:       return 'long baru agresif masuk'
    if inflow and not oi_up:   return 'short ditutup paksa/covering'
    if not inflow and oi_up:   return 'short baru agresif masuk'
    return 'long keluar/dilepas'


def fmt_usd(x):
    a = abs(x)
    if a >= 1e9: return f"{x/1e9:+.2f}B"
    if a >= 1e6: return f"{x/1e6:+.2f}M"
    return f"{x/1e3:+.0f}K"


def build(res, st, n_uni):
    head = "💧 <b>MONEY FLOW SCREENER</b> (1 jam terakhir)"
    if not res:
        return None   # tidak kirim apa-apa kalau tidak ada flow luar biasa
    res.sort(key=lambda r: abs(r['z']), reverse=True)
    lines = [head, f"Universe {n_uni} pair | lolos {len(res)}", ""]
    for r in res[:TOPN]:
        inflow = r['delta'] > 0
        tag = "🟢 INFLOW" if inflow else "🔴 OUTFLOW"
        oi_txt = f"{r['oi']:+.2f}%" if r['oi'] is not None else "n/a"
        liq = r['sliq'] if inflow else r['lliq']
        liq_pct = liq / abs(r['delta']) * 100.0 if r['delta'] else 0.0
        liq_note = f" | likuidasi {liq_pct:.0f}% dari flow" if liq_pct >= 20 else ""
        lines.append(f"<b>{r['sym']}</b> {tag}")
        lines.append(f"  Flow {fmt_usd(r['delta'])} | z {r['z']:+.1f} | imbalance {r['imb']*100:+.0f}%")
        lines.append(f"  Harga {r['px']:+.2f}% | OI {oi_txt}{liq_note}")
        lines.append(f"  → {read_flow(r)}")
        lines.append("")
    lines.append("<i>Deteksi aliran tidak biasa, bukan sinyal entry. Validasi struktur di chart.</i>")
    return "\n".join(lines)


async def send_tg(s, text):
    if not TG_TOKEN or not TG_CHAT:
        logging.warning("Token/chat Telegram kosong - pesan hanya dicetak.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    async with s.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=TMO)) as r:
        if r.status != 200:
            logging.error(f"Telegram HTTP {r.status}: {await r.text()}")


async def diagnostic(s):
    """Cetak data mentah BTC & ETH untuk memverifikasi makna lsr_taker."""
    for c in ("BTC_USDT", "ETH_USDT"):
        st = await _get(s, f"{GATE}/contract_stats",
                        {"contract": c, "interval": "15m", "limit": 5}, "diag stats")
        cd = await _get(s, f"{GATE}/candlesticks",
                        {"contract": c, "interval": "15m", "limit": 5}, "diag candles")
        print(f"=== {c} contract_stats ===")
        print(json.dumps(st, indent=1)[:3000])
        print(f"=== {c} candlesticks ===")
        print(json.dumps(cd, indent=1)[:2000])


async def main():
    async with aiohttp.ClientSession() as s:
        if DIAG:
            await diagnostic(s)
        uni = await universe(s, load_bybit_symbols_local())
        if not uni:
            logging.error("Universe kosong - berhenti.")
            return
        st = {'data_kurang': 0, 'normal': 0, 'bukan_fresh': 0, 'lolos': 0, 'top': []}
        sem = asyncio.Semaphore(SEM_N)
        out = await asyncio.gather(*[screen(s, it, sem, st) for it in uni])
        res = [r for r in out if r]
        top = sorted(st.pop('top'), key=lambda x: abs(x[1]), reverse=True)[:8]
        logging.info(f"Statistik: {st}")
        # Kalibrasi: koin dengan |z| tertinggi siklus ini, lolos atau tidak
        for sym, z, dl, imb in top:
            logging.info(f"  TOP z {sym:<14} z={z:+.2f}  flow={fmt_usd(dl)}  imb={imb*100:+.0f}%")
        msg = build(res, st, len(uni))
        if msg:
            await send_tg(s, msg)
        else:
            logging.info("Tidak ada flow luar biasa siklus ini - tidak kirim pesan.")


if __name__ == "__main__":
    asyncio.run(main())
