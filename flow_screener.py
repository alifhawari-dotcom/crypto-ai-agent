"""Smart Flow Screener — deteksi jejak aliran besar, lalu korelasikan dengan tren 4H,
level S/R, dan OI untuk memperkirakan arah, target, dan invalidasi.

MENGGANTIKAN flow_screener.py yang lama (nama file sama), jadi workflow
oi.screener.yml TIDAK perlu diubah: step "Run Money Flow Screener" tetap
`python flow_screener.py`, bot tujuan tetap DD_TELEGRAM_* (fallback OI_TELEGRAM_*).
Versi lama tetap ada di riwayat git bila perlu dikembalikan.

ALUR (dua tahap, supaya hemat request):
  TAHAP 1  Anomali flow per koin dari candle 15m + contract_stats Gate.io
           (delta taker z-score, lonjakan volume, perubahan OI, likuidasi,
           effort-vs-result). Hanya koin yang anomali lanjut ke tahap 2.
  TAHAP 2  Konteks dari candle 4H + 1H Gate.io:
           - Tren mayor 4H : voting EMA (default 50/100/200) dengan band ATR
           - Level S/R     : swing 4H & 1H yang terkonfirmasi (tanpa lookahead),
                             dikelompokkan; ekstrem range 4H ikut dihitung
           - Pola          : SWEEP+RECLAIM, ABSORPSI (jual/beli agresif terserap di
                             level), BREAKOUT (OI naik = posisi baru, OI turun = lemah)
           - Zona OI       : area harga tempat OI menumpuk (25 jam terakhir)
           - Skor 0-5      : tren, lokasi, struktur, flow, OI. Alert jika >= MIN_SCORE
           - Target        : level kuat terdekat berikutnya (T1, T2), + invalidasi

PENTING (jujur soal batasan):
  * SEMUA ambang di bawah adalah HEURISTIK awal, BELUM divalidasi backtest.
  * Pesan menyebut BUKTI (z, OI, likuidasi, level), bukan klaim "smart money masuk".
  * Delta taker memakai asumsi lsr_taker yang sama dengan flow_screener; maknanya
    belum diverifikasi (jalankan SMART_DIAGNOSTIC=true sekali).
  * Zona OI hanya mencakup ~25 jam (100 bar 15m). Sisi long/short OI tidak diketahui.

DEDUPE: state di smart_state.json (kunci simbol|pola|arah, kedaluwarsa SMART_DEDUPE_H jam).
  Agar persisten antar-run di GitHub Actions, commit file ini seperti regime_state.json
  (lihat contoh langkah workflow di bagian bawah file ini). Tanpa itu, alert yang
  sama bisa berulang selama kondisinya bertahan.

ENV (semua opsional):
  DD_TELEGRAM_TOKEN / DD_TELEGRAM_CHAT_ID  (fallback OI_TELEGRAM_*)
  SMART_DIAGNOSTIC=true   cetak data mentah BTC/ETH + kandidat di bawah ambang
  SMART_MIN_SCORE=3  SMART_DEDUPE_H=4  SMART_TREND_EMAS=50,100,200

Uji logika offline:  python flow_screener.py --selftest

Satu-satunya tambahan di workflow (setelah step "Run Money Flow Screener"):
      - name: Commit smart state
        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add smart_state.json
          git diff --staged --quiet || git commit -m "chore: smart state [skip ci]"
          git push
"""
import os, sys, json, time, asyncio, logging
import numpy as np
import pandas as pd
import aiohttp

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
TG_TOKEN = (os.getenv('DD_TELEGRAM_TOKEN') or os.getenv('OI_TELEGRAM_TOKEN') or '').strip()
TG_CHAT = (os.getenv('DD_TELEGRAM_CHAT_ID') or os.getenv('OI_TELEGRAM_CHAT_ID') or '').strip()
DIAG = os.getenv('SMART_DIAGNOSTIC', '').strip().lower() == 'true'


def _envf(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


# ───────────── AMBANG (heuristik awal, BELUM dioptimasi) ─────────────
# Tahap 1: anomali flow (lebih longgar dari flow_screener; konteks yang menyaring)
Z_LOOSE, VR_LOOSE, MIN_IMB, MIN_FLOW_USD = 2.0, 1.3, 0.08, 50_000
LIQ_MIN_USD, LIQ_MULT = 100_000, 3.0
Z_CONF = 2.5                    # flow dianggap "mengonfirmasi" jika |z| >= ini
ABS_VR, ABS_RES = 1.5, 0.7      # absorpsi: volume >= 1.5x tapi gerak harga <= 0.7x median
OI_UP, OI_FLUSH = 0.3, 0.5      # % perubahan OI 1 jam: posisi baru / posisi dilepas
# Tahap 2
SW4, SW1 = 3, 5                 # panjang fractal swing 4H dan 1H
RANGE_N4 = 100                  # ekstrem range 4H (bar)
LEVEL_TOL_ATR4 = 0.3            # toleransi pengelompokan level (x ATR 4H)
MIN_STRENGTH = 2.0              # level "kuat" jika bobot >= ini
NEAR_ATR = 0.8                  # "dekat level" = dalam 0.8 ATR(1H)
MIN_T_ATR = 0.6                 # target minimal sejauh 0.6 ATR(1H)
ZONE_SHARE = 0.25               # zona OI = bucket dengan >= 25% dari total OI baru
ER_RANGE = 0.25                 # efficiency ratio 4H di bawah ini = sideways
TREND_BAND = 0.25               # band ATR(4H) di sekitar EMA
TREND_EMAS = tuple(int(x) for x in os.getenv('SMART_TREND_EMAS', '50,100,200').split(',') if x.strip())
MIN_SCORE = int(_envf('SMART_MIN_SCORE', 3))
DEDUPE_H = _envf('SMART_DEDUPE_H', 4)
FUND_EXT, LSR_HI, LSR_LO = 0.0005, 2.0, 0.5   # sama dengan oi_screener.py
PX_MIN = 0.15                   # gerak harga 1j (%) minimal agar dianggap 'bergerak'
MIN_RR = _envf('SMART_MIN_RR', 1.2)             # R:R minimal ke T1 (jika ada target)
MAX_INVAL_ATR = _envf('SMART_MAX_INVAL_ATR', 4.0)  # jarak invalidasi maks (x ATR 1H)
# Retest setelah breakout
RETEST_LOOK_H = 30              # breakout dicari dalam N jam 1H terakhir
BRK_FOLLOW_ATR = 1.0            # breakout sah bila sempat lanjut >= 1 ATR(1H) melewati zona
RETEST_TOL = 0.3                # 'menyentuh zona' = low/high dalam 0.3 ATR(1H) dari tepi zona
RETEST_GATE_PCT = 4.0           # hanya hemat API: koin tanpa anomali dianalisis bila range 25 jam >= 4%
RETEST_MAX = 30                 # maks koin non-anomali yang dianalisis untuk retest
RETEST_ON = _envf('SMART_RETEST', 1) > 0   # SMART_RETEST=0 mematikan seluruh fitur retest
# Operasional
BARS, WIN = 100, 4              # 15m x 100; 4 bar = 1 jam
N4, N1 = 300, 240               # jumlah candle 4H dan 1H
MIN_TURNOVER, MAX_SYM, STAGE2_MAX, TOPN = 1_000_000, 400, 40, 6
SEM_N, RETRIES, DELAY, TMO = 12, 3, 3, 15
STATE_FILE = 'smart_state.json'

GATE = "https://api.gateio.ws/api/v4/futures/usdt"
BYBIT_SYMBOLS_FILE = "bybit_symbols.json"
SECS = {'1h': 3600, '4h': 14400}
NON_CRYPTO = {
    'NVDA', 'META', 'MU', 'SOXL', 'SOXS', 'AVGO', 'ORCL', 'IBM', 'AAOI', 'INTC', 'AMD',
    'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NFLX', 'COIN', 'MSTR', 'CRWV', 'ASML',
    'SNDK', 'WDC', 'SKHYNIX', 'SKHY', 'SAMSUNG', 'CXMT', 'DRAM', '4STOCK', 'LITE',
    'CRCL', 'OPENAI', 'ANTHROPIC', 'SPX', 'QQQ', 'ESPORTS', 'MVLL', 'MET', 'RAVE',
    'COHR', 'QCOM', 'GLW', 'SPCX', 'SNXX', 'BSP', 'AKE', 'TSM', 'ARM', 'PLTR', 'SMCI',
    'DELL', 'HPQ', 'STX', 'KLAC', 'LRCX', 'AMAT', 'NXPI', 'ADI', 'TXN', 'ON', 'MCHP',
    'SWKS', 'QRVO', 'MRVL', 'ALAB', 'CRDO', 'ANET', 'CIEN', 'JNPR', 'ERIC', 'NOK', 'ZM',
    'SNOW', 'DDOG', 'NET', 'CRWD', 'PANW', 'ZS', 'OKTA', 'MDB', 'TEAM', 'NOW', 'WDAY',
    'ADBE', 'IREN', 'NBIS', 'ASTS', 'KORU', 'RKLB', 'BEAT', 'UB', 'BZ', 'SAGA', 'HEMI',
    'XAU', 'XAG', 'XAUT', 'PAXG', 'OIL', 'GOLD', 'SILVER',
}


# ───────────────────────────── jaringan ─────────────────────────────
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
                    await asyncio.sleep(DELAY * 2)
                    continue
        except Exception as e:
            logging.debug(f"err {lbl}: {e}")
        if a < RETRIES - 1:
            await asyncio.sleep(DELAY)
    return None


def _f(d, k):
    try:
        v = d.get(k)
        return float(v) if v is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def load_bybit_symbols_local():
    try:
        with open(BYBIT_SYMBOLS_FILE) as f:
            syms = set(json.load(f).get('symbols', []))
        return syms or None
    except (FileNotFoundError, json.JSONDecodeError):
        logging.warning(f"{BYBIT_SYMBOLS_FILE} tidak ada/rusak - filter Bybit NONAKTIF.")
        return None


async def universe(s, bsyms):
    d = await _get(s, f"{GATE}/tickers", lbl="Gate tickers")
    if not d:
        return []
    out = []
    for i in d:
        c = i.get('contract', '')
        if not c.endswith('_USDT') or c.replace('_USDT', '') in NON_CRYPTO:
            continue
        bs = c.replace('_', '')
        if bsyms is not None and bs not in bsyms:
            continue
        tv, lp, fr = _f(i, 'volume_24h_quote') or 0, _f(i, 'last') or 0, _f(i, 'funding_rate')
        if tv < MIN_TURNOVER or lp <= 0:
            continue
        out.append({'c': c, 'sym': bs, 'tv': tv, 'fr': fr})
    out.sort(key=lambda x: x['tv'], reverse=True)
    return out[:MAX_SYM]


async def fetch_15m(s, c):
    """Candle 15m + contract_stats 15m digabung per timestamp. Candle berjalan dibuang."""
    cd = await _get(s, f"{GATE}/candlesticks",
                    {"contract": c, "interval": "15m", "limit": BARS + 1}, "Gate candles 15m")
    st = await _get(s, f"{GATE}/contract_stats",
                    {"contract": c, "interval": "15m", "limit": BARS + 1}, "Gate stats")
    if not isinstance(cd, list) or not isinstance(st, list) or len(cd) < 30 or len(st) < 30:
        return None
    try:
        cd = sorted(cd, key=lambda x: int(x.get('t', 0)))[:-1]
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
            t, vol = int(k['t']), float(k.get('sum') or 0)
            o, h, l, cl = float(k['o']), float(k['h']), float(k['l']), float(k['c'])
        except (KeyError, TypeError, ValueError):
            continue
        x = smap.get(t)
        if x is None:
            continue
        r = _f(x, 'lsr_taker')
        if r is None or r <= 0 or vol <= 0:
            continue
        share = r / (1.0 + r)
        rows.append({'t': t, 'open': o, 'high': h, 'low': l, 'close': cl, 'vol': vol,
                     'delta': vol * (2.0 * share - 1.0),     # ASUMSI, lihat docstring
                     'oi': _f(x, 'open_interest_usd'),
                     'lliq': _f(x, 'long_liq_usd') or 0.0, 'sliq': _f(x, 'short_liq_usd') or 0.0,
                     'lsr_acc': _f(x, 'lsr_account')})
    return rows if len(rows) >= 30 else None


async def fetch_candles(s, c, iv, limit):
    """Candle 1h/4h tertutup sebagai dict array. Candle berjalan dibuang."""
    d = await _get(s, f"{GATE}/candlesticks",
                   {"contract": c, "interval": iv, "limit": limit}, f"Gate candles {iv}")
    if not isinstance(d, list):
        return None
    try:
        d = sorted(d, key=lambda x: int(x.get('t', 0)))
        now = time.time()
        d = [x for x in d if int(x['t']) + SECS[iv] <= now]
        a = {k: np.array([float(x[k]) for x in d]) for k in ('o', 'h', 'l', 'c')}
        a['t'] = np.array([int(x['t']) for x in d])
    except (KeyError, TypeError, ValueError):
        return None
    return a if len(a['c']) >= (120 if iv == '4h' else 60) else None


# ───────────────────────────── indikator ─────────────────────────────
def ema(a, n):
    return pd.Series(a).ewm(span=n, adjust=False).mean().values


def atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().values


def eff_ratio(c, n=20):
    if len(c) <= n:
        return None
    noise = np.abs(np.diff(c[-n - 1:])).sum()
    return float(abs(c[-1] - c[-1 - n]) / noise) if noise > 0 else 0.0


def swings(h, l, sw):
    """Pivot terkonfirmasi: butuh sw bar di kanan -> tidak repaint, tanpa lookahead."""
    hi, lo = [], []
    for k in range(sw, len(h) - sw):
        if h[k] > h[k - sw:k].max() and h[k] >= h[k + 1:k + sw + 1].max():
            hi.append((k, float(h[k])))
        if l[k] < l[k - sw:k].min() and l[k] <= l[k + 1:k + sw + 1].min():
            lo.append((k, float(l[k])))
    return hi, lo


# ───────────────────────────── tahap 1: flow ─────────────────────────────
def flow_eval(rows, end):
    """Metrik flow untuk jendela 1 jam yang berakhir di bar `end`, dibanding histori
    koin itu sendiri (jendela sebelumnya, tidak tumpang tindih)."""
    n = len(rows)
    if end - WIN + 1 < WIN * 4 or end >= n:
        return None
    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    d, v, o, c = g('delta'), g('vol'), g('open'), g('close')
    ll, sl = g('lliq'), g('sliq')
    cs = lambda x: np.r_[0.0, np.cumsum(x)]
    cd_, cv_, cl_, cs_ = cs(d), cs(v), cs(ll), cs(sl)
    ws = lambda cum, e: cum[e + 1] - cum[e + 1 - WIN]
    ends = range(WIN - 1, end - WIN + 1)
    hd = np.array([ws(cd_, e) for e in ends]); hv = np.array([ws(cv_, e) for e in ends])
    hl = np.array([ws(cl_, e) + ws(cs_, e) for e in ends])
    sd_ = float(np.std(hd))
    if sd_ <= 0 or len(hd) < 8:
        return None
    dl, vl = ws(cd_, end), ws(cv_, end)
    z = (dl - float(np.mean(hd))) / sd_
    vr = vl / float(np.mean(hv)) if np.mean(hv) > 0 else 0.0
    net = c[end] - o[end - WIN + 1]
    hnet = np.array([abs(c[e] - o[e - WIN + 1]) for e in ends])
    med = float(np.median(hnet))
    oi0, oi1 = rows[end - WIN].get('oi'), rows[end].get('oi')
    oi_chg = (oi1 - oi0) / oi0 * 100.0 if oi0 and oi1 and oi0 > 0 else None
    lq, sq = ws(cl_, end), ws(cs_, end)
    hl_mean = float(np.mean(hl)) if len(hl) else 0.0
    liq_spike = (lq + sq) >= max(LIQ_MIN_USD, LIQ_MULT * hl_mean) and max(lq, sq) >= LIQ_MIN_USD
    return {'delta': float(dl), 'vol': float(vl), 'z': float(z), 'vr': float(vr),
            'imb': float(dl / vl) if vl > 0 else 0.0,
            'px_chg': float(net / o[end - WIN + 1] * 100.0) if o[end - WIN + 1] > 0 else 0.0,
            'res': float(abs(net) / med) if med > 0 else None,
            'oi_chg': oi_chg, 'lliq': float(lq), 'sliq': float(sq), 'liq_spike': bool(liq_spike)}


def anomalous(f):
    if not f:
        return False
    hit = (abs(f['z']) >= Z_LOOSE and f['vr'] >= VR_LOOSE and abs(f['delta']) >= MIN_FLOW_USD
           and abs(f['imb']) >= MIN_IMB)
    return bool(hit or f['liq_spike'])


# ───────────────────────────── tahap 2: konteks ─────────────────────────────
def trend_state(c4, atr4):
    """Voting EMA (default 50/100/200) dengan band ATR. dir: +1 naik, -1 turun, 0 netral,
    None bila data kurang."""
    votes = []
    for n in TREND_EMAS:
        if len(c4) < n + 20:
            continue
        e = ema(c4, n)[-1]
        b = TREND_BAND * atr4
        votes.append(1 if c4[-1] > e + b else (-1 if c4[-1] < e - b else 0))
    if len(votes) < 2:
        return {'dir': None, 'score': 0.0, 'votes': votes}
    sc = sum(votes) / len(votes)
    return {'dir': 1 if sc >= 0.67 else (-1 if sc <= -0.67 else 0), 'score': sc, 'votes': votes}


def build_levels(a4, a1, atr4v, price):
    n4, n1 = len(a4['h']), len(a1['h'])
    items = []
    for sw, a, n, w, tf in ((SW4, a4, n4, 2.0, '4h'), (SW1, a1, n1, 1.0, '1h')):
        hi, lo = swings(a['h'], a['l'], sw)
        for k, p in hi + lo:
            items.append((p, w * (0.5 + 0.5 * k / n), tf))
    items.append((float(a4['h'][-RANGE_N4:].max()), 3.0, '4h'))
    items.append((float(a4['l'][-RANGE_N4:].min()), 3.0, '4h'))
    items.sort(key=lambda x: x[0])
    tol = max(LEVEL_TOL_ATR4 * atr4v, price * 0.0015)
    out, cur = [], None
    for p, w, tf in items:
        if cur and p - cur['pw'] / cur['w'] <= tol:
            cur['pw'] += p * w; cur['w'] += w; cur['n'] += 1; cur['tfs'].add(tf)
            cur['lo'] = min(cur['lo'], p); cur['hi'] = max(cur['hi'], p)
        else:
            if cur:
                out.append(cur)
            cur = {'pw': p * w, 'w': w, 'n': 1, 'tfs': {tf}, 'lo': p, 'hi': p}
    if cur:
        out.append(cur)
    return [{'price': x['pw'] / x['w'], 'strength': x['w'], 'touches': x['n'],
             'tfs': sorted(x['tfs']), 'lo': x['lo'], 'hi': x['hi']} for x in out]


def detect_sweeps(a1, levels, atr1, look=3):
    """Wick menembus TEPI TERLUAR klaster level (bukan rata-ratanya) lalu close kembali ke
    sisi semula, dalam `look` bar terakhir. Level berupa zona: wick kecil di dalam zona
    adalah noise, bukan penyapuan likuiditas."""
    h, l, c = a1['h'], a1['l'], a1['c']
    n, pen, found = len(c), 0.05 * atr1, {}
    for j in range(max(n - look, 1), n):
        for L in levels:
            if L['strength'] < MIN_STRENGTH:
                continue
            top, bot = L.get('hi', L['price']), L.get('lo', L['price'])
            # Syarat 'datang dari sisi semula': bar sebelum wick harus close di sisi yang sama.
            # Tanpa ini, harga yang sebelumnya di atas zona lalu jatuh MENEMBUS zona salah dibaca
            # sebagai sweep (padahal itu breakdown).
            if h[j] > top + pen and c[j] < top and c[-1] < top and c[j - 1] < top:
                side, ext = 'high', float(h[j])
            elif l[j] < bot - pen and c[j] > bot and c[-1] > bot and c[j - 1] > bot:
                side, ext = 'low', float(l[j])
            else:
                continue
            cand = {'side': side, 'level': L, 'bar': j, 'extreme': ext,
                    'age': n - 1 - j}
            old = found.get(side)
            if old is None or (L['strength'], j) > (old['level']['strength'], old['bar']):
                found[side] = cand
    return found


def detect_breakout(a1, levels, atr1):
    """Dua close 1H terakhir di luar level kuat, sementara close sebelumnya masih di dalam."""
    c = a1['c']
    if len(c) < 4:
        return None
    best = None
    for L in levels:
        if L['strength'] < MIN_STRENGTH:
            continue
        p = L['price']
        up = min(c[-2:]) > p and c[-3] <= p
        dn = max(c[-2:]) < p and c[-3] >= p
        if (up or dn) and abs(c[-1] - p) <= 2.0 * atr1:
            cand = {'dir': 1 if up else -1, 'level': L}
            if best is None or L['strength'] > best['level']['strength']:
                best = cand
    return best


def detect_retest(a1, levels, atr1, price):
    """Breakout zona kuat (2 close di luar + lanjut >= BRK_FOLLOW_ATR) lalu harga kembali ke zona
    dari sisi yang baru ditembus. 'confirmed' = bar 1H terakhir menolak (close kembali di luar zona
    searah breakout, candle searah). Batal bila ada close menembus balik sisi jauh zona."""
    o, h, l, c = a1['o'], a1['h'], a1['l'], a1['c']
    n = len(c)
    if n < 12 or atr1 <= 0:
        return None
    start, best = max(n - RETEST_LOOK_H, 2), None
    for L in levels:
        if L['strength'] < MIN_STRENGTH:
            continue
        top, bot = L.get('hi', L['price']), L.get('lo', L['price'])
        for d in (1, -1):
            b = None
            for j in range(n - 4, start - 1, -1):          # breakout terbaru dulu; sisakan >= 3 bar setelahnya
                if d == 1 and c[j] > top and c[j + 1] > top and c[j - 1] <= top:
                    b = j; break
                if d == -1 and c[j] < bot and c[j + 1] < bot and c[j - 1] >= bot:
                    b = j; break
            if b is None:
                continue
            if d == 1:
                if h[b:].max() < top + BRK_FOLLOW_ATR * atr1 or (c[b + 2:] < bot).any():
                    continue
                touched = l[-2:].min() <= top + RETEST_TOL * atr1
                near = price - top <= NEAR_ATR * atr1 and price >= bot
                conf = bool(c[-1] > top and c[-1] > o[-1])
                ext = float(min(bot, l[-3:].min()))
            else:
                if l[b:].min() > bot - BRK_FOLLOW_ATR * atr1 or (c[b + 2:] > top).any():
                    continue
                touched = h[-2:].max() >= bot - RETEST_TOL * atr1
                near = bot - price <= NEAR_ATR * atr1 and price <= top
                conf = bool(c[-1] < bot and c[-1] < o[-1])
                ext = float(max(top, h[-3:].max()))
            if not (touched and near):
                continue
            cand = {'dir': d, 'level': L, 'confirmed': conf, 'extreme': ext, 'bar': b, 'age': n - 1 - b}
            if best is None or (L['strength'], b) > (best['level']['strength'], best['bar']):
                best = cand
    return best


def locate(price, levels, atr1, rng):
    """Lokasi harga: dekat level kuat, atau tepi range 4H bila sideways."""
    best = None
    for L in levels:
        if L['strength'] < MIN_STRENGTH:
            continue
        dist = abs(price - L['price'])
        if dist <= NEAR_ATR * atr1 and (best is None or dist < best[0]):
            best = (dist, L)
    if best:
        L = best[1]
        return {'kind': 'support' if L['price'] <= price else 'resistance', 'level': L,
                'src': 'level'}
    if rng and rng['valid']:
        if rng['pos'] <= 0.33:
            return {'kind': 'support', 'level': None, 'src': 'range'}
        if rng['pos'] >= 0.67:
            return {'kind': 'resistance', 'level': None, 'src': 'range'}
    return {'kind': None, 'level': None, 'src': None}


def oi_zones(rows, atr1, price):
    """Profil perubahan OI per bucket harga (25 jam terakhir): tempat posisi menumpuk."""
    w = max(0.5 * atr1, price * 0.001)
    b = {}
    for i in range(1, len(rows)):
        a, z = rows[i - 1].get('oi'), rows[i].get('oi')
        if a is None or z is None:
            continue
        p = (rows[i]['high'] + rows[i]['low'] + rows[i]['close']) / 3.0
        k = int(p // w)
        b[k] = b.get(k, 0.0) + (z - a)
    pos = sum(v for v in b.values() if v > 0)
    if pos <= 0:
        return []
    return sorted(({'price': k * w + w / 2, 'oi_add': v} for k, v in b.items()
                   if v > 0 and v >= ZONE_SHARE * pos), key=lambda x: -x['oi_add'])


def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def make_candidates(f, trend, loc, sweeps, brk, rng_pos_hint=None, retest=None, flow_setups=True):
    """Skenario + skor. Komponen: tren, lokasi, struktur, flow, OI (masing-masing -1/0/1)."""
    z, sd = f['z'], sgn(f['delta'])
    pxc = f.get('px_chg', 0.0)
    oi = f['oi_chg']
    td = trend['dir']
    tr = lambda d: 0 if td in (None, 0) else (1 if td == d else -1)
    cands = []
    if not flow_setups:       # koin tanpa anomali flow: hanya setup retest yang dinilai
        sweeps, brk = {}, None

    for side, sw in sweeps.items():
        d = 1 if side == 'low' else -1
        # Flow anomali melawan arah reclaim HANYA ditolak bila harga ikut bergerak searah flow
        # (flow 'menang'). Jika harga justru bergerak melawan flow, itu penyerapan (sah).
        if sd == -d and abs(z) >= Z_LOOSE and sgn(pxc) == sd and abs(pxc) >= PX_MIN:
            continue
        flush = f['lliq'] if d == 1 else f['sliq']
        comps = {'tren': tr(d), 'lokasi': 1 if ('4h' in sw['level']['tfs'] or sw['level']['strength'] >= 3.0) else 0,
                 'struktur': 1,
                 'flow': 1 if (sd == d and abs(z) >= Z_CONF) else 0,
                 'oi': 1 if (flush >= LIQ_MIN_USD or (oi is not None and oi <= -OI_FLUSH)) else 0}
        why = ("wick menyapu likuiditas di bawah level lalu close kembali ke atas" if d == 1
               else "wick menyapu likuiditas di atas level lalu close kembali ke bawah")
        cands.append({'name': 'SWEEP + RECLAIM', 'dir': d, 'comps': comps, 'why': why,
                      'level': sw['level'], 'extreme': sw['extreme']})

    if flow_setups and f['res'] is not None and f['vr'] >= ABS_VR and f['res'] <= ABS_RES and loc['kind']:
        d = 0
        if loc['kind'] == 'support' and sd < 0:
            d, why = 1, "jual agresif besar tapi harga tidak turun: terserap di area support"
        elif loc['kind'] == 'resistance' and sd > 0:
            d, why = -1, "beli agresif besar tapi harga tidak naik: terserap di area resistance"
        if d:
            comps = {'tren': tr(d), 'lokasi': 1, 'struktur': 1,
                     'flow': 1 if abs(z) >= Z_CONF else 0,
                     'oi': 1 if (oi is not None and oi >= OI_UP) else 0}
            cands.append({'name': 'ABSORPSI', 'dir': d, 'comps': comps, 'why': why,
                          'level': loc['level'], 'extreme': None})

    if brk and sd == brk['dir'] and not (sgn(pxc) == -brk['dir'] and abs(pxc) >= PX_MIN):
        d = brk['dir']
        weak = oi is not None and oi <= -OI_UP
        comps = {'tren': tr(d), 'lokasi': 1 if ('4h' in brk['level']['tfs'] or brk['level']['strength'] >= 3.0) else 0,
                 'struktur': 1, 'flow': 1 if abs(z) >= Z_CONF else 0,
                 'oi': 1 if (oi is not None and oi >= OI_UP) else 0}
        why = ("tembus level dengan OI turun: dorongan lemah (posisi lama ditutup)" if weak
               else "tembus level dengan flow searah" + (" dan OI naik (posisi baru)" if comps['oi'] else ""))
        cands.append({'name': 'BREAK LEMAH' if weak else 'BREAKOUT', 'dir': d, 'comps': comps,
                      'why': why, 'level': brk['level'], 'extreme': None})

    if retest:
        d, L = retest['dir'], retest['level']
        comps = {'tren': tr(d), 'lokasi': 1, 'struktur': 1,
                 'flow': 1 if (sd == d and abs(z) >= Z_CONF) else 0,
                 'oi': 1 if (oi is not None and oi >= OI_UP) else 0}
        if retest['confirmed']:
            name = 'RETEST ✅ TOLAKAN'
            why = "breakout lalu kembali menguji zona, dan candle 1H terakhir menolak searah breakout"
        else:
            name = 'RETEST ⏳ MENUNGGU'
            why = "breakout lalu harga kembali ke zona; belum ada tolakan: tunggu candle 1H close searah breakout"
        cands.append({'name': name, 'dir': d, 'comps': comps, 'why': why,
                      'level': L, 'extreme': retest['extreme']})

    for c in cands:
        c['score'] = sum(c['comps'].values())
    return cands


def plan_targets(c, price, levels, atr1, rng, zones):
    d = c['dir']
    far = MIN_T_ATR * atr1
    ls = [L for L in levels if L['strength'] >= MIN_STRENGTH and
          ((d == 1 and L['price'] > price + far) or (d == -1 and L['price'] < price - far))]
    ls.sort(key=lambda L: (L['price'] - price) * d)
    near_zone = lambda p: any(abs(z['price'] - p) <= 0.5 * atr1 for z in zones)
    tg = []
    for L in ls:
        if not tg or abs(L['price'] - tg[-1]['price']) >= 0.8 * atr1:
            tg.append({'price': L['price'], 'strength': L['strength'], 'tfs': L['tfs'],
                       'oi_zone': near_zone(L['price'])})
        if len(tg) == 2:
            break
    if c.get('extreme') is not None:
        inv = c['extreme'] - d * 0.2 * atr1
    elif c['level'] is not None:
        inv = c['level']['price'] - d * (0.5 if c['name'] == 'ABSORPSI' else 0.4) * atr1
    elif rng and rng['valid']:
        inv = (rng['lo'] if d == 1 else rng['hi']) - d * 0.3 * atr1
    else:
        inv = price - d * 1.5 * atr1
    rr = None
    if tg and abs(price - inv) > 0:
        rr = abs(tg[0]['price'] - price) / abs(price - inv)
    return {'targets': tg, 'inval': float(inv), 'rr': rr}


def passes_filters(best, plan, price, atr1, crowd=None):
    """Filter kelayakan setelah skor: struktur wajib, R:R ke T1, dan invalidasi tidak terlalu jauh."""
    if best['score'] < MIN_SCORE or best['comps']['struktur'] != 1:
        return False, 'skor'
    if crowd and best['score'] < MIN_SCORE + 1:   # sisi sudah ramai: butuh skor lebih tinggi
        return False, 'sisi ramai, skor kurang'
    if plan['rr'] is not None and plan['rr'] < MIN_RR:
        return False, f"R:R {plan['rr']:.1f} < {MIN_RR:g}"
    if atr1 > 0 and abs(price - plan['inval']) > MAX_INVAL_ATR * atr1:
        return False, f"invalidasi {abs(price - plan['inval']) / atr1:.1f} ATR terlalu jauh"
    return True, ''


def analyze_symbol(it, rows, a4, a1, flow_setups=True):
    """Return dict kandidat terbaik (atau None) + diagnosa."""
    price = rows[-1]['close']
    f = flow_eval(rows, len(rows) - 1)
    if not f:
        return None
    atr1 = float(atr(a1['h'], a1['l'], a1['c'])[-1])
    atr4 = float(atr(a4['h'], a4['l'], a4['c'])[-1])
    if atr1 <= 0 or atr4 <= 0:
        return None
    trend = trend_state(a4['c'], atr4)
    er4 = eff_ratio(a4['c'], 20)
    hi, lo = float(a4['h'][-RANGE_N4:].max()), float(a4['l'][-RANGE_N4:].min())
    rng_seg_hi, rng_seg_lo = float(a4['h'][-60:].max()), float(a4['l'][-60:].min())
    sideways = (er4 is not None and er4 < ER_RANGE) or trend['dir'] == 0
    rng = {'hi': rng_seg_hi, 'lo': rng_seg_lo,
           'pos': (price - rng_seg_lo) / (rng_seg_hi - rng_seg_lo) if rng_seg_hi > rng_seg_lo else 0.5,
           'valid': bool(sideways and (rng_seg_hi - rng_seg_lo) >= 3 * atr4)}
    levels = build_levels(a4, a1, atr4, price)
    loc = locate(price, levels, atr1, rng)
    sweeps = detect_sweeps(a1, levels, atr1)
    brk = detect_breakout(a1, levels, atr1)
    zones = oi_zones(rows, atr1, price)
    retest = detect_retest(a1, levels, atr1, price) if RETEST_ON else None
    cands = make_candidates(f, trend, loc, sweeps, brk, retest=retest, flow_setups=flow_setups)
    if not cands:
        return {'sym': it['sym'], 'flow': f, 'trend': trend, 'best': None, 'price': price}
    best = max(cands, key=lambda c: (c['score'], c['comps']['tren'], abs(f['z'])))
    plan = plan_targets(best, price, levels, atr1, rng, zones)
    fr, lsr = it.get('fr'), rows[-1].get('lsr_acc')
    d = best['dir']
    crowd = None
    if d == 1 and ((fr is not None and fr >= FUND_EXT) or (lsr is not None and lsr >= LSR_HI)):
        crowd = 'long sudah ramai (funding/rasio akun): rawan squeeze'
    if d == -1 and ((fr is not None and fr <= -FUND_EXT) or (lsr is not None and lsr <= LSR_LO)):
        crowd = 'short sudah ramai (funding/rasio akun): rawan squeeze'
    out = {'sym': it['sym'], 'price': price, 'flow': f, 'trend': trend, 'er4': er4, 'loc': loc,
            'rng': rng, 'best': best, 'plan': plan, 'zones': zones[:2], 'crowd': crowd, 'fr': fr,
            'atr1': atr1}
    ok, why = passes_filters(best, plan, price, atr1, crowd)
    out['alert'], out['skip'] = ok, why
    return out


# ───────────────────────────── state & pesan ─────────────────────────────
def load_state():
    try:
        with open(STATE_FILE) as fh:
            st = json.load(fh)
        cut = time.time() - DEDUPE_H * 3600
        return {k: v for k, v in st.items() if isinstance(v, (int, float)) and v >= cut}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(st):
    try:
        with open(STATE_FILE, 'w') as fh:
            json.dump(st, fh)
    except OSError as e:
        logging.warning(f"state tidak tersimpan: {e}")


def px(p):
    return f"{p:.6g}"


def fmt_usd(x):
    a = abs(x)
    if a >= 1e9: return f"{x/1e9:+.2f}B"
    if a >= 1e6: return f"{x/1e6:+.2f}M"
    return f"{x/1e3:+.0f}K"


def fmt_alert(r):
    b, f, tr, pl = r['best'], r['flow'], r['trend'], r['plan']
    d, price = b['dir'], r['price']
    arrow = "🟢 ▲ BIAS NAIK" if d == 1 else "🔴 ▼ BIAS TURUN"
    td = tr['dir']
    ttxt = {1: "NAIK", -1: "TURUN", 0: "NETRAL/SIDEWAYS", None: "data kurang"}[td]
    if td is not None and tr['votes']:
        agree = sum(1 for v in tr['votes'] if v == td) if td else 0
        ttxt += f" ({agree}/{len(tr['votes'])} EMA)" if td else ""
    counter = b['comps']['tren'] == -1
    head = f"{arrow} · <b>{r['sym']}</b> · skor {b['score']}/5" + (" · ⚠ COUNTER-TREND" if counter else "")
    lines = [head, f"<b>{b['name']}</b>: {b['why']}"]
    L = b.get('level')
    zone = (f"{px(L['lo'])}–{px(L['hi'])}" if L and L['hi'] > L['lo'] * 1.001 else (px(L['price']) if L else ''))
    loc_txt = f"zona {zone} ({'/'.join(L['tfs'])}, {L['touches']} sentuhan)" if L else \
        (f"tepi range 4H ({r['loc']['kind']})" if r['loc']['kind'] else "tidak dekat level")
    lines.append(f"Tren 4H: {ttxt} · Lokasi: {loc_txt}")
    is_rt = b['name'].startswith('RETEST')
    if is_rt:
        lines.append(f"Entry ideal: area zona {zone} · posisi range 4H {r['rng']['pos']:.0%} (0% dasar, 100% puncak)")
        if not b['name'].endswith('TOLAKAN'):
            lines.append("⏳ Belum entry-valid: tunggu close 1H searah breakout di luar zona.")
    oi_txt = f"{f['oi_chg']:+.2f}%" if f['oi_chg'] is not None else "n/a"
    liq = f" · likuidasi L {fmt_usd(f['lliq'])[1:]} / S {fmt_usd(f['sliq'])[1:]}" if f['liq_spike'] else ""
    lines.append(f"Flow 1j: {fmt_usd(f['delta'])} (z {f['z']:+.1f}, vol {f['vr']:.1f}x) · "
                 f"harga {f['px_chg']:+.2f}% · OI {oi_txt}{liq}")
    if pl['targets']:
        ts = []
        for i, t in enumerate(pl['targets'], 1):
            pct = (t['price'] - price) / price * 100
            ts.append(f"T{i} {px(t['price'])} ({pct:+.1f}%{', zona OI' if t['oi_zone'] else ''})")
        lines.append("Target: " + " · ".join(ts))
    else:
        lines.append("Target: tidak ada level kuat di arah ini (ruang terbuka)")
    if L and not is_rt:   # seberapa jauh harga sudah pergi dari zona: info 'terlambat atau belum'
        edge = L.get('hi', L['price']) if d == -1 else L.get('lo', L['price'])
        moved = (edge - price) / price * 100 * (1 if d == -1 else -1)
        atr1 = r.get('atr1') or 0
        atr_txt = f" ({abs(edge - price) / atr1:.1f} ATR 1j)" if atr1 > 0 else ""
        if moved > 0:
            lines.append(f"Jarak dari zona: harga sudah bergerak {moved:.1f}% searah bias{atr_txt}")
        else:
            lines.append(f"Jarak dari zona: harga masih di zona / belum bergerak searah bias")
    ipct = (pl['inval'] - price) / price * 100
    rr = f" · R:R T1 {pl['rr']:.1f}" if pl['rr'] else ""
    lines.append(f"Invalidasi: {px(pl['inval'])} ({ipct:+.1f}%){rr}")
    if r['zones']:
        lines.append("OI menumpuk di: " + ", ".join(f"{px(z['price'])} ({fmt_usd(z['oi_add'])})" for z in r['zones']))
    if r['crowd']:
        lines.append(f"⚠ {r['crowd']}")
    return "\n".join(lines)


def build_message(alerts, n_uni, n_stage2):
    head = "🧭 <b>SMART FLOW</b> (anomali flow + konteks 4H/1H)"
    out = [head, f"Universe {n_uni} · dianalisis {n_stage2} · alert {len(alerts)}", ""]
    for r in alerts[:TOPN]:
        out.append(fmt_alert(r))
        out.append("")
    out.append("<i>Heuristik, belum divalidasi backtest. Bukan sinyal entry; cek struktur di chart.</i>")
    return "\n".join(out)


async def send_tg(s, text):
    if not TG_TOKEN or not TG_CHAT:
        logging.warning("Token/chat Telegram kosong - pesan hanya dicetak.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    ch, cur = [], ""
    for ln in text.split("\n"):
        if len(cur) + len(ln) + 1 > 3800:
            ch.append(cur); cur = ln
        else:
            cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        ch.append(cur)
    for c in ch:
        try:
            async with s.post(url, json={"chat_id": TG_CHAT, "text": c, "parse_mode": "HTML",
                                         "disable_web_page_preview": True},
                              timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status != 200:
                    logging.error(f"Telegram HTTP {r.status}: {await r.text()}")
        except Exception as e:
            logging.error(f"Telegram error: {e}")
        await asyncio.sleep(0.4)


# ───────────────────────────── orkestrasi ─────────────────────────────
async def stage1(s, it, sem):
    async with sem:
        rows = await fetch_15m(s, it['c'])
    if not rows:
        return None
    f = flow_eval(rows, len(rows) - 1)
    if not f:
        return None
    if anomalous(f):
        prio = max(abs(f['z']), 3.0 if f['liq_spike'] else 0.0)
        return {'it': it, 'rows': rows, 'flow': f, 'prio': prio, 'anom': True}
    cl = rows[-1]['close']
    rpct = (max(r['high'] for r in rows) - min(r['low'] for r in rows)) / cl * 100.0 if cl > 0 else 0.0
    if RETEST_ON and rpct >= RETEST_GATE_PCT:      # tanpa anomali: hanya diperiksa untuk setup retest
        return {'it': it, 'rows': rows, 'flow': f, 'prio': rpct, 'anom': False}
    return None


async def stage2(s, cand, sem):
    async with sem:
        a4 = await fetch_candles(s, cand['it']['c'], '4h', N4)
        a1 = await fetch_candles(s, cand['it']['c'], '1h', N1)
    if a4 is None or a1 is None:
        return None
    return analyze_symbol(cand['it'], cand['rows'], a4, a1, flow_setups=cand['anom'])


async def diagnostic(s):
    for c in ("BTC_USDT", "ETH_USDT"):
        st = await _get(s, f"{GATE}/contract_stats", {"contract": c, "interval": "15m", "limit": 3}, "diag")
        print(f"=== {c} contract_stats ===")
        print(json.dumps(st, indent=1)[:2500])


async def main():
    async with aiohttp.ClientSession() as s:
        if DIAG:
            await diagnostic(s)
        uni = await universe(s, load_bybit_symbols_local())
        if not uni:
            logging.error("Universe kosong - berhenti.")
            return
        sem = asyncio.Semaphore(SEM_N)
        c1 = [x for x in await asyncio.gather(*[stage1(s, it, sem) for it in uni]) if x]
        an = sorted([x for x in c1 if x['anom']], key=lambda x: x['prio'], reverse=True)[:STAGE2_MAX]
        mv = sorted([x for x in c1 if not x['anom']], key=lambda x: x['prio'], reverse=True)[:RETEST_MAX]
        logging.info(f"Tahap 1: {len(an)} anomali + {len(mv)} kandidat retest dari {len(uni)}")
        res = [x for x in await asyncio.gather(*[stage2(s, c, sem) for c in an + mv]) if x]
        for r in res:
            b = r.get('best')
            logging.info(f"  {r['sym']:<14} z={r['flow']['z']:+.1f} tren={r['trend']['dir']} "
                         f"best={(b['name'] + ' ' + str(b['score'])) if b else '-'}"
                         f"{' SKIP: ' + r['skip'] if r.get('skip') else ''}")
        alerts = [r for r in res if r.get('alert')]
        state = load_state()
        fresh = []
        for r in alerts:
            key = f"{r['sym']}|{r['best']['name']}|{r['best']['dir']}"
            if key in state:
                continue
            state[key] = time.time()
            fresh.append(r)
        fresh.sort(key=lambda r: (r['best']['score'], abs(r['flow']['z'])), reverse=True)
        save_state(state)
        if fresh:
            await send_tg(s, build_message(fresh, len(uni), len(res)))
        else:
            logging.info("Tidak ada alert baru siklus ini - tidak kirim pesan.")


# ───────────────────────────── selftest ─────────────────────────────
def _synth(case):
    """Data sintetis deterministik: tren 4H naik, 1H berosilasi 100-110 (level kuat di 99.7 & 110.3)."""
    rng = np.random.default_rng(3)
    n4, n1, n15 = N4, N1, BARS
    i4 = np.arange(n4)
    c4 = 60 + 40 * (i4 / (n4 - 1)) + 0.8 * np.sin(i4 / 3)
    a4 = {'o': c4 - 0.1, 'h': c4 + 0.4, 'l': c4 - 0.4, 'c': c4, 't': 1_700_000_000 + i4 * 14400}
    k = np.arange(n1) - (n1 - 1)
    c1 = 105 - 5 * np.cos(2 * np.pi * k / 24)
    a1 = {'o': c1.copy(), 'h': c1 + 0.3, 'l': c1 - 0.3, 'c': c1.copy(), 't': 1_700_000_000 + np.arange(n1) * 3600}
    if case == 'sweep':
        a1['o'][-1], a1['l'][-1], a1['c'][-1], a1['h'][-1] = 100.4, 98.9, 100.6, 100.9
        last_px = 100.6
    else:   # absorpsi: bar terakhir normal, harga datar di dekat support
        a1['o'][-1], a1['l'][-1], a1['c'][-1], a1['h'][-1] = 100.0, 99.95, 100.05, 100.3
        last_px = 100.05
    rows = []
    oi = 1.0e7
    for i in range(n15):
        tail = i >= n15 - WIN
        lsr = max(0.2, 1.0 + rng.normal(0, 0.15))
        vol = 1e5 * max(0.3, 1 + rng.normal(0, 0.3))
        lq = 0.0
        if tail:
            vol = 4e5
            if case == 'sweep':
                lsr, lq = 2.5, 60_000
            else:
                lsr = 0.4
        if case == 'sweep':
            oi += (-0.005e7 if tail else rng.normal(0, 2_000))
            cl = 100.0 + (last_px - 100.0) * (i / (n15 - 1))
        else:
            oi += (+0.0035e7 if tail else rng.normal(0, 2_000))
            cl = last_px + (0.0 if tail else rng.normal(0, 0.12))   # ekor datar = gerak harga kecil
        share = lsr / (1 + lsr)
        rows.append({'t': 1_700_000_000 + i * 900, 'open': cl, 'high': cl + 0.05, 'low': cl - 0.05,
                     'close': cl, 'vol': vol, 'delta': vol * (2 * share - 1), 'oi': oi,
                     'lliq': lq, 'sliq': 0.0, 'lsr_acc': 1.1})
    return rows, a4, a1


def selftest():
    ok = True

    def check(c, msg):
        nonlocal ok
        print(("PASS " if c else "FAIL ") + msg)
        ok &= bool(c)

    # swings tanpa lookahead: bar tertinggi di ujung kanan BUKAN pivot
    h = np.array([1, 2, 3, 2, 1, 2, 5.0]); l = h - 0.5
    hi, _ = swings(h, l, 2)
    check(all(k < len(h) - 2 for k, _ in hi), "swing hanya terkonfirmasi (butuh sw bar di kanan)")
    # tren: naik vs datar
    c_up = np.linspace(50, 100, 300); c_flat = 100 + np.sin(np.arange(300) / 5)
    check(trend_state(c_up, 1.0)['dir'] == 1, "tren naik terdeteksi")
    check(trend_state(c_flat, 3.0)['dir'] == 0, "pasar datar -> netral")
    check(trend_state(c_up[:30], 1.0)['dir'] is None, "data 4H kurang -> None (bukan tebakan)")

    rows, a4, a1 = _synth('sweep')
    r = analyze_symbol({'sym': 'TESTUSDT', 'fr': 0.0001}, rows, a4, a1)
    b = r['best']
    check(r['flow']['z'] >= Z_CONF and anomalous(r['flow']), f"tahap 1 anomali (z={r['flow']['z']:.1f})")
    check(b is not None and b['name'] == 'SWEEP + RECLAIM' and b['dir'] == 1, f"pola SWEEP+RECLAIM naik: {b and b['name']}")
    check(b['score'] == 5 and r['alert'], f"skor 5/5 & alert: {b['score']} {b['comps']}")
    t1 = r['plan']['targets'][0]['price'] if r['plan']['targets'] else None
    check(t1 is not None and 109 <= t1 <= 111, f"T1 di level resisten ~110.3: {t1}")
    check(r['plan']['inval'] < 98.9, f"invalidasi di bawah ekstrem sweep: {r['plan']['inval']:.2f}")
    msg = fmt_alert(r)
    check('SWEEP' in msg and 'Invalidasi' in msg and 'Target' in msg, "pesan memuat pola, target, invalidasi")

    rows, a4, a1 = _synth('absorb')
    r = analyze_symbol({'sym': 'ABSUSDT', 'fr': 0.0}, rows, a4, a1)
    b = r['best']
    check(b is not None and b['name'] == 'ABSORPSI' and b['dir'] == 1,
          f"ABSORPSI bullish (jual agresif terserap di support): {b and (b['name'], b['dir'], b['score'])}")

    # arah absorpsi tidak boleh terbalik: beli agresif di support BUKAN absorpsi bullish
    rows2, a42, a12 = _synth('absorb')
    for rr in rows2[-WIN:]:
        rr['delta'] = abs(rr['delta'])
    r2 = analyze_symbol({'sym': 'X', 'fr': 0.0}, rows2, a42, a12)
    check(not (r2['best'] and r2['best']['name'] == 'ABSORPSI'), "beli agresif di support != absorpsi")

    # sweep harus menembus tepi terluar zona: wick di dalam zona = noise
    Lz = {'price': 100.0, 'lo': 99.5, 'hi': 100.5, 'strength': 3.0, 'touches': 6, 'tfs': ['1h']}
    base = np.full(10, 101.0)
    mk = lambda low, close: {'h': base + 0.3, 'l': np.r_[base[:-1] - 0.3, low],
                             'c': np.r_[base[:-1], close], 'o': base.copy()}
    check(not detect_sweeps(mk(99.8, 100.9), [Lz], 1.0), "wick di dalam zona (99.8 > tepi 99.5) bukan sweep")
    # harga sebelumnya DI ATAS zona lalu jatuh menembus = breakdown, bukan sweep resisten
    ab = np.array([103, 103, 103, 103, 103, 103, 103, 103, 103, 98.0])
    fall = {'h': ab + 0.3, 'l': ab - 0.3, 'c': ab.copy(), 'o': ab.copy()}
    fall['h'][-1] = 103.3
    check('high' not in detect_sweeps(fall, [Lz], 1.0), "jatuh menembus zona dari atas bukan sweep-high")
    check('low' in detect_sweeps(mk(99.2, 100.9), [Lz], 1.0), "wick menembus tepi zona (99.2 < 99.5) = sweep")

    # flow vs respons harga pada sweep: flow 'menang' => tolak; flow terserap => sah
    Lv = {'price': 100.0, 'lo': 99.5, 'hi': 100.5, 'strength': 3.0, 'touches': 6, 'tfs': ['1h']}
    mkf = lambda delta, pxc: {'z': -2.6 if delta < 0 else 2.6, 'delta': delta, 'oi_chg': 0.0, 'lliq': 0.0,
                              'sliq': 0.0, 'res': None, 'vr': 1.0, 'px_chg': pxc}
    tr0, lc0 = {'dir': 0}, {'kind': None}
    up = {'low': {'level': Lv, 'extreme': 98.5}}
    dn = {'high': {'level': Lv, 'extreme': 101.5}}
    names = lambda cs: [c['name'] for c in cs]
    check('SWEEP + RECLAIM' not in names(make_candidates(mkf(-2e5, -1.7), tr0, lc0, up, None)),
          "long-sweep ditolak: jual anomali DAN harga turun (flow menang)")
    check('SWEEP + RECLAIM' in names(make_candidates(mkf(-2e5, +0.5), tr0, lc0, up, None)),
          "long-sweep sah: jual anomali tapi harga naik (terserap)")
    check('SWEEP + RECLAIM' in names(make_candidates(mkf(+2e5, -0.8), tr0, lc0, dn, None)),
          "short-sweep sah: beli anomali terserap, harga turun")
    check('SWEEP + RECLAIM' not in names(make_candidates(mkf(+2e5, +1.2), tr0, lc0, dn, None)),
          "short-sweep ditolak: beli anomali DAN harga naik")
    brk_dn = {'dir': -1, 'level': Lv}
    check('BREAKOUT' not in names(make_candidates(mkf(-3e5, +0.97), tr0, lc0, {}, brk_dn)),
          "breakout turun ditolak saat harga 1j justru naik")
    check('BREAKOUT' in names(make_candidates(mkf(-3e5, -0.97), tr0, lc0, {}, brk_dn)),
          "breakout turun sah saat harga turun")

    # filter R:R dan invalidasi
    bst = {'score': 4, 'comps': {'struktur': 1}}
    check(not passes_filters(bst, {'rr': 0.4, 'inval': 99.0}, 100.0, 1.0)[0], "R:R 0.4 ditolak")
    check(not passes_filters(bst, {'rr': 2.0, 'inval': 80.0}, 100.0, 1.0)[0], "invalidasi 20 ATR ditolak")
    check(passes_filters(bst, {'rr': 1.5, 'inval': 99.0}, 100.0, 1.0)[0], "R:R 1.5 & invalidasi dekat lolos")
    check(not passes_filters({'score': 3, 'comps': {'struktur': 1}}, {'rr': 1.5, 'inval': 99.0}, 100.0, 1.0, 'ramai')[0],
          "skor 3 + sisi ramai ditolak")
    check(not passes_filters(bst, {'rr': 1.0, 'inval': 99.0}, 100.0, 1.0)[0], "R:R 1.0 ditolak (min 1.2)")
    check(passes_filters(bst, {'rr': None, 'inval': 99.0}, 100.0, 1.0)[0], "tanpa target (ruang terbuka) tidak ditolak karena R:R")

    # ── retest setelah breakout ──
    LZ = {'price': 100.0, 'lo': 99.5, 'hi': 100.5, 'strength': 3.0, 'touches': 6, 'tfs': ['1h', '4h']}
    def bars(closes, last_o=None):
        cl = np.array(closes, float); op = np.r_[cl[0], cl[:-1]]
        if last_o is not None:
            op[-1] = last_o
        return {'o': op, 'h': np.maximum(op, cl) + 0.2, 'l': np.minimum(op, cl) - 0.2, 'c': cl}
    up = [97.0] * 20 + [101.5, 102.0, 103.0, 104.0, 103.0, 102.0, 101.5, 101.0, 100.9]
    r_ok = detect_retest(bars(up + [101.2], last_o=100.9), [LZ], 1.0, 101.2)
    check(r_ok and r_ok['dir'] == 1 and r_ok['confirmed'], "retest naik + tolakan: terkonfirmasi")
    r_wait = detect_retest(bars(up + [100.6], last_o=101.0), [LZ], 1.0, 100.6)
    check(r_wait and r_wait['dir'] == 1 and not r_wait['confirmed'], "retest naik tanpa tolakan: menunggu")
    check(detect_retest(bars(up + [99.0], last_o=100.9), [LZ], 1.0, 99.0) is None,
          "close menembus balik zona = breakout gagal, bukan retest")
    weak = [97.0] * 20 + [100.8] * 10
    check(detect_retest(bars(weak), [LZ], 1.0, 100.8) is None, "breakout tanpa follow-through bukan breakout")
    far = up + [103.5]
    check(detect_retest(bars(far, last_o=102.8), [LZ], 1.0, 103.5) is None, "harga sudah jauh dari zona: bukan retest")
    dn = [103.0] * 20 + [98.5, 98.0, 97.0, 96.0, 97.0, 98.0, 98.5, 99.0, 99.1]
    r_s = detect_retest(bars(dn + [98.8], last_o=99.1), [LZ], 1.0, 98.8)
    check(r_s and r_s['dir'] == -1 and r_s['confirmed'], "retest turun + tolakan: terkonfirmasi")
    # integrasi skor, setup alert, invalidasi
    mkf2 = {'z': 0.3, 'delta': 1e3, 'oi_chg': 0.0, 'lliq': 0.0, 'sliq': 0.0, 'res': None, 'vr': 1.0, 'px_chg': 0.1}
    cs = make_candidates(mkf2, {'dir': 1}, {'kind': None}, {}, None, retest=r_ok, flow_setups=False)
    check(len(cs) == 1 and cs[0]['name'].startswith('RETEST') and cs[0]['score'] >= 3,
          f"koin tanpa anomali: hanya setup retest, skor {cs[0]['score'] if cs else None}")
    cs2 = make_candidates(mkf2, {'dir': 0}, {'kind': None}, {}, None, retest=r_ok, flow_setups=False)
    check(cs2[0]['score'] < 3, "retest melawan/tanpa tren: skor kurang dari 3 (tidak alert)")
    pl = plan_targets(cs[0], 101.2, [LZ], 1.0, None, [])
    check(pl['inval'] < 99.5, f"invalidasi retest di bawah dasar zona: {pl['inval']:.2f}")

    # dedupe state
    global STATE_FILE
    old, STATE_FILE = STATE_FILE, '/tmp/_smart_state_test.json'
    try:
        save_state({'A|X|1': time.time(), 'B|X|1': time.time() - 99 * 3600})
        st = load_state()
        check('A|X|1' in st and 'B|X|1' not in st, "state: entri baru dipertahankan, kedaluwarsa dibuang")
    finally:
        STATE_FILE = old
    print("\nSELFTEST", "LULUS" if ok else "GAGAL")
    return 0 if ok else 1


if __name__ == "__main__":
    if '--selftest' in sys.argv:
        sys.exit(selftest())
    asyncio.run(main())
