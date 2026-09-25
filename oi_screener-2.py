"""OI Quadrant Screener — 4 kuadran OI x Harga, syarat SEMUA timeframe searah.

ATURAN (sesuai pola screening manual):
  LONG BUILDUP    : harga NAIK  + OI NAIK  di 15m, 1h, 4h
  SHORT BUILDUP   : harga TURUN + OI NAIK  di 15m, 1h, 4h
  SHORT COVERING  : harga NAIK  + OI TURUN di 15m, 1h, 4h
  LONG UNWINDING  : harga TURUN + OI TURUN di 15m, 1h, 4h
  SQUEEZE WATCH   : buildup tapi funding/LSR ekstrem searah posisi -> rapuh

Kriteria BINER dan bisa diuji: semua TF harus sepakat, tidak ada skor tumpukan
bobot yang dikarang. Konsekuensinya kandidat SEDIKIT (sering 0-5 per siklus).
Itu memang sifat kriteria ketat, bukan kegagalan.

YANG SENGAJA TIDAK DIPAKAI:
  - Filter/tampilan RVOL (volume) - dibuang atas permintaan
  - Tier likuiditas - proxy Gate.io tidak mengukur likuiditas pasar riil
  - EMA tren - redundan, 3 TF searah SUDAH struktur tren
  - Skor berbobot - tidak ada dasarnya, diganti urutan by kekuatan OI

Data: Gate.io (Bybit tickers/instruments-info kena geo-block dari GitHub
Actions). Difilter ke simbol yang tradable di Bybit lewat bybit_symbols.json.

=== REGIME ALIGNMENT (ditambahkan) ===
Lapisan tambahan SEBELUM daftar kuadran: skor searah/lawan kondisi market
secara keseluruhan, dari breadth Gate.io x Hyperliquid (dua sumber independen,
bukan direkonstruksi dari hasil trade sendiri seperti regime_proxy_score lama).

Threshold +-0.30 diambil langsung dari riset 533 trade riwayat manual trading:
  - Alignment kuat (>0.3) + ditahan sebagai swing (>24 jam): WR 74.5%, mean R +0.975 (n=47)
  - Alignment lemah/lawan + dipotong di zona intraday (1-24 jam): WR 10.2%, mean R -0.653 (n=118)
Kalau Gate & Hyperliquid BERTENTANGAN arah, skor ditarik ke 0 (bukan dirata-
ratakan naif) -- itu sinyal "tidak jelas", bukan "netral". Sama filosofinya
dengan syarat "3 TF harus sepakat" di kuadran OI di atas.
Reversal antar-cycle dideteksi dari regime_state.json yang di-commit balik ke
repo tiap run (lihat CATATAN WORKFLOW di bagian bawah file).
Kirim di pesan Telegram YANG SAMA -- tidak ada bot/token baru.
"""
import os, asyncio, logging, json, aiohttp, numpy as np
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
TG_TOKEN = (os.getenv('OI_TELEGRAM_TOKEN') or '').strip()
TG_CHAT  = (os.getenv('OI_TELEGRAM_CHAT_ID') or '').strip()
DIAG     = os.getenv('OI_DIAGNOSTIC', '').strip().lower() == 'true'

# Ambang minimum agar sebuah pergerakan dianggap "ada", bukan noise nol.
# Sengaja KECIL: tugasnya menyaring pergerakan nyaris-nol, bukan menyeleksi.
# Ambang per timeframe: 15m wajar lebih kecil dari 4h.
# Ambang seragam 0.10 membuat 68 dari 84 pair gugur (15 Sep 2026) -
# mayoritas karena candle 15m nyaris datar, bukan karena sinyal lemah.
MIN_MOVE = {'15m': 0.03, '1h': 0.08, '4h': 0.15}

# Funding: batas "crowded" (per 8 jam). Hanya untuk penanda squeeze + catatan.
FUND_EXT, FUND_VEXT = 0.0005, 0.0010
LSR_HI, LSR_LO = 2.0, 0.5

MIN_TURNOVER = 1_000_000     # buang pair yang praktis tidak bisa ditradingkan
MAX_SYM, TOPN = 400, 10
TFS = [('15m', '15m', 1), ('1h', '1h', 4), ('4h', '4h', 16)]  # (label, kline, lookback OI)
SEM_N, RETRIES, DELAY, TMO = 12, 3, 3, 15

GATE  = "https://api.gateio.ws/api/v4/futures/usdt"
BYBIT = "https://api.bybit.com/v5/market"
BYBIT_SYMBOLS_FILE = "bybit_symbols.json"

# Regime alignment: threshold & file state (lihat docstring atas untuk dasar riset)
ALIGN_STRONG = 0.30
ALIGN_WEAK = -0.30
REGIME_STATE_FILE = "regime_state.json"

# Token saham/komoditas di Gate.io - bukan crypto, dibuang.
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
            logging.info(f"Bybit: {len(syms)} simbol dari file lokal "
                         f"(update: {data.get('updated_at', '?')})")
            return syms
    except FileNotFoundError:
        logging.warning(f"{BYBIT_SYMBOLS_FILE} tidak ditemukan.")
    except (json.JSONDecodeError, KeyError) as e:
        logging.warning(f"{BYBIT_SYMBOLS_FILE} rusak: {e}")
    return None


async def bybit_syms(s):
    local = load_bybit_symbols_local()
    if local:
        return local
    d = await _get(s, f"{BYBIT}/instruments-info",
                   {"category": "linear", "limit": 1000}, "Bybit instruments-info")
    if not d or d.get('retCode') != 0:
        logging.warning("Simbol Bybit tak terambil - filter Bybit NONAKTIF.")
        return None
    return {i['symbol'] for i in d.get('result', {}).get('list', [])
            if i.get('symbol', '').endswith('USDT') and i.get('status') == 'Trading'}


async def on_bybit(s, sym):
    """Cek tradable di Bybit lewat kline - satu-satunya endpoint Bybit yang
    lolos geo-block dari GitHub Actions."""
    d = await _get(s, f"{BYBIT}/kline",
                   {"category": "linear", "symbol": sym, "interval": "60", "limit": 1},
                   f"Bybit kline {sym}")
    if not d or d.get('retCode') != 0:
        return False
    return bool(d.get('result', {}).get('list'))


async def universe(s, bsyms):
    d = await _get(s, f"{GATE}/tickers", lbl="Gate tickers")
    if not d:
        return []
    out, skip, skip_nc = [], 0, 0
    for i in d:
        c = i.get('contract', '')
        if not c.endswith('_USDT'):
            continue
        base = c.replace('_USDT', '')
        if base in NON_CRYPTO:
            skip_nc += 1; continue
        bs = c.replace('_', '')
        if bsyms is not None and bs not in bsyms:
            skip += 1; continue
        try:
            tv = float(i.get('volume_24h_quote') or 0)
            lp = float(i.get('last') or 0)
            fr = float(i.get('funding_rate') or 0)
        except (TypeError, ValueError):
            continue
        if tv < MIN_TURNOVER or lp <= 0:
            continue
        out.append({'c': c, 'sym': bs, 'tv': tv, 'fr': fr})
    out.sort(key=lambda x: x['tv'], reverse=True)
    out = out[:MAX_SYM]
    logging.info(f"Universe: {len(out)} pair (tak ada di Bybit: {skip}, "
                 f"non-crypto: {skip_nc})")
    return out


def _f(d, *ks):
    for k in ks:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


async def oi_series(s, c):
    """Satu seri 15m -> perubahan OI dengan lookback 1/4/16 bar (=15m/1h/4h).
    Dihitung sendiri karena agregasi per-interval Gate menghasilkan nilai
    IDENTIK antara 1h dan 4h (terbukti 14 Sep 2026)."""
    d = await _get(s, f"{GATE}/contract_stats",
                   {"contract": c, "interval": "15m", "limit": 20},
                   "Gate contract_stats")
    if not d or not isinstance(d, list) or len(d) < 17:
        return {}, None
    try:
        d = sorted(d, key=lambda x: x.get('time', 0))
    except Exception:
        pass
    oi = [_f(x, 'open_interest_usd', 'open_interest') for x in d]
    if any(v is None for v in oi[-17:]):
        return {}, None

    def chg(lb):
        base = oi[-1 - lb]
        return ((oi[-1] - base) / base * 100.0) if base and base > 0 else None

    out = {lb: chg(n) for lb, _, n in TFS}
    return out, _f(d[-1], 'lsr_account')


async def px_change(s, c, iv):
    """% perubahan harga candle TERTUTUP terakhir (candle berjalan dibuang)."""
    d = await _get(s, f"{GATE}/candlesticks",
                   {"contract": c, "interval": iv, "limit": 6}, "Gate candles")
    if not d or not isinstance(d, list) or len(d) < 3:
        return None
    try:
        d = sorted(d, key=lambda x: int(x.get('t', 0)))
        cl = [float(r['c']) for r in d]
    except (KeyError, TypeError, ValueError):
        return None
    if cl[-3] <= 0:
        return None
    return (cl[-2] - cl[-3]) / cl[-3] * 100.0


QUAD = {
    ( True,  True): ('LONG_BUILDUP',   '🟢 LONG BUILDUP',   'Px↑ OI↑'),
    (False,  True): ('SHORT_BUILDUP',  '🔴 SHORT BUILDUP',  'Px↓ OI↑'),
    ( True, False): ('SHORT_COVERING', '🔵 SHORT COVERING', 'Px↑ OI↓'),
    (False, False): ('LONG_UNWINDING', '🟠 LONG UNWINDING', 'Px↓ OI↓'),
}


def fstate(fr):
    if fr is None: return 'UNKNOWN', ''
    if fr >= FUND_VEXT:  return 'LONG_VCROWD',  'long sangat ramai'
    if fr >= FUND_EXT:   return 'LONG_CROWD',   'long ramai'
    if fr <= -FUND_VEXT: return 'SHORT_VCROWD', 'short sangat ramai'
    if fr <= -FUND_EXT:  return 'SHORT_CROWD',  'short ramai'
    return 'BALANCED', ''


async def screen(s, it, sem, st):
    """Lolos HANYA jika ketiga TF menunjukkan arah harga DAN arah OI yang sama."""
    c = it['c']
    async with sem:
        oimap, lsr = await oi_series(s, c)
        if not oimap:
            st['no_oi'] += 1
            return None
        pxmap = {}
        for lb, iv, _ in TFS:
            pxmap[lb] = await px_change(s, c, iv)

    if any(pxmap.get(lb) is None or oimap.get(lb) is None for lb, _, _ in TFS):
        st['data_kurang'] += 1
        return None

    # Semua TF harus melewati ambang minimum (bukan gerakan nyaris-nol)
    if any(abs(pxmap[lb]) < MIN_MOVE[lb] or abs(oimap[lb]) < MIN_MOVE[lb]
           for lb, _, _ in TFS):
        st['terlalu_kecil'] += 1
        return None

    px_up = [pxmap[lb] > 0 for lb, _, _ in TFS]
    oi_up = [oimap[lb] > 0 for lb, _, _ in TFS]

    # SYARAT INTI: ketiga TF sepakat, untuk harga maupun OI
    if len(set(px_up)) != 1 or len(set(oi_up)) != 1:
        st['tf_tak_sepakat'] += 1
        return None

    st['lolos'] += 1
    key, label, pola = QUAD[(px_up[0], oi_up[0])]

    fr = it['fr']
    fs, fnote = fstate(fr)
    lsr_note = ''
    if lsr is not None:
        if lsr >= LSR_HI:   lsr_note = 'akun mayoritas long'
        elif lsr <= LSR_LO: lsr_note = 'akun mayoritas short'

    # SQUEEZE: buildup yang sisi ramainya SEARAH posisi itu sendiri -> rapuh
    sq = ((key == 'LONG_BUILDUP'  and (fs in ('LONG_CROWD', 'LONG_VCROWD')
                                       or (lsr is not None and lsr >= LSR_HI))) or
          (key == 'SHORT_BUILDUP' and (fs in ('SHORT_CROWD', 'SHORT_VCROWD')
                                       or (lsr is not None and lsr <= LSR_LO))))

    # Urutan = kekuatan OI rata-rata lintas TF. Transparan, bukan bobot karangan.
    strength = float(np.mean([abs(oimap[lb]) for lb, _, _ in TFS]))

    return {'sym': it['sym'], 'key': key, 'sq': sq, 'strength': strength,
            'px': pxmap, 'oi': oimap, 'fr': fr, 'fnote': fnote,
            'lsr': lsr, 'lsr_note': lsr_note, 'tv': it['tv']}


HL_API = "https://api.hyperliquid.xyz/info"
VOL_SAMPLE = 12      # berapa koin dipakai menghitung baseline volume
VOL_DAYS   = 8       # 1 hari berjalan + 7 hari pembanding


async def _hl_context(s):
    """Hyperliquid metaAndAssetCtxs - SATU panggilan POST untuk seluruh
    universe perp (~224 koin). Gratis, tanpa API key. Dipakai sebagai
    pembanding lintas-bursa karena Gate.io saja bisa bias (contoh: XLM
    tercatat tipis di Gate padahal likuid secara global).

    Hyperliquid adalah DEX, jadi kemungkinan tidak kena geo-block seperti
    Bybit. Kalau gagal, fungsi ini mengembalikan None dan sisanya tetap jalan.
    """
    try:
        async with s.post(HL_API, json={"type": "metaAndAssetCtxs"},
                          timeout=aiohttp.ClientTimeout(total=TMO)) as r:
            if r.status != 200:
                logging.warning(f"Hyperliquid HTTP {r.status}")
                return None
            data = await r.json()
    except Exception as e:
        logging.warning(f"Hyperliquid gagal: {e}")
        return None

    if not isinstance(data, list) or len(data) < 2:
        return None
    ctxs = data[1]
    if not isinstance(ctxs, list):
        return None

    vol = oi = 0.0
    up = down = 0
    for c in ctxs:
        try:
            v = float(c.get('dayNtlVlm') or 0)
            o = float(c.get('openInterest') or 0)
            mk = float(c.get('markPx') or 0)
            pv = float(c.get('prevDayPx') or 0)
        except (TypeError, ValueError):
            continue
        vol += v
        oi += o * mk
        if pv > 0 and mk > 0:
            if mk > pv:   up += 1
            elif mk < pv: down += 1
    return {'vol': vol, 'oi': oi, 'up': up, 'down': down, 'n': len(ctxs)}


async def _vol_baseline(s, uni):
    """Volume hari ini vs rata-rata 7 hari sebelumnya, dari candle 1d.

    Menjawab 'sepi atau ramai' TANPA perlu menyimpan data antar-run:
    baseline-nya diambil ulang tiap kali dari candle harian. Memakai
    sampel koin tervolume terbesar agar hemat panggilan API.
    """
    today = prev = 0.0
    ok = 0
    for it in uni[:VOL_SAMPLE]:
        d = await _get(s, f"{GATE}/candlesticks",
                       {"contract": it['c'], "interval": "1d",
                        "limit": VOL_DAYS}, "Gate 1d")
        if not d or not isinstance(d, list) or len(d) < 3:
            continue
        try:
            d = sorted(d, key=lambda x: int(x.get('t', 0)))
            vols = [float(x.get('sum') or x.get('v') or 0) for x in d]
        except (TypeError, ValueError):
            continue
        if len(vols) < 3:
            continue
        today += vols[-1]
        prev += float(np.mean(vols[:-1]))   # rata-rata hari-hari sebelumnya
        ok += 1
    if ok == 0 or prev <= 0:
        return None
    return today / prev


# ============================================================
# REGIME ALIGNMENT — ditambahkan
# ============================================================

def compute_regime_score(gate_pct_up, hl_pct_up=None):
    """Skor alignment [-1, +1] dari breadth (% pair naik) dua sumber.

    +1.0 = seluruhnya naik & dua sumber sepakat penuh
    -1.0 = seluruhnya turun & dua sumber sepakat penuh
     0.0 = 50/50 (choppy) ATAU dua sumber saling bertentangan arah

    Kalau Hyperliquid gagal diambil (hl_pct_up=None), fallback ke Gate saja
    tapi skor didiskon 30% -- konfirmasi silang tidak tersedia, jadi confidence
    diturunkan alih-alih dianggap sama kuatnya dengan skor 2-sumber.
    Return (skor, has_cross_confirm).
    """
    gate_score = (gate_pct_up - 50.0) / 50.0  # -1..+1

    if hl_pct_up is None:
        return round(gate_score * 0.7, 3), False

    hl_score = (hl_pct_up - 50.0) / 50.0
    agree = (gate_score > 0) == (hl_score > 0) or (abs(gate_score) < 0.05 and abs(hl_score) < 0.05)

    if not agree:
        # Dua sumber bertentangan arah -> SINYAL "tidak jelas", ditarik ke nol
        # (bukan dirata-ratakan naif, yang akan menyembunyikan konfliknya).
        return 0.0, False

    return round((gate_score + hl_score) / 2, 3), True


def classify_regime(score, has_cross_confirm):
    """Label + rekomendasi durasi hold, sesuai gradien WR yang terbukti di data historis.

    PENTING: skor ini murni breadth market (dari compute_regime_score), BELUM
    dikalikan dengan arah trade -- jadi labelnya BULLISH/BEARISH (arah market),
    bukan SEARAH/LAWAN (yang berarti dibandingkan ke posisi tertentu).
    "Kuat" di sini artinya breadth-nya tegas ke satu arah (mayoritas besar
    koin sepakat), bukan "kuat mendukung trade Anda" -- itu baru berarti
    sesuatu setelah dibandingkan ke arah LONG/SHORT yang Anda ambil sendiri.
    """
    if score >= ALIGN_STRONG:
        label, emoji = "BULLISH KUAT", "🟢"
        advice = "Mayoritas koin naik tegas. Riwayat: LONG yang searah ini layak ditahan sebagai swing (>24 jam) bila entry valid. Untuk SHORT, ini kondisi melawan -- riwayat WR rendah, pertimbangkan skip."
    elif score <= ALIGN_WEAK:
        label, emoji = "BEARISH KUAT", "🔴"
        advice = "Mayoritas koin turun tegas. Riwayat: SHORT yang searah ini layak ditahan sebagai swing (>24 jam) bila entry valid. Untuk LONG, ini kondisi melawan -- riwayat WR rendah, pertimbangkan skip."
    elif -0.10 <= score <= 0.10:
        label, emoji = "CHOPPY", "⚪"
        advice = "Tidak ada arah jelas, kedua arah sama-sama berisiko. Riwayat: kombinasi lemah+ditahan lama = WR terendah (10%). Kalau entry, potong cepat, jangan swing."
    else:
        label, emoji = ("BULLISH LEMAH", "🟡") if score > 0 else ("BEARISH LEMAH", "🟠")
        advice = "Sinyal ada tapi belum tegas. Riwayat: nyaris breakeven di kondisi ini -- selektif, arah manapun."

    confirm_note = "" if has_cross_confirm else " (⚠️ tanpa konfirmasi silang Hyperliquid)"
    return label, emoji, advice, confirm_note


def load_prev_regime_state():
    try:
        with open(REGIME_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_regime_state(score, label):
    state = {"score": score, "label": label,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(REGIME_STATE_FILE, "w") as f:
        json.dump(state, f)


def detect_reversal(current_score, prev_state):
    """Reversal = skor ganti tanda (melewati nol) DAN pergerakannya cukup
    besar (>=0.25) supaya bukan noise di sekitar nol."""
    if prev_state is None:
        return None
    prev_score = prev_state.get("score", 0)
    changed_sign = (current_score > 0.05 and prev_score < -0.05) or \
                   (current_score < -0.05 and prev_score > 0.05)
    if changed_sign and abs(current_score - prev_score) >= 0.25:
        direction = "BULLISH → BEARISH" if current_score < prev_score else "BEARISH → BULLISH"
        prev_time = prev_state.get("timestamp", "?")
        hh_mm = prev_time[11:16] if len(prev_time) > 16 else prev_time
        return f"⚡ REGIME BERBALIK: {direction} (dari cycle {hh_mm} UTC)"
    return None


def build_regime_section(gate_pct_up, hl_pct_up, weekend):
    """Return string siap ditempel ke pesan Telegram YANG SAMA, di ATAS
    daftar kuadran koin. Dipanggil dari build()."""
    score, has_confirm = compute_regime_score(gate_pct_up, hl_pct_up)
    label, emoji, advice, confirm_note = classify_regime(score, has_confirm)

    prev_state = load_prev_regime_state()
    reversal_msg = detect_reversal(score, prev_state)
    save_regime_state(score, label)

    lines = [
        f"{emoji} <b>REGIME: {label}</b>{confirm_note}",
        f"Skor alignment: {score:+.2f} (Gate {gate_pct_up:.0f}% naik"
        + (f", Hyperliquid {hl_pct_up:.0f}% naik" if hl_pct_up is not None else "")
        + ")",
        f"<i>{advice}</i>",
    ]
    if reversal_msg:
        lines.insert(0, reversal_msg)
    if weekend:
        lines.append("⚠️ Akhir pekan -- riwayat WR lebih rendah di hari ini, terlepas dari regime.")
    return "\n".join(lines)


async def market_context(s, uni):
    """Rangkuman kondisi pasar. Regime section pakai breadth dari sini,
    lalu deskriptif tambahan (volume, LSR) tetap seperti semula."""
    d = await _get(s, f"{GATE}/tickers", lbl="Gate tickers (context)")
    up = down = 0
    total_vol = 0.0
    if d:
        for i in d:
            c = i.get('contract', '')
            if not c.endswith('_USDT') or c.replace('_USDT', '') in NON_CRYPTO:
                continue
            try:
                chg = float(i.get('change_percentage') or 0)
                vol = float(i.get('volume_24h_quote') or 0)
            except (TypeError, ValueError):
                continue
            if vol < MIN_TURNOVER:
                continue
            total_vol += vol
            if chg > 0:   up += 1
            elif chg < 0: down += 1

    n = up + down
    if n == 0:
        return None, None
    pct_up = up / n * 100.0

    ratio = await _vol_baseline(s, uni)
    hl = await _hl_context(s)
    hl_pct_up = None
    if hl:
        hl_n = hl['up'] + hl['down']
        if hl_n > 0:
            hl_pct_up = hl['up'] / hl_n * 100.0

    lsrs = []
    for it in uni[:10]:
        try:
            _, lsr = await oi_series(s, it['c'])
            if lsr is not None:
                lsrs.append(lsr)
        except Exception:
            pass
    lsr_avg = float(np.mean(lsrs)) if lsrs else None

    now = datetime.now(timezone.utc).astimezone()
    weekend = now.weekday() >= 5

    def money(v):
        return f"${v/1e9:.1f}M" if v >= 1e9 else f"${v/1e6:.0f}jt"

    # --- Regime alignment section (baru) ---
    regime_section = build_regime_section(pct_up, hl_pct_up, weekend)

    # --- Deskriptif tambahan (seperti semula) ---
    if pct_up >= 60:   regime, emo = "BULLISH", "🟢"
    elif pct_up <= 40: regime, emo = "BEARISH", "🔴"
    else:              regime, emo = "NEUTRAL", "⚪"

    lines = [f"{emo} <b>{regime}</b> · {pct_up:.0f}% naik ({up}↑/{down}↓)"]
    vol_line = f"Vol Gate {money(total_vol)}"
    if ratio is not None:
        if ratio >= 1.3:   tag = "RAMAI"
        elif ratio <= 0.7: tag = "SEPI"
        else:              tag = "normal"
        vol_line += f" · {ratio:.1f}x rata2 7hr ({tag})"
    if weekend:
        vol_line += " · ⚠️ akhir pekan"
    lines.append(vol_line)

    if hl:
        hl_n = hl['up'] + hl['down']
        hl_pct = (hl['up'] / hl_n * 100.0) if hl_n else 0
        lines.append(f"Hyperliquid: vol {money(hl['vol'])} · OI {money(hl['oi'])} "
                     f"· {hl_pct:.0f}% naik")

    if lsr_avg is not None:
        lines.append(f"L/S top-10: {lsr_avg:.2f} "
                     f"(condong {'long' if lsr_avg > 1 else 'short'})")

    ctx_descriptive = "\n".join(lines)
    return regime_section, ctx_descriptive


def fmt(r, i):
    px, oi = r['px'], r['oi']
    tv = r['tv']
    tv_s = f"{tv/1e6:.0f}jt" if tv >= 1e6 else f"{tv/1e3:.0f}rb"
    fr_s = f"{r['fr']*100:+.2f}%" if r['fr'] is not None else "n/a"

    out = [f"<b>{i}. {r['sym']}</b>  ${tv_s}",
           "   " + " · ".join(f"{lb} {px[lb]:+.1f}%" for lb, _, _ in TFS),
           "   OI " + " · ".join(f"{oi[lb]:+.1f}%" for lb, _, _ in TFS),
           f"   Funding {fr_s}"]

    notes = [n for n in (r['fnote'], r['lsr_note']) if n]
    if notes:
        out[-1] += " — " + ", ".join(notes)
    return "\n".join(out)


def build(res, regime_section=None, ctx=None):
    now = datetime.now(timezone.utc).astimezone()
    sq = [r for r in res if r['sq']]
    groups = []
    for key in ('LONG_BUILDUP', 'SHORT_BUILDUP', 'SHORT_COVERING', 'LONG_UNWINDING'):
        items = [r for r in res if r['key'] == key and not r['sq']]
        if items:
            _, label, pola = next(v for v in QUAD.values() if v[0] == key)
            groups.append((items, label, pola))

    p = [f"📡 <b>OI SCREENER</b> · {now.strftime('%d %b %H:%M')}"]

    # Regime alignment DULU (paling atas, sebelum konteks deskriptif & kuadran)
    if regime_section:
        p.append(regime_section)
        p.append("─" * 28)

    if ctx:
        p.append(ctx)
    p.append("<i>syarat: 15m, 1h, 4h semua searah</i>")

    for items, label, pola in groups:
        p.append(f"\n<b>{label}</b> <i>{pola}</i>")
        p += [fmt(r, i) for i, r in enumerate(items[:TOPN], 1)]

    if sq:
        p.append("\n<b>⚡ SQUEEZE WATCH</b> <i>buildup tapi sisinya kelewat ramai</i>")
        p += [fmt(r, i) for i, r in enumerate(sq[:TOPN], 1)]

    if not res:
        p.append("\nTidak ada kandidat.")
    return "\n".join(p)


async def send(s, txt):
    if not TG_TOKEN or not TG_CHAT:
        logging.warning("Secret Telegram kosong - cetak ke log.")
        print("\n" + txt); return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    ch, cur = [], ""
    for ln in txt.split("\n"):
        if len(cur) + len(ln) + 1 > 3800:
            ch.append(cur); cur = ln
        else:
            cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        ch.append(cur)
    for c in ch:
        try:
            async with s.post(url, json={"chat_id": TG_CHAT, "text": c,
                                         "parse_mode": "HTML",
                                         "disable_web_page_preview": True},
                              timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status != 200:
                    logging.error(f"TG HTTP {r.status}: {await r.text()}")
                else:
                    logging.info("Pesan terkirim ke Telegram.")
        except Exception as e:
            logging.error(f"TG error: {e}")
        await asyncio.sleep(0.4)


async def main():
    async with aiohttp.ClientSession(headers={'Accept': 'application/json'}) as s:
        bs = await bybit_syms(s)
        uni = await universe(s, bs)
        if not uni:
            logging.error("Universe kosong dari Gate.io.")
            await send(s, "📡 <b>OI SCREENER</b>\n\n❌ Universe kosong. Cek log.")
            return

        sem = asyncio.Semaphore(SEM_N)
        st = {'no_oi': 0, 'data_kurang': 0, 'terlalu_kecil': 0,
              'tf_tak_sepakat': 0, 'lolos': 0}
        raw = await asyncio.gather(*[screen(s, i, sem, st) for i in uni],
                                   return_exceptions=True)

        errs = [r for r in raw if isinstance(r, Exception)]
        if errs:
            from collections import Counter
            for msg, n in Counter(f"{type(e).__name__}: {e}" for e in errs).most_common(3):
                logging.error(f"TASK GAGAL x{n} -> {msg}")

        res = [r for r in raw if r and not isinstance(r, Exception)]
        res.sort(key=lambda x: x['strength'], reverse=True)

        logging.info(
            f"FUNNEL {len(uni)} pair -> tanpa OI:{st['no_oi']} | "
            f"data kurang:{st['data_kurang']} | gerakan terlalu kecil:"
            f"{st['terlalu_kecil']} | TF tak sepakat:{st['tf_tak_sepakat']} | "
            f"LOLOS:{st['lolos']}")

        regime_section, ctx = await market_context(s, uni)
        await send(s, build(res, regime_section, ctx))


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================
# CATATAN WORKFLOW (GitHub Actions YAML) — untuk regime_state.json
# ============================================================
# Tambahkan step INI setelah "python screener.py" di workflow yang sudah ada,
# supaya reversal detection punya state dari cycle sebelumnya:
#
#   - name: Commit regime state
#     run: |
#       git config user.name "github-actions"
#       git config user.email "actions@github.com"
#       git add regime_state.json
#       git diff --staged --quiet || git commit -m "chore: update regime state [skip ci]"
#       git push
#
# "[skip ci]" WAJIB ada -- supaya commit ini sendiri tidak memicu run baru
# (infinite loop trigger). Tidak perlu OI_TELEGRAM_TOKEN baru; regime_section
# sudah ikut terkirim lewat send() yang sama, satu pesan, satu bot.
