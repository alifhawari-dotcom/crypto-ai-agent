"""OI Quadrant Screener - Gate.io data, filter Bybit (jika endpoint lolos).
Harga^+OI^=LONG BUILDUP | Harga v+OI^=SHORT BUILDUP
Funding/LSR ekstrem = sinyal CROWDING (rapuh), bukan sinyal arah.
Threshold BELUM dikalibrasi backtest. Deskriptif, bukan prediktif."""
import os, asyncio, logging, json, aiohttp, numpy as np
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
TG_TOKEN = (os.getenv('OI_TELEGRAM_TOKEN') or '').strip()
TG_CHAT = (os.getenv('OI_TELEGRAM_CHAT_ID') or '').strip()
DIAG = os.getenv('OI_DIAGNOSTIC', '').strip().lower() == 'true'
# Default RINGKAS. Set secret OI_DETAIL=true kalau mau rincian per-timeframe.
DETAIL = os.getenv('OI_DETAIL', '').strip().lower() == 'true'

# Kalibrasi #3 (14 Sep 2026, berbasis DISTRIBUSI nyata n=77, sesi 03:55 WIB):
#   OI%  -> p50=0.038 p70=0.086 p80=0.123 p90=0.286 max=8.622
#   PX%  -> p50=0.296 p70=0.738 p80=1.132 p90=1.687 max=4.426
# Temuan: OI bergerak JAUH lebih lambat dari harga (median 0.038% vs 0.296%).
# Threshold lama OI=0.3 setara p90 -> cuma 10% pair lolos, terlalu ketat.
# OI_MIN=0.08 (~p70, 30% pair lolos), PX_MIN=0.25 (~p45).
# CATATAN: ini sesi dini hari (sepi). Cek ulang DISTRIBUSI saat sesi ramai;
# kalau kandidat jadi terlalu banyak, naikkan OI_MIN ke p80 (~0.12).
OI_MIN, PX_MIN, RVOL_MIN = 0.08, 0.25, 1.0
FUND_EXT, FUND_VEXT = 0.0005, 0.0010
LSR_HI, LSR_LO = 2.0, 0.5
# TIMING: bedakan sinyal DINI (OI bergerak, harga belum) vs TERLAMBAT
# (harga sudah jauh bergerak). Rasio = |OI%| / |Px%|.
# Tinggi = posisi dibangun tapi harga belum ikut -> masih dini.
LEAD_RATIO_EARLY = 1.5     # OI bergerak >=1.5x lebih besar dari harga
PX_EXTENDED      = 8.0     # harga sudah bergerak >= ini % -> dianggap telat
# Universe DIPERLEBAR (14 Sep 2026): turnover min $3jt -> $500rb.
# Aman karena filter Bybit jadi penyaring kualitas: Bybit melisting jauh
# lebih sedikit coin dari Gate.io (1.700+), jadi yang lolos sudah melewati
# standar listing Bybit. MAX_SYM dinaikkan, semaphore dinaikkan agar runtime
# tetap wajar (tiap simbol = 6 panggilan API).
MIN_TURNOVER, MIN_TF, MAX_SYM, TOPN = 500_000, 2, 400, 10
MIN_SCORE = 60   # kandidat di bawah ini tidak dikirim (kurangi kebisingan)

# Token SAHAM / KOMODITAS / FOREX di Gate.io — BUKAN crypto.
# Dibuang karena: (a) bukan instrumen yang kamu tradingkan, (b) saat bursa
# asalnya tutup, OI bisa bergerak sementara harga diam -> sinyal palsu.
NON_CRYPTO = {
    # saham AS / semikonduktor
    'NVDA','META','MU','SOXL','SOXS','AVGO','ORCL','IBM','AAOI','INTC','AMD',
    'TSLA','AAPL','MSFT','GOOGL','AMZN','NFLX','COIN','MSTR','CRWV','ASML',
    'SNDK','WDC','SKHYNIX','SKHY','SAMSUNG','CXMT','DRAM','4STOCK','LITE',
    'CRCL','OPENAI','ANTHROPIC','SPX','QQQ','ESPORTS','MVLL','MET','RAVE',
    # komoditas / forex
    'XAU','XAG','XAUT','PAXG','OIL','GOLD','SILVER',
}
TFS = [('15m','15m'), ('1h','1h'), ('4h','4h')]
SEM_N, RETRIES, DELAY, TMO = 12, 3, 3, 15
GATE = "https://api.gateio.ws/api/v4/futures/usdt"
BYBIT = "https://api.bybit.com/v5/market"


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


BYBIT_SYMBOLS_FILE = "bybit_symbols.json"


def load_bybit_symbols_local():
    """Baca daftar simbol Bybit dari file lokal (diambil manual dari IP
    non-US via fetch_bybit_symbols.py, karena endpoint live Bybit diblokir
    dari GitHub Actions). Lebih andal daripada memanggil API tiap run."""
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
        logging.warning(f"{BYBIT_SYMBOLS_FILE} rusak/tidak valid: {e}")
    return None


async def bybit_syms(s):
    # PRIORITAS 1: file lokal (endpoint live Bybit diblokir dari Actions)
    local = load_bybit_symbols_local()
    if local:
        return local

    # FALLBACK: coba endpoint live (kadang-kadang bisa berubah status)
    d = await _get(s, f"{BYBIT}/instruments-info",
                   {"category": "linear", "limit": 1000}, "Bybit instruments-info")
    if not d or d.get('retCode') != 0:
        logging.warning("Simbol Bybit tak terambil (file lokal & API) - "
                        "filter Bybit NONAKTIF.")
        return None
    sy = {i['symbol'] for i in d.get('result', {}).get('list', [])
          if i.get('symbol', '').endswith('USDT') and i.get('status') == 'Trading'}
    logging.info(f"Bybit: {len(sy)} simbol tradable (live API)")
    return sy


async def on_bybit(s, sym):
    """Cek apakah simbol tradable di Bybit lewat endpoint KLINE — satu-satunya
    endpoint Bybit yang terbukti lolos geo-block dari GitHub Actions.
    Dipanggil HANYA untuk kandidat yang sudah lolos screening (jumlahnya
    sedikit), jadi ringan dan tidak memicu rate limit."""
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
    out, skip, skip_noncrypto = [], 0, 0
    for i in d:
        c = i.get('contract', '')
        if not c.endswith('_USDT'):
            continue
        base = c.replace('_USDT', '')
        if base in NON_CRYPTO:
            skip_noncrypto += 1; continue
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
                 f"non-crypto: {skip_noncrypto})")
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


async def stats_oi_series(s, c):
    """Ambil SATU seri 15m, lalu turunkan perubahan OI dengan lookback berbeda:
    1 bar=15m, 4 bar=1h, 16 bar=4h.

    Kenapa begini: agregasi contract_stats per-interval milik Gate menghasilkan
    perubahan OI yang IDENTIK antara 1h dan 4h (terbukti 14 Sep 2026 - bucket
    terakhir 1h dan 4h sama-sama jatuh di batas yang sama). Menghitung sendiri
    dari satu seri konsisten menghilangkan ketergantungan pada agregasi itu,
    sekaligus memangkas 3 panggilan API jadi 1.
    """
    d = await _get(s, f"{GATE}/contract_stats",
                   {"contract": c, "interval": "15m", "limit": 20},
                   "Gate contract_stats")
    if not d or not isinstance(d, list) or len(d) < 17:
        return {}, None, None
    try:
        d = sorted(d, key=lambda x: x.get('time', 0))
    except Exception:
        pass

    oi = [_f(x, 'open_interest_usd', 'open_interest') for x in d]
    if any(v is None for v in oi[-17:]):
        return {}, None, None

    if DIAG:
        logging.info(f"[DIAG-OI] {c} seri15m OI 3 terakhir={oi[-3:]} "
                     f"lookback1={oi[-2]} lookback4={oi[-5]} lookback16={oi[-17]}")

    def chg(lb):
        base = oi[-1 - lb]
        return ((oi[-1] - base) / base * 100.0) if base and base > 0 else None

    out = {'15m': chg(1), '1h': chg(4), '4h': chg(16)}
    last = d[-1]
    return out, _f(last, 'lsr_account'), _f(last, 'top_lsr_account')


async def kline(s, c, iv):
    """Pakai candle TERAKHIR YANG SUDAH TERTUTUP, bukan yang sedang berjalan.
    (Bug versi lama: membandingkan candle in-progress dengan rata-rata candle
     lengkap -> RVOL 4h/15m selalu <1 karena candle-nya memang belum selesai,
     bukan karena volumenya rendah.)"""
    d = await _get(s, f"{GATE}/candlesticks",
                   {"contract": c, "interval": iv, "limit": 30}, "Gate candles")
    if not d or not isinstance(d, list) or len(d) < 24:
        return None, None
    try:
        d = sorted(d, key=lambda x: int(x.get('t', 0)))
        cl = np.array([float(r['c']) for r in d])
        vo = np.array([float(r['v']) for r in d])
        ts = [int(r.get('t', 0)) for r in d]
    except (KeyError, TypeError, ValueError):
        return None, None

    if DIAG:
        logging.info(f"[DIAG-TF] {c} iv={iv} n={len(d)} "
                     f"t_terakhir={ts[-1]} t_sebelum={ts[-2]} "
                     f"selisih_detik={ts[-1]-ts[-2]} close={cl[-2]:.6f}")

    # index -1 = candle berjalan (dibuang), -2 = candle tertutup terakhir
    if len(cl) < 24 or cl[-3] <= 0:
        return None, None
    px = (cl[-2] - cl[-3]) / cl[-3] * 100.0
    va = vo[-23:-2].mean()          # 21 candle tertutup sebelum candle -2
    rv = float(vo[-2] / va) if va > 0 else 0.0
    return px, rv


def quad(px, oi):
    if px is None or oi is None:
        return None
    if abs(px) < PX_MIN or abs(oi) < OI_MIN:
        return 'NEUTRAL'
    if px > 0 and oi > 0: return 'LONG_BUILDUP'
    if px < 0 and oi > 0: return 'SHORT_BUILDUP'
    if px > 0 and oi < 0: return 'SHORT_COVERING'
    return 'LONG_UNWINDING'


def fstate(fr):
    if fr is None: return 'UNKNOWN', 'x'
    if fr >= FUND_VEXT: return 'LONG_VCROWD', '[!!]'
    if fr >= FUND_EXT: return 'LONG_CROWD', '[!]'
    if fr <= -FUND_VEXT: return 'SHORT_VCROWD', '[!!]'
    if fr <= -FUND_EXT: return 'SHORT_CROWD', '[!]'
    return 'BALANCED', 'ok'



def pick_tf(r_tf, dom):
    """TF representatif: yang kuadrannya cocok label & volumenya paling kuat.
    Dipakai BERSAMA oleh perhitungan timing dan tampilan Telegram, supaya
    label tidak pernah dihitung dari TF berbeda dengan angka yang dibaca user."""
    cands = [(lb, t) for lb, t in r_tf.items()
             if t.get('q') == dom and t.get('px') is not None]
    if cands:
        return max(cands, key=lambda x: (x[1].get('rv') or 0))
    return '1h', r_tf.get('1h', {})


async def screen(s, it, sem, st):
    async with sem:
        oimap, lsr_v, tlsr_v = await stats_oi_series(s, it['c'])
        tf = {}
        for lb, iv in TFS:
            px, rv = await kline(s, it['c'], iv)
            oi = oimap.get(lb)
            tf[lb] = {'oi': oi, 'px': px, 'rv': rv, 'lsr': lsr_v,
                      'tlsr': tlsr_v, 'q': quad(px, oi)}
    # Rekam nilai riil 1h untuk laporan distribusi (dasar kalibrasi threshold)
    t1 = tf.get('1h', {})
    if t1.get('oi') is not None:
        st['_oi_vals'].append(abs(t1['oi']))
    if t1.get('px') is not None:
        st['_px_vals'].append(abs(t1['px']))

    if all(v['oi'] is None for v in tf.values()):
        st['no_oi'] += 1; return None
    qs = [v['q'] for v in tf.values() if v['q'] and v['q'] != 'NEUTRAL']
    if not qs:
        st['neutral'] += 1; return None
    dom = max(set(qs), key=qs.count)
    con = qs.count(dom)
    if con < MIN_TF:
        st['inconsist'] += 1; return None
    if dom not in ('LONG_BUILDUP', 'SHORT_BUILDUP',
                   'LONG_UNWINDING', 'SHORT_COVERING'):
        st['not_buildup'] += 1; return None
    rv1 = tf.get('1h', {}).get('rv') or 0.0
    if rv1 < RVOL_MIN:
        st['low_rvol'] += 1; return None
    st['pass'] += 1

    fr = it['fr']
    fs, fe = fstate(fr)
    lsr = tf.get('1h', {}).get('lsr')
    tlsr = tf.get('1h', {}).get('tlsr')
    lhi = lsr is not None and lsr >= LSR_HI
    llo = lsr is not None and lsr <= LSR_LO
    sq = ((dom == 'LONG_BUILDUP' and (fs in ('LONG_CROWD', 'LONG_VCROWD') or lhi)) or
          (dom == 'SHORT_BUILDUP' and (fs in ('SHORT_CROWD', 'SHORT_VCROWD') or llo)))

    # Funding ekstrem ke arah MANA PUN = informasi penting, harus ditandai.
    # (Bug versi lama: hanya ditandai kalau crowding SEARAH dengan buildup,
    #  sehingga funding -2% pada long buildup lolos tanpa peringatan.)
    fr_ext, fr_note = False, ""
    if fr is not None and abs(fr) >= FUND_VEXT * 3:
        fr_ext = True
        side = "long" if fr > 0 else "short"
        fr_note = f"{side} bayar sangat mahal - potensi squeeze {('turun' if fr > 0 else 'naik')}"

    # --- TIMING SCORE ---
    # Pakai TF yang SAMA dengan yang nanti ditampilkan di Telegram.
    tf_lb, t_used = pick_tf(tf, dom)
    px_abs = abs(t_used.get('px') or 0)
    oi_abs = abs(t_used.get('oi') or 0)
    lead = (oi_abs / px_abs) if px_abs > 0.01 else 0
    if px_abs >= PX_EXTENDED:
        timing, t_emoji = 'EXTENDED', '🕐'
    elif lead >= LEAD_RATIO_EARLY:
        timing, t_emoji = 'EARLY', '🌱'
    else:
        timing, t_emoji = 'ONGOING', '▶️'

    sc = con * 15 + min(int(rv1 * 10), 25)
    # Timing mempengaruhi skor: dini diberi bonus, terlambat diberi penalti.
    if timing == 'EARLY':
        sc += 12
    elif timing == 'EXTENDED':
        sc -= 15
    # Exit-flow (posisi keluar) secara literatur lebih lemah/ambigu dibanding
    # buildup (uang baru masuk) -> penalti agar urutan skor mencerminkan itu.
    if dom in ('LONG_UNWINDING', 'SHORT_COVERING'):
        sc -= 8
    ois = [abs(v['oi']) for v in tf.values() if v['oi'] is not None]
    if ois:
        sc += min(int(np.mean(ois) * 2), 20)
    if fs == 'BALANCED':
        sc += 10
    return {'sym': it['sym'], 'q': dom, 'con': con, 'ntf': len(TFS), 'fr': fr,
            'fr_extreme': fr_ext, 'fr_note': fr_note,
            'timing': timing, 't_emoji': t_emoji, 'lead': round(lead, 1),
            'tf_used': tf_lb,
            'fs': fs, 'fe': fe, 'lsr': lsr, 'tlsr': tlsr, 'sq': sq,
            'rv': round(rv1, 2), 'sc': max(0, min(100, sc)), 'tf': tf}


def fmt(r, i):
    """Default RINGKAS (2 baris). Set OI_DETAIL=true untuk rincian per-TF.
    TF yang ditampilkan = TF yang mendukung label, dengan RVOL tertinggi."""
    # TF yang sama persis dengan yang dipakai menghitung timing
    lb, t = pick_tf(r['tf'], r['q'])

    px = f"{t['px']:+.1f}%" if t.get('px') is not None else "n/a"
    oi = f"{t['oi']:+.2f}%" if t.get('oi') is not None else "n/a"
    rv = f"{t['rv']:.1f}x" if t.get('rv') is not None else "n/a"
    fr = f"{r['fr']*100:+.3f}%" if r['fr'] is not None else "n/a"
    ls = f" · L/S {r['lsr']:.2f}" if r['lsr'] is not None else ""

    tmg = r.get('t_emoji', '')
    out = [f"<b>{i}. {r['sym']}</b>  ·  {r['sc']}/100  ·  TF {r['con']}/{r['ntf']}  {tmg}",
           f"    {lb}  Px {px} · OI {oi} · Vol {rv}",
           f"    Fund {fr}{ls}"]

    if r.get('timing') == 'EARLY':
        out.append(f"   🌱 DINI — OI {r['lead']}x lebih cepat dari harga")
    elif r.get('timing') == 'EXTENDED':
        out.append(f"   🕐 TERLAMBAT — harga sudah bergerak jauh")
    if r.get('fr_extreme'):
        out.append(f"   🚨 {r['fr_note']}")
    if r.get('sq'):
        sd = "long" if r['q'] == 'LONG_BUILDUP' else "short"
        out.append(f"   ⚠️ sisi {sd} sudah terlalu ramai")

    if DETAIL:
        for l2, _ in TFS:
            t2 = r['tf'].get(l2, {})
            m = "✓" if t2.get('q') == r['q'] else " "
            p2 = f"{t2['px']:+.2f}%" if t2.get('px') is not None else "n/a"
            o2 = f"{t2['oi']:+.2f}%" if t2.get('oi') is not None else "n/a"
            v2 = f"{t2['rv']:.1f}x" if t2.get('rv') is not None else "n/a"
            out.append(f"      {m}{l2}: Px {p2} OI {o2} Vol {v2}")
    return "\n".join(out)


DIV = "━" * 18

def build(res, bfilter):
    res = [r for r in res if r['sc'] >= MIN_SCORE]
    now = datetime.now(timezone.utc).astimezone()
    L  = [r for r in res if r['q'] == 'LONG_BUILDUP'   and not r['sq']]
    S  = [r for r in res if r['q'] == 'SHORT_BUILDUP'  and not r['sq']]
    LU = [r for r in res if r['q'] == 'LONG_UNWINDING' and not r['sq']]
    SC = [r for r in res if r['q'] == 'SHORT_COVERING' and not r['sq']]
    Q  = [r for r in res if r['sq']]

    p = [f"📡 <b>OI SCREENER</b>",
         f"{now.strftime('%d %b %Y · %H:%M')} WIB",
         f"{len(res)} kandidat" + ("" if bfilter else "  ⚠️ blm difilter Bybit")]

    secs = [
        (L,  "🟢 LONG BUILDUP",   "Px↑ OI↑ — uang baru masuk long", 8),
        (S,  "🔴 SHORT BUILDUP",  "Px↓ OI↑ — uang baru masuk short", 8),
        (LU, "🟠 LONG UNWINDING", "Px↓ OI↓ — long keluar. Short menarik bila "
                                  "habis pump; waspada bila sudah turun panjang", 6),
        (SC, "🔵 SHORT COVERING", "Px↑ OI↓ — short tutup. Bullish tapi cepat habis", 6),
        (Q,  "⚡ SQUEEZE WATCH",  "Buildup tapi sisinya kelewat ramai — rawan "
                                  "cascade. Amati, jangan masuk", 6),
    ]
    for items, title, desc, cap in secs:
        if not items:
            continue
        p.append("")
        p.append(DIV)
        p.append(f"<b>{title}</b>")
        p.append(f"<i>{desc}</i>")
        p.append(DIV)
        for i, r in enumerate(items[:cap], 1):
            p.append("")
            p.append(fmt(r, i))

    if not res:
        p.append("\nTidak ada kandidat siklus ini.")
    p.append("")
    p.append(DIV)
    p.append("<i>Deskriptif, bukan rekomendasi. Cek chart sebelum masuk.</i>")
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
            await send(s, "📡 <b>OI QUADRANT SCREENER</b>\n\n"
                          "❌ Universe kosong - Gate.io tickers gagal. Cek log Actions.")
            return
        sem = asyncio.Semaphore(SEM_N)
        st = {'no_oi': 0, 'neutral': 0, 'inconsist': 0,
              'not_buildup': 0, 'low_rvol': 0, 'pass': 0,
              '_oi_vals': [], '_px_vals': []}
        raw = await asyncio.gather(*[screen(s, i, sem, st) for i in uni],
                                   return_exceptions=True)
        res = [r for r in raw if r and not isinstance(r, Exception)]
        res.sort(key=lambda x: x['sc'], reverse=True)

        # FILTER BYBIT (lapis akhir): buang kandidat yang tidak tradable di
        # Bybit. Pakai kline karena tickers/instruments-info kena geo-block.
        bybit_ok = bs is not None   # sudah terfilter di tahap universe?
        if not bybit_ok and res:
            checks = await asyncio.gather(
                *[on_bybit(s, r['sym']) for r in res[:60]],
                return_exceptions=True)
            keep, dropped = [], 0
            for r, ok in zip(res[:60], checks):
                if ok is True:
                    keep.append(r)
                else:
                    dropped += 1
            ok_n = sum(1 for c in checks if c is True)
            logging.info(f"Cek Bybit via kline: {ok_n}/{len(checks)} kandidat "
                         f"terkonfirmasi ada di Bybit")
            if ok_n > 0:
                logging.info(f"Filter Bybit (via kline): {len(keep)} lolos, "
                             f"{dropped} dibuang (tidak ada di Bybit)")
                res = keep
                bybit_ok = True
            else:
                logging.warning("Cek Bybit via kline gagal total — "
                                "filter tidak diterapkan.")
        logging.info(
            f"FUNNEL {len(uni)} pair -> tanpa OI:{st['no_oi']} | "
            f"NEUTRAL:{st['neutral']} | TF tak konsisten:{st['inconsist']} | "
            f"bukan buildup:{st['not_buildup']} | RVOL rendah:{st['low_rvol']} | "
            f"LOLOS:{st['pass']}")

        # DISTRIBUSI NYATA -> dasar kalibrasi threshold, bukan tebakan.
        # Baca persentil: kalau mau ~20% pair lolos, set threshold di p80.
        for nm, vals, cur in (('OI%', st['_oi_vals'], OI_MIN),
                              ('PX%', st['_px_vals'], PX_MIN)):
            if vals:
                a = np.array(vals)
                logging.info(
                    f"DISTRIBUSI {nm} (1h, n={len(a)}) threshold_kini={cur} -> "
                    f"p50={np.percentile(a,50):.3f} p70={np.percentile(a,70):.3f} "
                    f"p80={np.percentile(a,80):.3f} p90={np.percentile(a,90):.3f} "
                    f"max={a.max():.3f}")
        await send(s, build(res, bybit_ok))


if __name__ == "__main__":
    asyncio.run(main())
