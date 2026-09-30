#!/usr/bin/env python3
"""
dd_screener.py — Decision Dashboard (port dari Pine Script) + alert Telegram.

Menjawab 3 pertanyaan per coin, di candle TERTUTUP terakhir:
  1) Arah mana yang didukung data?        -> Bias score (-100..+100)
  2) Regime pasar apa?                    -> TRENDING / SIDEWAYS / CHOPPY / TRANSISI
  3) Kalau masuk, di mana stop & target?  -> Rencana ATR + ukuran posisi

Alert HANYA dikirim saat ada setup VALID BARU (state berubah di candle terakhir),
jadi tidak spam dan tidak butuh file state. Kalau satu run terlewat, sinyalnya
ikut terlewat (lihat DD_FRESH di bawah).

Sumber data (mengikuti pelajaran dari oi_screener.py):
  - Universe : Gate.io futures tickers (Bybit /tickers = 403 dari GitHub Actions)
  - Kline    : Bybit /v5/market/kline (lolos dari Actions), fallback Gate.io
  - Opsional : bybit_symbols.json di root repo untuk memfilter ke pair yang ada di Bybit

Pemakaian:
  python dd_screener.py                       # scan universe, kirim alert kalau ada
  python dd_screener.py --symbols BTCUSDT,ETHUSDT --full --dry   # lihat dashboard lengkap
  python dd_screener.py --selftest            # tes logika offline (data sintetis)

ENV (semua opsional):
  DD_TELEGRAM_TOKEN / DD_TELEGRAM_CHAT_ID   (fallback: OI_TELEGRAM_TOKEN / OI_TELEGRAM_CHAT_ID)
  DD_TF=1h  DD_HTF=1d  DD_TOP=60  DD_EQUITY=1000  DD_RISK_PCT=1
  DD_WATCH=0   DD_DIGEST_ALWAYS=0   DD_BTC_GATE=0

Semua bobot & ambang adalah HEURISTIK, belum divalidasi backtest.
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dd")

GATE = "https://api.gateio.ws/api/v4"
BYBIT = "https://api.bybit.com/v5/market"

SECS = {"15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
BYBIT_INT = {"15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}
GATE_INT = {"15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}

REG_NAME = {1: "TRENDING ↑", -1: "TRENDING ↓", 2: "SIDEWAYS", 3: "CHOPPY",
            0: "TRANSISI", 9: "N/A"}
REG_PLAY = {
    1: "Long dominan. Ikuti tren, cari pullback, jangan short melawan.",
    -1: "Short dominan. Ikuti tren, cari pullback naik, jangan long melawan.",
    2: "Range. Beli dekat support, jual dekat resistance, hindari kejar breakout.",
    3: "Whipsaw. Kurangi size atau skip, stop mudah tersapu.",
    0: "Belum jelas. Tunggu konfirmasi struktur.",
    9: "Histori belum cukup.",
}


# ─────────────────────────────── KONFIGURASI ──────────────────────────────
def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _s(name: str, default: str) -> str:
    v = (os.getenv(name) or "").strip()
    return v if v else default


class Cfg:
    tf = _s("DD_TF", "1h")
    htf = _s("DD_HTF", "1d")
    top = int(_f("DD_TOP", 60))
    # struktur
    sw_len = int(_f("DD_SW_LEN", 5))
    range_len = int(_f("DD_RANGE_LEN", 100))
    # bobot skor (sama dengan Pine)
    w_htf, w_ltf, w_str, w_mom, w_flow = 30.0, 20.0, 25.0, 15.0, 10.0
    # ambang
    bias_thr = _f("DD_BIAS_THR", 35)
    ext_thr = _f("DD_EXT_THR", 2.5)
    rsi_hi, rsi_lo = 75.0, 25.0
    hi_vol = 8.0
    # regime
    er_len = 20
    er_trend = _f("DD_ER_TREND", 0.30)
    er_chop = _f("DD_ER_CHOP", 0.18)
    adx_trend = _f("DD_ADX_TREND", 20)
    chop_vol = 60.0
    btc_gate = _s("DD_BTC_GATE", "0") == "1"
    # rencana
    equity = _f("DD_EQUITY", 1000)
    risk_pct = _f("DD_RISK_PCT", 1.0)
    stop_mult, min_stop_atr = 1.5, 1.0
    rr1, rr2 = 1.5, 3.0
    # perilaku alert
    watch = _s("DD_WATCH", "0") == "1"
    digest_always = _s("DD_DIGEST_ALWAYS", "0") == "1"
    fresh = int(_f("DD_FRESH", 1))


# ─────────────────────────────── INDIKATOR ────────────────────────────────
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def atr_(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def rsi_(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = rma(up, n) / rma(dn, n).replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def dmi_(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr_(df, n)
    pdi = 100 * rma(pdm, n) / a
    mdi = 100 * rma(mdm, n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pdi, mdi, rma(dx.fillna(0), n)


def regime_frame(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Kode: 1 TRENDING↑, -1 TRENDING↓, 2 SIDEWAYS, 3 CHOPPY, 0 TRANSISI, 9 N/A."""
    c = df["close"]
    e20, e50 = ema(c, 20), ema(c, 50)
    pdi, mdi, adx = dmi_(df)
    noise = c.diff().abs().rolling(cfg.er_len).sum()
    er = ((c - c.shift(cfg.er_len)).abs() / noise.replace(0, np.nan)).fillna(0.0)
    atrp = atr_(df) / c
    volp = atrp.rolling(101).apply(lambda x: (x[:-1] <= x[-1]).mean() * 100, raw=True).fillna(50.0)
    idx = np.arange(len(df))
    up = (c > e50) & (e20 > e50) & (pdi > mdi)
    dn = (c < e50) & (e20 < e50) & (mdi > pdi)
    tu = (adx >= cfg.adx_trend) & (er >= cfg.er_trend) & up
    td = (adx >= cfg.adx_trend) & (er >= cfg.er_trend) & dn
    low_er = er < cfg.er_chop
    code = np.select([idx < 60, tu, td, low_er & (volp >= cfg.chop_vol), low_er],
                     [9, 1, -1, 3, 2], default=0)
    return pd.DataFrame({"code": code.astype(int), "er": er, "adx": adx, "volp": volp},
                        index=df.index)


def structure_scan(df: pd.DataFrame, sw: int):
    """Swing + BOS/CHoCH. Return (struct[n], lastSH, lastSL, evt, bars_since_evt)."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    struct = np.zeros(n, dtype=int)
    sh = sl = math.nan
    shb = slb = True
    st = 0
    evt, evt_i = "–", -1
    for i in range(n):
        k = i - sw
        if k >= sw:
            if h[k] >= h[k - sw:i + 1].max():
                sh, shb = h[k], False
            if l[k] <= l[k - sw:i + 1].min():
                sl, slb = l[k], False
        if (not math.isnan(sh)) and (not shb) and c[i] > sh:
            shb, evt, evt_i = True, ("CHoCH ↑" if st == -1 else "BOS ↑"), i
            st = 1
        elif (not math.isnan(sl)) and (not slb) and c[i] < sl:
            slb, evt, evt_i = True, ("CHoCH ↓" if st == 1 else "BOS ↓"), i
            st = -1
        struct[i] = st
    return struct, sh, sl, evt, (n - 1 - evt_i) if evt_i >= 0 else None


def htf_frame(htf: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    c = htf["close"]
    idx = np.arange(len(htf))
    e50 = ema(c, 50).where(idx >= 50)
    e200 = ema(c, 200).where(idx >= 200)
    bull = (c > e50) & (e50 > e200)
    bear = (c < e50) & (e50 < e200)
    s = np.where(e200.notna(),
                 np.where(bull, 1.0, np.where(bear, -1.0, 0.0)),
                 np.where(e50.notna(), np.where(c > e50, 0.5, -0.5), 0.0))
    return pd.DataFrame({"ct": htf["t"] + SECS[cfg.htf], "htfS": s,
                         "reg": regime_frame(htf, cfg)["code"].values})


# ─────────────────────────────── ANALISIS ─────────────────────────────────
def analyze(sym: str, df: pd.DataFrame, htf: pd.DataFrame | None,
            btc: pd.DataFrame | None, cfg: Cfg) -> dict | None:
    """Semua dihitung di candle tertutup. Index -1 = candle tertutup terakhir."""
    if df is None or len(df) < 120:
        return None
    df = df.reset_index(drop=True)
    n = len(df)
    idx = np.arange(n)
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    ema20 = ema(c, 20)
    ema50 = ema(c, 50).where(idx >= 50)
    atr = atr_(df)
    rsi = rsi_(c)
    macd = ema(c, 12) - ema(c, 26)
    macdh = macd - ema(macd, 9)
    rng = (h - l).replace(0, np.nan)
    mfm = (((c - l) - (h - c)) / rng).fillna(0.0)
    cmf = ((mfm * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)).fillna(0.0)
    rvol = (v / v.rolling(20).mean().replace(0, np.nan)).fillna(0.0)
    hh, ll = h.rolling(cfg.range_len).max(), l.rolling(cfg.range_len).min()
    range_pos = ((c - ll) / (hh - ll).replace(0, np.nan) * 100).fillna(50.0)
    atr_pct = atr / c * 100
    ext = (c - ema20) / atr

    # HTF & regime
    reg = regime_frame(df, cfg)
    if htf is not None and len(htf) > 60:
        hf = htf_frame(htf, cfg)
        left = pd.DataFrame({"ct": df["t"] + SECS[cfg.tf]})
        m = pd.merge_asof(left, hf.sort_values("ct"), on="ct", direction="backward")
        htf_s = m["htfS"].fillna(0.0)
        reg_htf = m["reg"].fillna(9).astype(int)
    else:
        htf_s = pd.Series(0.0, index=df.index)
        reg_htf = pd.Series(9, index=df.index)

    if btc is not None:
        bf = pd.DataFrame({"t": btc["t"].values, "reg": regime_frame(btc.reset_index(drop=True), cfg)["code"].values})
        mb = pd.merge_asof(df[["t"]], bf.sort_values("t"), on="t", direction="backward")
        reg_btc = mb["reg"].fillna(9).astype(int)
    else:
        reg_btc = pd.Series(9, index=df.index)

    struct, last_sh, last_sl, evt, evt_ago = structure_scan(df, cfg.sw_len)
    struct_s = pd.Series(struct.astype(float), index=df.index)

    ltf_s = pd.Series(np.where(idx >= 50, np.where(c > ema50, .5, -.5) + np.where(ema20 > ema50, .5, -.5), 0.0), index=df.index)
    mom_s = pd.Series(np.where(macdh > 0, .4, -.4) + np.where(macdh > macdh.shift(1), .2, -.2)
                      + np.where(rsi > 50, .4, -.4), index=df.index)
    flow_s = (cmf / 0.15).clip(-1, 1)

    wsum = cfg.w_htf + cfg.w_ltf + cfg.w_str + cfg.w_mom + cfg.w_flow
    score = 100 * (cfg.w_htf * htf_s + cfg.w_ltf * ltf_s + cfg.w_str * struct_s
                   + cfg.w_mom * mom_s + cfg.w_flow * flow_s) / wsum
    d = pd.Series(np.where(score >= cfg.bias_thr, 1, np.where(score <= -cfg.bias_thr, -1, 0)), index=df.index)

    extended = ((d == 1) & ((ext > cfg.ext_thr) | (rsi > cfg.rsi_hi))) | \
               ((d == -1) & ((ext < -cfg.ext_thr) | (rsi < cfg.rsi_lo)))
    htf_conflict = (d != 0) & (np.sign(htf_s) == -d)
    r = reg["code"]
    regime_ok = (d != 0) & ((r == d) | (r == 9) | ((r == 0) & (score.abs() >= cfg.bias_thr + 20)))
    btc_against = (d != 0) & (reg_btc == -d)
    weak = extended | htf_conflict | (~regime_ok) | (cfg.btc_gate & btc_against)
    state = pd.Series(np.where(d == 0, 0, np.where(weak, d, 2 * d)), index=df.index)

    i = n - 1
    comps = {"htf": float(htf_s.iloc[i]), "ltf": float(ltf_s.iloc[i]), "str": float(struct_s.iloc[i]),
             "mom": float(mom_s.iloc[i]), "flow": float(flow_s.iloc[i])}
    dr = int(d.iloc[i])
    agree = sum(1 for x in comps.values() if dr != 0 and np.sign(x) == dr)

    res = {
        "sym": sym, "price": float(c.iloc[i]), "t": int(df["t"].iloc[i]),
        "score": float(score.iloc[i]), "dir": dr,
        "state": int(state.iloc[i]),
        "state_hist": [int(x) for x in state.iloc[-(cfg.fresh + 2):]],
        "comps": comps, "agree": agree,
        "regime": int(r.iloc[i]), "reg_er": float(reg["er"].iloc[i]), "reg_adx": float(reg["adx"].iloc[i]),
        "reg_htf": int(reg_htf.iloc[i]), "reg_btc": int(reg_btc.iloc[i]),
        "extended": bool(extended.iloc[i]), "htf_conflict": bool(htf_conflict.iloc[i]),
        "regime_ok": bool(regime_ok.iloc[i]), "btc_against": bool(btc_against.iloc[i]),
        "ext": float(ext.iloc[i]), "rsi": float(rsi.iloc[i]), "cmf": float(cmf.iloc[i]),
        "rvol": float(rvol.iloc[i]), "atr_pct": float(atr_pct.iloc[i]),
        "range_pos": float(range_pos.iloc[i]), "macd_up": bool(macdh.iloc[i] > macdh.iloc[i - 1]),
        "macd_pos": bool(macdh.iloc[i] > 0),
        "evt": evt, "evt_ago": evt_ago,
    }

    # rencana (identik dengan Pine)
    if dr != 0:
        atr_v, ema20_v = float(atr.iloc[i]), float(ema20.iloc[i])
        entry = ema20_v if res["extended"] else res["price"]
        if dr == 1:
            raw = (last_sl - 0.2 * atr_v) if (not math.isnan(last_sl) and last_sl < entry) else entry - cfg.stop_mult * atr_v
            stop = min(raw, entry - cfg.min_stop_atr * atr_v)
        else:
            raw = (last_sh + 0.2 * atr_v) if (not math.isnan(last_sh) and last_sh > entry) else entry + cfg.stop_mult * atr_v
            stop = max(raw, entry + cfg.min_stop_atr * atr_v)
        rd = abs(entry - stop)
        qty = (cfg.equity * cfg.risk_pct / 100) / rd if rd > 0 else float("nan")
        res["plan"] = {
            "entry": entry, "stop": stop, "tp1": entry + dr * rd * cfg.rr1, "tp2": entry + dr * rd * cfg.rr2,
            "stop_atr": rd / atr_v if atr_v else float("nan"), "qty": qty,
            "notional": qty * entry, "lev": qty * entry / cfg.equity,
        }

    res["verdict"] = verdict_of(res)
    res["warnings"] = warnings_of(res, cfg)
    return res


def verdict_of(r: dict) -> str:
    st, dr = r["state"], r["dir"]
    if st == 2:
        return "▲ LONG · SETUP VALID"
    if st == -2:
        return "▼ SHORT · SETUP VALID"
    if st == 0:
        return "■ NO TRADE · bias belum jelas"
    head = "▲ LONG BIAS" if dr == 1 else "▼ SHORT BIAS"
    if r["extended"]:
        return f"{head} · JANGAN KEJAR, TUNGGU PULLBACK"
    if r["htf_conflict"]:
        return f"{head} · MELAWAN HTF, RISIKO TINGGI"
    if not r["regime_ok"]:
        return f"{head} · REGIME {REG_NAME[r['regime']]}, TUNGGU"
    return f"{head} · BTC MELAWAN, RISIKO TINGGI"


def warnings_of(r: dict, cfg: Cfg) -> list[str]:
    w, dr = [], r["dir"]
    if r["extended"]:
        w.append(f"Overextended: {r['ext']:.1f} ATR dari EMA20, RSI {r['rsi']:.0f}")
    if r["htf_conflict"]:
        w.append("Melawan tren HTF")
    if dr != 0 and r["rvol"] < 0.8:
        w.append(f"Volume sepi ({r['rvol']:.1f}x)")
    if dr != 0 and "plan" in r and r["plan"]["stop_atr"] > 3:
        w.append(f"Stop jauh ({r['plan']['stop_atr']:.1f} ATR), R:R jelek")
    if r["atr_pct"] > cfg.hi_vol:
        w.append(f"Volatilitas ekstrem (ATR {r['atr_pct']:.1f}%), kecilkan size")
    if dr != 0 and r["regime"] in (2, 3):
        w.append(f"Regime {REG_NAME[r['regime']]}: sinyal searah tren sering gagal")
    if dr != 0 and r["regime"] == -dr:
        w.append("Melawan regime TF chart")
    if r["btc_against"]:
        w.append("BTC regime melawan arah")
    return w


# ─────────────────────────────── DATA ─────────────────────────────────────
async def get_json(session: aiohttp.ClientSession, url: str, params: dict | None = None, retries: int = 3):
    for a in range(retries):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status == 200:
                    return await r.json()
                log.warning("HTTP %s %s", r.status, url.split("?")[0])
                if r.status in (403, 451):   # diblokir: retry tidak membantu
                    return None
        except Exception as e:  # noqa: BLE001
            log.warning("%s %s", type(e).__name__, url.split("?")[0])
        await asyncio.sleep(0.6 * (a + 1))
    return None


def _clean(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    df = df.sort_values("t").drop_duplicates("t").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    # buang candle yang masih berjalan
    return df[df["t"] + SECS[tf] <= time.time()].reset_index(drop=True)


async def klines(session, sym: str, tf: str, limit: int = 1000) -> pd.DataFrame | None:
    d = await get_json(session, f"{BYBIT}/kline", {"category": "linear", "symbol": sym,
                                                   "interval": BYBIT_INT[tf], "limit": limit})
    lst = (((d or {}).get("result") or {}).get("list")) or []
    if lst:
        df = pd.DataFrame(lst, columns=["t", "open", "high", "low", "close", "volume", "turnover"])
        df["t"] = df["t"].astype("int64") // 1000
        return _clean(df, tf)
    # fallback Gate.io
    contract = sym[:-4] + "_USDT" if sym.endswith("USDT") else sym
    d = await get_json(session, f"{GATE}/futures/usdt/candlesticks",
                       {"contract": contract, "interval": GATE_INT[tf], "limit": min(limit, 1000)})
    if isinstance(d, list) and d:
        df = pd.DataFrame([{"t": int(x["t"]), "open": x["o"], "high": x["h"], "low": x["l"],
                            "close": x["c"], "volume": x.get("v", 0)} for x in d])
        return _clean(df, tf)
    return None


def load_bybit_filter() -> set[str] | None:
    p = Path(__file__).parent / "bybit_symbols.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        if isinstance(d, dict):
            d = d.get("symbols") or (d.get("result") or {}).get("list") or list(d.keys())
        out = set()
        for x in d:
            s = x.get("symbol") if isinstance(x, dict) else x
            if isinstance(s, str):
                out.add(s.replace("_", "").replace("-", "").upper())
        return out or None
    except Exception as e:  # noqa: BLE001
        log.warning("bybit_symbols.json tidak terbaca: %s", e)
        return None


async def universe(session, top: int) -> list[str]:
    d = await get_json(session, f"{GATE}/futures/usdt/tickers")
    rows = []
    for x in d or []:
        c = x.get("contract", "")
        if not c.endswith("_USDT"):
            continue
        try:
            vol = float(x.get("volume_24h_quote") or x.get("volume_24h_settle") or 0)
        except (TypeError, ValueError):
            vol = 0.0
        rows.append((c.replace("_", ""), vol))
    rows.sort(key=lambda z: z[1], reverse=True)
    allow = load_bybit_filter()
    syms = [s for s, _ in rows if (allow is None or s in allow)]
    return syms[:top]


# ─────────────────────────────── OUTPUT ───────────────────────────────────
def px(x: float) -> str:
    if x != x:
        return "n/a"
    a = abs(x)
    return f"{x:.2f}" if a >= 100 else f"{x:.4f}" if a >= 1 else f"{x:.6f}"


def arrow(s: float) -> str:
    return "▲" if s > 0.2 else "▼" if s < -0.2 else "■"


def render(r: dict, cfg: Cfg, full: bool) -> str:
    icon = {2: "🟢", -2: "🔴", 1: "🟡", -1: "🟡", 0: "⚪"}[r["state"]]
    L = [f"{icon} <b>{html.escape(r['verdict'])}</b>",
         f"<code>{r['sym']}</code> · {cfg.tf} · harga {px(r['price'])}",
         f"Skor {r['score']:+.0f}/100" + (f" · konfirmasi {r['agree']}/5" if r["dir"] else ""),
         f"Regime: {REG_NAME[r['regime']]} (ER {r['reg_er']:.2f}, ADX {r['reg_adx']:.0f}) · "
         f"HTF {REG_NAME[r['reg_htf']]} · BTC {REG_NAME[r['reg_btc']]}"]
    if full:
        c = r["comps"]
        L += [f"{arrow(c['htf'])} HTF  {arrow(c['ltf'])} TF chart  {arrow(c['str'])} Struktur "
              f"({r['evt']}{'' if r['evt_ago'] is None else f', {r['evt_ago']} bar lalu'})",
              f"{arrow(c['mom'])} Momentum (RSI {r['rsi']:.0f})  {arrow(c['flow'])} Flow "
              f"(CMF {r['cmf']:+.2f}, RVOL {r['rvol']:.1f}x)",
              f"Jarak EMA20 {r['ext']:+.1f} ATR · range {r['range_pos']:.0f}% · ATR {r['atr_pct']:.1f}%"]
    if r["dir"] != 0 and "plan" in r:
        p = r["plan"]
        pull = " (entry di pullback EMA20)" if r["extended"] else ""
        L += ["", f"<b>Rencana {'LONG' if r['dir'] == 1 else 'SHORT'}</b>{pull}",
              f"Entry {px(p['entry'])} · SL {px(p['stop'])} ({p['stop_atr']:.1f} ATR)",
              f"T1 {px(p['tp1'])} ({cfg.rr1:g}R) · T2 {px(p['tp2'])} ({cfg.rr2:g}R)",
              f"Risk {cfg.risk_pct:g}% dari {cfg.equity:g} → qty {p['qty']:.4g} · "
              f"nilai {p['notional']:.0f} · lev ~{p['lev']:.1f}x"]
    if r["warnings"]:
        L += ["", "⚠️ " + " · ".join(html.escape(w) for w in r["warnings"])]
    L.append(f"<i>Playbook: {html.escape(REG_PLAY[r['regime']])}</i>")
    return "\n".join(L)


def digest(results: list[dict], btc_reg: int) -> str:
    n = len(results)
    cnt = {k: sum(1 for r in results if r["regime"] == k) for k in (1, -1, 2, 3, 0)}
    nl = sum(1 for r in results if r["dir"] == 1)
    ns = sum(1 for r in results if r["dir"] == -1)
    net = (nl - ns) / n if n else 0
    dom = "LONG" if net > 0.15 else "SHORT" if net < -0.15 else "NETRAL / CAMPURAN"
    return (f"🧭 <b>DECISION DASHBOARD</b> · {Cfg.tf} / HTF {Cfg.htf}\n"
            f"BTC: <b>{REG_NAME[btc_reg]}</b>\n"
            f"Universe {n} coin → ↑{cnt[1]} ↓{cnt[-1]} sideways {cnt[2]} choppy {cnt[3]} transisi {cnt[0]}\n"
            f"Bias arah: long {nl} vs short {ns} → dominasi <b>{dom}</b> (ambang ±15%, heuristik)")


def tg_creds() -> tuple[str, str]:
    tok = _s("DD_TELEGRAM_TOKEN", _s("OI_TELEGRAM_TOKEN", ""))
    cid = _s("DD_TELEGRAM_CHAT_ID", _s("OI_TELEGRAM_CHAT_ID", ""))
    return tok, cid


async def send_telegram(session, text: str, dry: bool) -> None:
    tok, cid = tg_creds()
    if dry or not tok or not cid:
        if not dry:
            log.warning("Token/Chat ID Telegram kosong — dicetak ke log saja.")
        print("\n" + text)
        return
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3800:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    for ch in chunks:
        try:
            async with session.post(f"https://api.telegram.org/bot{tok}/sendMessage", json={
                "chat_id": cid, "text": ch, "parse_mode": "HTML", "disable_web_page_preview": True,
            }, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status != 200:
                    log.error("Telegram HTTP %s: %s", r.status, await r.text())
                else:
                    log.info("Pesan terkirim ke Telegram.")
        except Exception as e:  # noqa: BLE001
            log.error("Telegram error: %s", e)
        await asyncio.sleep(0.4)


# ─────────────────────────────── MAIN ─────────────────────────────────────
def pick_alerts(results: list[dict], cfg: Cfg) -> list[dict]:
    out = []
    for r in results:
        h = r["state_hist"]           # [.., prev, now] panjang fresh+2
        now, before = h[-1], h[-1 - cfg.fresh]
        if abs(now) == 2 and before != now:
            out.append(r)
        elif cfg.watch and abs(now) == 1 and before != now and abs(r["score"]) >= cfg.bias_thr + 15:
            out.append(r)
    out.sort(key=lambda r: (abs(r["state"]) != 2, -abs(r["score"])))
    return out


async def main_async(args) -> int:
    cfg = Cfg()
    if args.tf:
        cfg.tf = args.tf
    if args.htf:
        cfg.htf = args.htf
    if args.top:
        cfg.top = args.top
    Cfg.tf, Cfg.htf = cfg.tf, cfg.htf
    if cfg.tf not in SECS or cfg.htf not in SECS:
        log.error("TF harus salah satu dari %s", list(SECS))
        return 2

    async with aiohttp.ClientSession(headers={"Accept": "application/json"}) as session:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else await universe(session, cfg.top)
        if not syms:
            await send_telegram(session, "⚠️ <b>DD Screener</b>: universe kosong (Gate.io tickers gagal). Cek log Actions.", args.dry)
            return 1
        log.info("Scan %d simbol · TF %s · HTF %s", len(syms), cfg.tf, cfg.htf)

        btc_df = await klines(session, "BTCUSDT", cfg.tf)
        if btc_df is None:
            log.warning("Kline BTC gagal — regime BTC = N/A")

        sem = asyncio.Semaphore(6)
        htf_cache: dict[str, pd.DataFrame | None] = {}

        async def one(sym: str):
            async with sem:
                ltf = await klines(session, sym, cfg.tf)
                htf = await klines(session, sym, cfg.htf) if ltf is not None else None
            if ltf is None:
                return None
            try:
                return analyze(sym, ltf, htf, btc_df, cfg)
            except Exception as e:  # noqa: BLE001
                log.warning("analisis %s gagal: %s: %s", sym, type(e).__name__, e)
                return None

        results = [r for r in await asyncio.gather(*(one(s) for s in syms)) if r]
        log.info("Berhasil dianalisis: %d/%d", len(results), len(syms))
        if not results:
            await send_telegram(session, "⚠️ <b>DD Screener</b>: tidak ada data yang berhasil diambil (kemungkinan diblokir). Cek log Actions.", args.dry)
            return 1

        btc_reg = next((r["reg_btc"] for r in results), 9)
        alerts = results if args.full else pick_alerts(results, cfg)
        log.info("Sinyal baru: %d", len(alerts))

        if not alerts and not cfg.digest_always:
            log.info("Tidak ada sinyal baru — tidak kirim pesan.")
            return 0

        parts = [digest(results, btc_reg), ""]
        if alerts:
            for r in alerts[:12]:
                parts += [render(r, cfg, args.full), "━━━━━━━━━━"]
            if len(alerts) > 12:
                parts.append(f"(+{len(alerts) - 12} sinyal lain tidak ditampilkan)")
        else:
            parts.append("Tidak ada setup valid baru di candle terakhir.")
        parts.append("<i>Heuristik, belum di-backtest. Fee/funding/slippage tidak dihitung. Cek chart sebelum masuk.</i>")
        await send_telegram(session, "\n".join(parts), args.dry)
    return 0


# ─────────────────────────────── SELFTEST ─────────────────────────────────
def _synth(n: int, secs: int, drift: float, seed: int, vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[100.0], close[:-1]])
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    t0 = int(time.time()) - n * secs - 10
    return pd.DataFrame({"t": t0 + np.arange(n) * secs, "open": open_, "high": hi, "low": lo,
                         "close": close, "volume": rng.uniform(500, 1500, n)})


def selftest() -> int:
    cfg = Cfg()
    Cfg.tf, Cfg.htf = "1h", "1d"
    btc = _synth(1000, 3600, 0.0008, 1)
    cases = {"UPTREND": (0.0012, 2), "DOWNTREND": (-0.0012, 3), "SIDEWAYS": (0.0, 4)}
    ok = True
    for name, (drift, seed) in cases.items():
        ltf = _synth(1000, 3600, drift, seed, 0.006 if name == "SIDEWAYS" else 0.01)
        htf = _synth(400, 86400, drift * 24, seed + 10, 0.03)
        r = analyze(name, ltf, htf, btc, cfg)
        if r is None:
            print(name, "-> None"); ok = False; continue
        print(f"{name:10s} skor {r['score']:+6.1f} | regime {REG_NAME[r['regime']]:10s} | "
              f"HTF {REG_NAME[r['reg_htf']]:10s} | state {r['state']:+d} | {r['verdict']}")
        print(render(r, cfg, True).replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")
              .replace("<i>", "").replace("</i>", ""), "\n")
    print("SELFTEST", "OK" if ok else "GAGAL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decision Dashboard screener")
    ap.add_argument("--symbols", help="daftar simbol Bybit dipisah koma, mis. BTCUSDT,ETHUSDT")
    ap.add_argument("--full", action="store_true", help="tampilkan dashboard lengkap semua simbol yang diberikan (abaikan filter sinyal)")
    ap.add_argument("--dry", action="store_true", help="cetak ke layar, jangan kirim Telegram")
    ap.add_argument("--tf")
    ap.add_argument("--htf")
    ap.add_argument("--top", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    sys.exit(asyncio.run(main_async(a)))
