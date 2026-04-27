import os
import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

TG_TOKEN   = (os.getenv('TELEGRAM_TOKEN')   or '').strip()
TG_CHAT_ID = (os.getenv('TELEGRAM_CHAT_ID') or '').strip()

# ── CONSTANTS ─────────────────────────────────────────────────
TOP_COINS_BY_RVOL = 50
RANKED_CANDIDATES = 12
CANDLES_REQUIRED  = 100
ADX_PERIOD        = 14
ADX_MIN_NUCLEAR   = 18.0
ADX_MIN_NORMAL    = 22.0
CORR_WINDOW       = 30
CORR_THRESHOLD    = 0.78
EMA_FAST, EMA_SLOW = 20, 50
SWING_LOOKBACK    = 10
SEMAPHORE_P1      = 5
SEMAPHORE_P2      = 3
PHASE2_DELAY      = 0.3
MAX_RETRIES       = 3
RETRY_DELAY       = 5
WEEKEND_BLACKOUT  = True
CONV_VALID        = 55
CONV_HIGH         = 60
CONV_INST         = 75

# Gate.io v4 API — futures USDT perpetual
GATE_TICKER_URL = "https://api.gateio.ws/api/v4/futures/usdt/tickers"
GATE_KLINE_URL  = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"

# Bybit kline sebagai fallback
BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_TF_MAP    = {'5m': '5', '15m': '15', '1h': '60'}
GATE_TF_MAP     = {'5m': '5m', '15m': '15m', '1h': '1h'}

# ── INDICATORS ────────────────────────────────────────────────
def calc_atr(df, period=14):
    high, low, pc = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    median = tr.rolling(50, min_periods=1).median()
    clean = pd.Series(np.where(tr > median * 4, median, tr), index=df.index)
    return clean.rolling(period).mean()

def calc_rsi(close, period=14):
    d = close.diff()
    gain = d.where(d > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    # FIX: alpha=1/period (bukan 1/1/period yang ada di versi AI lain)
    loss = (-d.where(d < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / np.where(loss == 0, 1e-9, loss)))

def calc_ema(close, span):
    return close.ewm(span=span, adjust=False).mean()

def calc_macd(close):
    m = calc_ema(close, 12) - calc_ema(close, 26)
    s = m.ewm(span=9, adjust=False).mean()
    return m, s, m - s

def calc_squeeze(df, atr):
    mean = df['close'].rolling(20).mean()
    sd   = df['close'].rolling(20).std()
    return (mean + 2*sd < mean + 1.5*atr) & (mean - 2*sd > mean - 1.5*atr)

def calc_adx(df, period=14):
    high, low = df['high'], df['low']
    up, down  = high.diff(), -low.diff()
    pDM = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    mDM = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_raw = calc_atr(df, period).replace(0, np.nan)
    pDI = 100 * (pDM.ewm(alpha=1/period, adjust=False).mean() / atr_raw)
    mDI = 100 * (mDM.ewm(alpha=1/period, adjust=False).mean() / atr_raw)
    dxDenom = (pDI + mDI).replace(0, np.nan)
    dx  = 100 * (pDI - mDI).abs() / dxDenom
    val = float(dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1])
    return round(val, 1) if not np.isnan(val) else 0.0

def _find_swings(highs, lows, lookback):
    sh, sl = [], []
    n = len(highs)
    for i in range(lookback, n - lookback):
        lh = highs[i-lookback:i]; rh = highs[i+1:i+lookback+1]
        ll = lows[i-lookback:i];  rl = lows[i+1:i+lookback+1]
        if len(lh)==lookback and len(rh)==lookback and highs[i]>=max(lh) and highs[i]>=max(rh):
            sh.append(float(highs[i]))
        if len(ll)==lookback and len(rl)==lookback and lows[i]<=min(ll) and lows[i]<=min(rl):
            sl.append(float(lows[i]))
    return sh, sl

def calc_swing_levels(df, lookback=10):
    highs, lows = df['high'].values, df['low'].values
    for lb in range(lookback, 2, -1):
        sh, sl = _find_swings(highs, lows, lb)
        if sh and sl:
            return {'swing_high': sh[-1], 'swing_low': sl[-1]}
    return {
        'swing_high': float(df['high'].rolling(20).max().iloc[-1]),
        'swing_low':  float(df['low'].rolling(20).min().iloc[-1])
    }

def is_weekend_blackout():
    if not WEEKEND_BLACKOUT: return False
    now = datetime.utcnow()
    overflow = (now.hour + 7) >= 24
    wd = (now.weekday() + (1 if overflow else 0)) % 7
    hh = (now.hour + 7) % 24
    if wd == 4 and hh >= 20: return True
    if wd in (5, 6):          return True
    if wd == 0 and hh < 9:   return True
    return False

def compute_rvol(ticker):
    vol  = float(ticker.get('quoteVolume') or 0)
    chg  = abs(float(ticker.get('percentage') or ticker.get('change') or 0))
    last = float(ticker.get('last') or 1e-9)
    return (vol * max(chg, 0.1)) / last

# ── SMC DETECTORS ─────────────────────────────────────────────
def detect_fvg_simple(df, lookback=20):
    if len(df) < 5: return False, False
    atr = calc_atr(df).iloc[-1]
    if atr == 0: return False, False
    min_size = atr * 0.3
    for i in range(2, min(lookback+1, len(df))):
        gap_top = df['low'].iloc[i-2]
        gap_bot = df['low'].iloc[i]
        if gap_top > gap_bot and (gap_top - gap_bot) >= min_size:
            mid = df['close'].iloc[i-1]
            if not (mid <= gap_top and mid >= gap_bot): return True, False
        gap_top2 = df['high'].iloc[i]
        gap_bot2 = df['high'].iloc[i-2]
        if gap_top2 < gap_bot2 and (gap_bot2 - gap_top2) >= min_size:
            mid = df['close'].iloc[i-1]
            if not (mid >= gap_top2 and mid <= gap_bot2): return False, True
    return False, False

def detect_ob_simple(df, lookback=10):
    if len(df) < 5: return False, False
    atr = calc_atr(df).iloc[-1]
    if atr == 0: return False, False
    vol_avg = df['volume'].rolling(20).mean().iloc[-1]
    if vol_avg == 0: return False, False
    for i in range(1, min(lookback+1, len(df)-1)):
        imp    = abs(df['close'].iloc[i-1] - df['close'].iloc[i])
        vol_ok = df['volume'].iloc[i-1] > vol_avg * 1.5
        if imp > atr * 2.0 and vol_ok:
            if df['open'].iloc[i] > df['close'].iloc[i] and df['close'].iloc[i-1] > df['open'].iloc[i-1]:
                if df['close'].iloc[-1] > df['open'].iloc[i]: return True, False
            if df['open'].iloc[i] < df['close'].iloc[i] and df['close'].iloc[i-1] < df['open'].iloc[i-1]:
                if df['close'].iloc[-1] < df['open'].iloc[i]: return False, True
    return False, False

# ── DATA PIPELINE ─────────────────────────────────────────────
async def fetch_ohlcv_gate(session, symbol, tf, limit):
    """Gate.io v4 futures USDT kline — terbukti tidak diblokir GitHub Actions."""
    # Gate.io symbol format: BTC_USDT (underscore, bukan slash)
    gate_sym = symbol.replace('/USDT:USDT', '').replace('/', '') + '_USDT'
    interval = GATE_TF_MAP.get(tf, '15m')
    params   = {"contract": gate_sym, "interval": interval, "limit": str(limit)}
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(GATE_KLINE_URL, params=params,
                                   timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    rows = await resp.json()
                    if rows and len(rows) >= CANDLES_REQUIRED:
                        df = pd.DataFrame(rows)
                        # Gate.io v4 kline fields: t=timestamp, o=open, h=high, l=low, c=close, v=volume
                        df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high',
                                                'l': 'low', 'c': 'close', 'v': 'volume'})
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    return None

async def fetch_ohlcv_bybit(session, symbol, tf, limit):
    """Bybit kline — fallback jika Gate.io gagal."""
    bybit_sym = symbol.replace('/USDT:USDT', '').replace('/', '').upper() + 'USDT'
    interval  = BYBIT_TF_MAP.get(tf, '15')
    params    = {"category": "linear", "symbol": bybit_sym,
                 "interval": interval, "limit": str(limit)}
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(BYBIT_KLINE_URL, params=params,
                                   timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    rows = data.get('result', {}).get('list', [])
                    if rows and len(rows) >= CANDLES_REQUIRED:
                        rows = list(reversed(rows))
                        df = pd.DataFrame(rows,
                            columns=['timestamp','open','high','low','close','volume','turnover'])
                        for col in ['open','high','low','close','volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        return df[['timestamp','open','high','low','close','volume']]
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    return None

async def fetch_ohlcv_safe(session, symbol, tf, limit):
    """Bybit kline primary (confirmed working from GitHub Actions IP),
       Gate.io kline fallback."""
    df = await fetch_ohlcv_bybit(session, symbol, tf, limit)
    if df is not None:
        return df
    return await fetch_ohlcv_gate(session, symbol, tf, limit)

def build_indicators(df):
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
    true_rvol = round(float(v.iloc[-1] / max(vm, 1e-9)), 2)
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
        'true_rvol':    true_rvol,
        '_df':          df
    }

async def phase1_scan(session, coin):
    df = await fetch_ohlcv_safe(session, coin['symbol'], '15m', CANDLES_REQUIRED)
    if df is None:
        logging.debug(f"  SKIP {coin['symbol']} — candle fetch gagal")
        return None
    logging.debug(f"  OK   {coin['symbol']} — {len(df)} candles")
    i = build_indicators(df)
    return {
        'symbol':         coin['symbol'],
        'Symbol':         coin['symbol'].split(':')[0],
        'Price':          i['close'],
        'power_15m':      i['power_score'],
        'RSI_15m':        i['rsi'],
        'ATR_15m':        i['atr'],
        'Squeeze_15m':    i['is_squeezing'],
        'Swing_High_15m': i['swing_high'],
        'Swing_Low_15m':  i['swing_low'],
        'ADX_15m':        i['adx'],
        'rvol_score':     coin.get('_rvol', 0.0),
        '_df_15m':        i['_df']
    }

async def phase2_enrich(session, c):
    await asyncio.sleep(PHASE2_DELAY)
    df_1h, df_5m = await asyncio.gather(
        fetch_ohlcv_safe(session, c['symbol'], '1h', CANDLES_REQUIRED),
        fetch_ohlcv_safe(session, c['symbol'], '5m', CANDLES_REQUIRED)
    )
    if df_1h is not None:
        i1    = build_indicators(df_1h)
        p     = i1['close']
        trend = ("UPTREND"   if p > i1['ema_f'] > i1['ema_s'] else
                 "DOWNTREND" if p < i1['ema_f'] < i1['ema_s'] else "RANGING")
        c.update({
            'Trend_1h':  trend, 'power_1h': i1['power_score'], 'ADX_1h': i1['adx'],
            'rvol_score': i1['true_rvol'], 'True_RVOL': i1['true_rvol'],
            '_close_1h': df_1h['close'].values[-CORR_WINDOW:].tolist(), '_df_1h': i1['_df']
        })
    else:
        c.update({'Trend_1h': 'N/A', 'power_1h': c['power_15m'], 'ADX_1h': 0.0,
                  'True_RVOL': c.get('rvol_score', 0.0), '_close_1h': [], '_df_1h': None})
    if df_5m is not None:
        i5  = build_indicators(df_5m)
        z   = i5['z_score']
        sqz = i5['is_squeezing']
        sm_sig = ("💥NUC+SQZ" if z > 3.0 and sqz else "🐳NUCLEAR" if z > 3.0 else
                  "🔥SQUEEZE" if sqz else "👀ACTIVE" if z > 1.5 else "😴QUIET")
        c.update({'z_score_5m': z, 'SM_Signal': sm_sig, 'power_5m': i5['power_score']})
    else:
        c.update({'z_score_5m': 0.0, 'SM_Signal': '😴QUIET', 'power_5m': c['power_15m']})
    return c

def finalize_screener(c):
    p15 = c['power_15m']
    p1h = c.get('power_1h', p15)
    p5m = c.get('power_5m', p15)
    comp = (p1h * 0.40) + (p15 * 0.40) + (p5m * 0.20)
    trend  = c.get('Trend_1h', 'RANGING')
    td_dir = 1 if trend == 'UPTREND' else -1 if trend == 'DOWNTREND' else 0
    if td_dir != 0 and (1 if p15 > 0 else -1) != td_dir: comp -= 15
    sm_lvl    = c.get('SM_Signal', '😴QUIET')
    is_nuclear = 'NUC' in sm_lvl
    sm_bonus  = (23 if 'NUC+SQZ' in sm_lvl else 15 if is_nuclear else
                  7 if 'ACTIVE' in sm_lvl else 0)
    if sm_bonus > 0: comp += sm_bonus if comp > 0 else -sm_bonus
    # FIX: parenthesis benar (versi AI lain missing closing paren)
    aligned = ((p5m > 0 and p15 > 0 and p1h > 0) or (p5m < 0 and p15 < 0 and p1h < 0))
    if aligned: comp += 5 if comp > 0 else -5
    adx_val  = max(c.get('ADX_1h', 0.0), c.get('ADX_15m', 0.0))
    is_side  = adx_val < (ADX_MIN_NUCLEAR if is_nuclear else ADX_MIN_NORMAL) and not is_nuclear
    is_cancel = ((trend == 'DOWNTREND' and comp > 0) or
                 (trend == 'UPTREND' and comp < 0) or is_side)
    c.update({
        'Composite':   round(comp, 1), 'Aligned': aligned, 'ADX': adx_val,
        'Sideways':    is_side, 'Is_Cancel': is_cancel,
        'Matrix_Sync': ("FULL BULL" if comp >= 40 else "BULLISH" if comp > 10 else
                        "FULL BEAR" if comp <= -40 else "BEARISH" if comp < -10 else "NEUTRAL")
    })
    is_long = comp > 0
    _df_1h  = c.get('_df_1h')
    _df_15m = c.get('_df_15m')
    df_smc  = None
    if _df_1h is not None and not _df_1h.empty:    df_smc = _df_1h
    elif _df_15m is not None and not _df_15m.empty: df_smc = _df_15m
    has_fvg_bull, has_fvg_bear = (detect_fvg_simple(df_smc) if df_smc is not None else (False, False))
    has_ob_bull,  has_ob_bear  = (detect_ob_simple(df_smc)  if df_smc is not None else (False, False))
    has_fvg = (has_fvg_bull and is_long) or (has_fvg_bear and not is_long)
    has_ob  = (has_ob_bull  and is_long) or (has_ob_bear  and not is_long)
    conv = 0
    if has_fvg:          conv += 20
    if aligned:          conv += 12
    if is_nuclear:       conv += 15
    if adx_val > 25:     conv += 10
    elif adx_val < 18:   conv -= 30
    if c['Squeeze_15m']: conv += 8
    if has_ob:           conv += 8
    conv = max(0, min(100, conv))
    c['Conviction'] = conv
    c['Checklist']  = (int(has_fvg) + int(aligned) + int(not is_side) +
                       int(conv >= CONV_VALID) + int(has_ob))
    if is_cancel or conv < CONV_VALID:             c['Tier'] = "REJECT"
    elif conv >= CONV_INST and c['Checklist'] >= 5: c['Tier'] = "INSTITUTIONAL"
    elif conv >= CONV_VALID and c['Checklist'] >= 4: c['Tier'] = "VALID"
    else:                                            c['Tier'] = "WEAK"
    return c

def filter_by_correlation(candidates):
    if len(candidates) <= 1: return candidates
    kept, series = [], {}
    for c in candidates:
        closes = c.get('_close_1h', [])
        if len(closes) >= 10: series[c['symbol']] = np.array(closes[-CORR_WINDOW:], dtype=float)
    for c in candidates:
        sym, too_corr = c['symbol'], False
        if sym in series:
            for k in kept:
                if k['symbol'] not in series: continue
                s1, s2 = series[sym], series[k['symbol']]
                n = min(len(s1), len(s2))
                if n >= 10:
                    corr = float(np.corrcoef(s1[-n:], s2[-n:])[0, 1])
                    if not np.isnan(corr) and corr > CORR_THRESHOLD:
                        too_corr = True; break
        if not too_corr: kept.append(c)
        if len(kept) >= 8: break
    return kept

async def get_screener_data():
    # HYBRID ANTI-BLOKIR:
    # 1. Gate.io v4 API → tickers futures USDT (tidak pernah blokir GitHub IP)
    # 2. Gate.io v4 API → OHLCV kline (primary)
    # 3. Bybit kline → OHLCV fallback jika Gate.io gagal per-coin
    logging.info("Mengambil tickers dari Gate.io v4...")
    try:
        async with aiohttp.ClientSession() as session:
            # Gate.io v4 futures tickers
            async with session.get(GATE_TICKER_URL,
                                   timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    raise Exception(f"Gate.io HTTP {resp.status}")
                data = await resp.json()

        if not data:
            raise Exception("Empty ticker list dari Gate.io")

        all_tickers = []
        for item in data:
            contract = item.get('contract', '')
            if not contract.endswith('_USDT'):
                continue
            base = contract.replace('_USDT', '')
            vol  = float(item.get('volume_24h_quote', 0) or item.get('volume_24h', 0) or 0)
            last = float(item.get('last', 0) or 0)
            chg  = abs(float(item.get('change_percentage', 0) or 0))
            if vol <= 0 or last <= 0:
                continue
            rvol_score = (vol * max(chg, 0.1)) / max(last, 1e-9)
            all_tickers.append({
                'symbol':      f"{base}/USDT:USDT",
                'last':        last,
                'quoteVolume': vol,
                'percentage':  chg,
                '_rvol':       rvol_score
            })

        all_vols        = [t['quoteVolume'] for t in all_tickers if t['quoteVolume'] > 0]
        dynamic_min_vol = (max(5_000_000, np.percentile(all_vols, 70) * 0.5)
                           if all_vols else 7_000_000)
        liquid = [t for t in all_tickers
                  if t['quoteVolume'] >= dynamic_min_vol and t['last'] > 0]
        top    = sorted(liquid, key=lambda x: x['_rvol'], reverse=True)[:TOP_COINS_BY_RVOL]

        logging.info(f"Gate.io: {len(all_tickers)} contracts → Top {len(top)} liquid. Fetch OHLCV...")

        async with aiohttp.ClientSession() as session:
            # Phase 1
            sem1 = asyncio.Semaphore(SEMAPHORE_P1)
            async def sp1(coin):
                async with sem1: return await phase1_scan(session, coin)
            p1    = await asyncio.gather(*[sp1(c) for c in top])
            cands = sorted([r for r in p1 if r],
                           key=lambda x: abs(x['power_15m']), reverse=True)[:RANKED_CANDIDATES]

            if not cands:
                logging.warning("Tidak ada candle berhasil di-fetch.")
                return []

            # Phase 2
            sem2 = asyncio.Semaphore(SEMAPHORE_P2)
            async def sp2(c):
                async with sem2: return await phase2_enrich(session, c)
            enriched  = await asyncio.gather(*[sp2(c) for c in cands])

        finalized = sorted([finalize_screener(c) for c in enriched],
                           key=lambda x: x['Conviction'], reverse=True)
        return filter_by_correlation(finalized)

    except Exception as e:
        logging.error(f"Gagal fetch data: {e}")
        return []

# ── TELEGRAM ──────────────────────────────────────────────────
def build_screener_message(data_list):
    valid_coins = [d for d in data_list if d['Tier'] != "REJECT"]
    if not valid_coins:
        return ("😴 *SCREENER RESULTS*\n"
                "Semua koin filter out (Sideways/Counter-trend).\n"
                "Market sedang tidak bersahabat untuk intraday.")
    msg = ("🔬 *SCREENER RESULTS (Buka chart & cocokkan Pine v7.1)*\n"
           "Prioritaskan koin berlabel INST/VALID dengan Conviction tinggi.\n\n")
    for i, d in enumerate(valid_coins[:5], 1):
        direction  = "🟢 LONG"  if d['Composite'] > 0 else "🔴 SHORT"
        aln_badge  = "✅3TF"    if d['Aligned']        else "⚡2TF"
        rvol       = d.get('True_RVOL', 0)
        rvol_badge = f" 📈{rvol:.1f}x" if rvol >= 1.5 else ""
        tier       = d['Tier']
        tier_emoji = "💎" if tier == "INSTITUTIONAL" else "✅" if tier == "VALID" else "⚠️"
        tier_text  = (f"{tier_emoji} *[{tier}]*"
                      if tier in ("INSTITUTIONAL", "VALID") else f"{tier_emoji} {tier}")
        msg += (f"{i}. {d['Symbol']} — {direction} `{d['Matrix_Sync']}` {tier_text}\n"
                f"   🧠 Conviction: `{d['Conviction']}/100` | Checklist: `{d['Checklist']}/7`\n"
                f"   SM: {d['SM_Signal']} | ADX: `{d['ADX']}` {aln_badge}{rvol_badge}\n\n")
    return msg

async def send_telegram(text):
    if not all([TG_TOKEN, TG_CHAT_ID]): return
    url    = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            payload = {"chat_id": TG_CHAT_ID, "text": chunk, "parse_mode": "Markdown"}
            try:
                async with session.post(url, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 400:
                        payload.pop("parse_mode", None)
                        await session.post(url, json=payload)
            except Exception as e:
                logging.error(f"TG Error: {e}")

# ── MAIN ──────────────────────────────────────────────────────
async def main():
    ts = datetime.now().strftime("%d %b %Y, %H:%M WIB")
    if is_weekend_blackout():
        await send_telegram(f"😴 *SCREENER BLACKOUT*\n🕐 {ts}\n⛔ Weekend — Bot istirahat.")
        return
    logging.info("🚀 Screener v7.1 dimulai...")
    data = await get_screener_data()
    if not data:
        await send_telegram("⚠️ Gagal fetch data dari Gate.io & Bybit.")
        return
    n_inst  = sum(1 for d in data if d['Tier'] == "INSTITUTIONAL")
    n_valid = sum(1 for d in data if d['Tier'] == "VALID")
    header  = (f"👁️ *GOD MODE SCREENER v7.1*\n"
               f"🕐 {ts} | Gate.io Ticker + Gate/Bybit Kline\n"
               f"📊 Scanned: {len(data)} | 💎 INST: {n_inst} | ✅ VALID: {n_valid}\n"
               f"{'─' * 35}\n\n")
    await send_telegram(header + build_screener_message(data))
    logging.info(f"✅ Selesai. {n_inst} INST, {n_valid} VALID.")

if __name__ == "__main__":
    asyncio.run(main())
