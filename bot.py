import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def get_god_mode_metrics(exchange, coin):
    symbol = coin['symbol']
    try:
        # TARIK DATA HISTORIS: 50 Candle terakhir di Timeframe 1 Jam
        ohlcv = await exchange.fetch_ohlcv(symbol, '1h', limit=50)
        if not ohlcv or len(ohlcv) < 50:
            return None
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # --- 1. CORE SCORING ENGINE ---
        v_range = df['high'] - df['low']
        df['n_delta'] = np.where(v_range == 0, 0, ((df['close'] - df['low']) - (df['high'] - df['close'])) / v_range * df['volume'])
        
        df['a_delta'] = df['n_delta'].abs().rolling(20).mean()
        df['s_flow'] = np.clip((df['n_delta'] / np.where(df['a_delta'] == 0, 1, df['a_delta'])) * 20, -40, 40)
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / np.where(loss == 0, 1, loss)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['s_mom'] = np.clip((df['rsi'] - 50) * 1.2, -30, 30)
        
        df['power_score'] = df['s_flow'] + df['s_mom']
        
        # --- 2. WHALE DETECTION ---
        vol_avg = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        df['z_score'] = (df['volume'] - vol_avg) / np.where(vol_std == 0, 1, vol_std)
        
        latest = df.iloc[-1]
        z_val = latest['z_score']
        p_score = latest['power_score']
        
        # STANDAR DITURUNKAN: Label diperhalus agar AI punya opsi
        w_txt = "NUCLEAR" if z_val > 3.0 else "ACTIVE" if z_val > 1.2 else "QUIET"
        sync_stat = "FULL BULL" if p_score >= 40 else "BULLISH" if p_score > 10 else "FULL BEAR" if p_score <= -40 else "BEARISH" if p_score < -10 else "NEUTRAL"

        # FILTER DIHAPUS: Semua data dikembalikan untuk diranking
        return {
            'Symbol': symbol.split(':')[0],
            'Price': latest['close'],
            'Matrix_Sync': sync_stat,
            'Power_Score': round(p_score, 1), # Angka float agar bisa disortir
            'Whale_Action': w_txt,
            'RSI_1H': round(latest['rsi'], 1),
            'Volume_Surge': round(z_val, 2)
        }
    except Exception as e:
        return None

async def get_high_precision_data():
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        # Ambil Top 40 berdasarkan Volume USDT
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:40]
        
        sem = asyncio.Semaphore(5)
        
        async def safe_get_metrics(coin):
            async with sem:
                return await get_god_mode_metrics(exchange, coin)
                
        tasks = [safe_get_metrics(coin) for coin in top_coins]
        results = await asyncio.gather(*tasks)
        valid_results = [r for r in results if r is not None]
        
        # SISTEM RANKING: Urutkan berdasarkan Power Score paling kuat (entah itu + atau -)
        # Ambil 15 koin teratas yang paling "bergejolak" untuk disuapkan ke AI
        sorted_results = sorted(valid_results, key=lambda x: abs(x['Power_Score']), reverse=True)[:15]
        
        return sorted_results
    finally:
        await exchange.close()

async def ask_ai_agent(data_list):
    if not data_list:
        return "⚠️ Kesalahan koneksi, gagal mengambil data pasar."
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    
    prompt = f"""
    Kamu adalah AI Trader Assistant. Berikut adalah 15 koin paling bergejolak di pasar saat ini berdasarkan algoritma "God Mode Matrix":
    {data_list}
    
    Tugasmu:
    1. WAJIB pilih TEPAT 3 KOIN TERBAIK dari daftar tersebut untuk dijadikan opsi trading.
    2. Jika tidak ada yang sempurna (Nuclear/Full Sync), pilihlah yang paling "mendingan" atau memiliki setup momentum terbaik (Power Score tertinggi atau terendah).
    3. Buat 3 list singkat dengan format Markdown.
    
    Format Wajib untuk masing-masing koin:
    ### 1. [Nama Koin] - [Aksi: LONG/SHORT]
    * **Alasan (1 kalimat):** (Sebutkan korelasi Matrix Sync, Power Score, dan Volume)
    * **Plan:** Entry: [Area] | SL: [Harga] | TP: [Harga]
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    return f"⚠️ API Error: {await response.text()}"
        except Exception as e:
            return f"⚠️ Kesalahan AI: {e}"

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID,
