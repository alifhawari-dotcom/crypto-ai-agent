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
            
        # Ubah ke Pandas DataFrame untuk perhitungan matematika ala TradingView
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # --- 1. CORE SCORING ENGINE (Pine Script Translation) ---
        
        # Net Delta Volume (Flow)
        v_range = df['high'] - df['low']
        # Mencegah error pembagian dengan 0
        df['n_delta'] = np.where(v_range == 0, 0, ((df['close'] - df['low']) - (df['high'] - df['close'])) / v_range * df['volume'])
        
        # Flow Score (SMA 20)
        df['a_delta'] = df['n_delta'].abs().rolling(20).mean()
        df['s_flow'] = np.clip((df['n_delta'] / np.where(df['a_delta'] == 0, 1, df['a_delta'])) * 20, -40, 40)
        
        # Momentum Score (RSI 14 ala TradingView)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / np.where(loss == 0, 1, loss)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['s_mom'] = np.clip((df['rsi'] - 50) * 1.2, -30, 30)
        
        # Total Power Score (Flow + Momentum)
        df['power_score'] = df['s_flow'] + df['s_mom']
        
        # --- 2. WHALE DETECTION (Z-SCORE) ---
        vol_avg = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        df['z_score'] = (df['volume'] - vol_avg) / np.where(vol_std == 0, 1, vol_std)
        
        # AMBIL DATA CANDLE TERAKHIR
        latest = df.iloc[-1]
        
        # Terjemahan HUD Visual
        z_val = latest['z_score']
        w_txt = "NUCLEAR" if z_val > 3.5 else "ACTIVE" if z_val > 2.0 else "QUIET"
        
        p_score = latest['power_score']
        sync_stat = "FULL BULL" if p_score >= 40 else "FULL BEAR" if p_score <= -40 else "NEUTRAL"
        
        # Hanya kirim koin yang ada pergerakan Whale (Active/Nuclear) ATAU ada Full Sync
        if w_txt == "QUIET" and sync_stat == "NEUTRAL":
            return None 

        return {
            'Symbol': symbol.split(':')[0],
            'Price': latest['close'],
            'Matrix_Sync': sync_stat,
            'Power_Score': f"{p_score:.1f}/100",
            'Whale_Action': w_txt,
            'RSI_1H': f"{latest['rsi']:.1f}",
            'Volume_Surge': f"{z_val:.2f}x StdDev"
        }
    except Exception as e:
        return None

async def get_high_precision_data():
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        # Filter Top 40 Koin teraktif
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:40]
        
        sem = asyncio.Semaphore(5)
        
        async def safe_get_metrics(coin):
            async with sem:
                return await get_god_mode_metrics(exchange, coin)
                
        tasks = [safe_get_metrics(coin) for coin in top_coins]
        results = await asyncio.gather(*tasks)
        
        # Saring hasil
        return [r for r in results if r is not None]
    finally:
        await exchange.close()

async def ask_ai_agent(data_list):
    if not data_list:
        return "⚠️ Pasar sedang tenang. Tidak ada 'Whale Action' atau 'Matrix Sync' yang terdeteksi saat ini."
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    
    prompt = f"""
    Kamu adalah AI Sniper Trading untuk institusi. Berikut adalah data hasil penyaringan algoritma "God Mode Matrix" (RSI, Whale Z-Score, Power Score):
    {data_list}
    
    Tugasmu:
    1. Analisis data tersebut dan pilih TEPAT 3 KOIN TERBAIK yang memiliki tingkat akurasi tertinggi untuk dieksekusi sekarang.
    2. Prioritaskan koin dengan status 'FULL BULL / FULL BEAR' dan Whale Action 'NUCLEAR / ACTIVE'.
    3. Buat 3 list singkat dengan format Markdown yang rapi.
    
    Format Wajib untuk masing-masing koin:
    ### 1. [Nama Koin] - [Aksi: LONG/SHORT]
    * **Alasan (1 kalimat):** (Sebutkan korelasi Power Score dan Whale Action)
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
    payload = {"chat_id": TG_CHAT_ID, "text": f"👁️ **GOD MODE MATRIX: SNIPER REPORT**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): 
        logging.error("API Keys belum lengkap!")
        return
        
    logging.info("Memulai pemindaian God Mode Matrix (Candle 1H)...")
    try:
        data = await get_high_precision_data()
        logging.info(f"Ditemukan {len(data)} koin yang masuk radar Whale/Sync.")
        
        analysis = await ask_ai_agent(data) 
        await send_to_telegram(analysis)
        logging.info("Laporan Sniper sukses dikirim ke Telegram!")
    except Exception as e:
        logging.error(f"Sistem Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
