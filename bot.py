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
            'Symbol': symbol.
