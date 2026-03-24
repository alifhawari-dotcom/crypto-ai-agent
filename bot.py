import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt
import requests
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def get_coin_metrics(exchange, coin):
    symbol = coin['symbol']
    try:
        # Kita ambil OI dan Funding sebagai data pendukung untuk AI
        oi_info = await exchange.fetch_open_interest(symbol)
        funding = await exchange.fetch_funding_rate(symbol)
        
        return {
            'symbol': symbol.split(':')[0],
            'price': coin['last'],
            'change_24h': f"{coin['percentage']:.2f}%",
            'vol_usdt': f"{coin['quoteVolume']:,.0f}",
            'oi': f"{float(oi_info['baseVolume']):,.0f}",
            'funding': f"{funding['fundingRate'] * 100:.4f}%",
            'high_24h': coin['high']
        }
    except:
        return None

async def get_high_precision_data():
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        
        # KITA AMBIL TOP 30 KOIN DENGAN VOLUME TERBESAR
        # Di sinilah uang berkumpul, pasti ada peluang trading.
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:30]
        
        tasks = [get_coin_metrics(exchange, coin) for coin in top_coins]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    finally:
        await asyncio.close_all([exchange]) # Memastikan koneksi ditutup rapi

def ask_ai_agent(data_list):
    # Jika data ada, kita paksa AI untuk memberikan analisis terbaiknya
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    
    prompt = f"""
    Kamu adalah Senior Crypto Strategist. Analisis data 30 koin paling aktif ini:
    {data_list}
    
    Tugasmu:
    1. Dari 30 koin ini, pilih 1 koin yang paling menjanjikan untuk Open Posisi SEKARANG.
    2. Kamu boleh memilih LONG (jika koin kuat) atau SHORT (jika koin sudah overbought/lemah).
    3. Jelaskan analisismu secara singkat (lihat korelasi antara Harga, Volume, dan Funding Rate).
    4. Berikan Trading Plan:
       - 🎯 Aksi: (LONG / SHORT)
       - 🟢 Entry Area
       - 🔴 Stop Loss
       - 🏁 Take Profit
    Gunakan Bahasa Indonesia yang tajam dan to-the-point. Format Markdown rapi.
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        return "⚠️ AI sedang kewalahan menganalisis pasar yang ramai. Coba cek beberapa saat lagi."

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🔥 **AI MARKET RADAR (TOP 30)**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): return
    logging.info("Memulai pemindaian 30 koin teraktif...")
    try:
        data = await get_high_precision_data()
        logging.info(f"Berhasil mengumpulkan data {len(data)} koin.")
        analysis = ask_ai_agent(data)
        await send_to_telegram(analysis)
        logging.info("Laporan dikirim!")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(main())
