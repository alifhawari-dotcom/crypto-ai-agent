import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt
import requests
import json

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def get_coin_metrics(exchange, coin):
    symbol = coin['symbol']
    try:
        # Mengambil OI dan Funding secara paralel (Async)
        oi_task = exchange.fetch_open_interest(symbol)
        funding_task = exchange.fetch_funding_rate(symbol)
        
        oi_info, funding = await asyncio.gather(oi_task, funding_task)
        
        oi_value = float(oi_info['baseVolume']) # Gate.io menggunakan baseVolume
        funding_rate = funding['fundingRate'] * 100
        
        price_change = coin['percentage']
        current_price = coin['last']
        high_24h = coin['high']
        
        # Logika Screening (Sesuai keinginanmu)
        if price_change > 0.5 and funding_rate < 0.03:
            if current_price >= (high_24h * 0.95):
                distance_to_high = ((high_24h - current_price) / current_price) * 100
                return {
                    'symbol': symbol.split(':')[0],
                    'price': current_price,
                    'change_24h': f"{price_change:.2f}%",
                    'open_interest': f"{oi_value:,.0f}",
                    'funding_rate': f"{funding_rate:.4f}%",
                    'distance_to_high': f"{distance_to_high:.2f}%"
                }
    except:
        return None

async def get_high_precision_data():
    # Gunakan Gate.io agar TIDAK kena blokir lokasi (451 error)
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:20]
        
        tasks = [get_coin_metrics(exchange, coin) for coin in top_coins]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    finally:
        await exchange.close()

def ask_ai_agent(data_list):
    if not data_list:
        return "📉 Pasar sedang tenang. Belum ada koin yang masuk radar breakout."

    # Gunakan Model 2.5 Flash lewat Direct API (Anti Error 404)
    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_KEY}"
    
    prompt = f"Sebagai Crypto Analyst, analisis data koin ini: {data_list}. Pilih 1 koin terbaik, jelaskan alasan teknisnya, lalu berikan Trading Plan (Entry, SL, TP). Gunakan Bahasa Indonesia."
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        res_json = response.json()
        return res_json['candidates'][0]['content']['parts'][0]['text']
    except:
        return "⚠️ Gagal mendapatkan analisis dari AI."

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🚀 **AI TRADING SIGNAL**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): return
    logging.info("Memulai scanning pasar...")
    try:
        data = await get_high_precision_data()
        analysis = ask_ai_agent(data)
        await send_to_telegram(analysis)
        logging.info("Selesai! Pesan terkirim.")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
