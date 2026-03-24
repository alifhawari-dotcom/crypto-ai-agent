import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt
from google import genai # Library standar 2026

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def get_market_data():
    # Mengambil data dari Gate.io (Bursa paling aman dari blokir IP GitHub)
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        # Ambil 15 koin dengan volume tertinggi
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:15]
        summary = []
        for c in top_coins:
            summary.append({
                'koin': c['symbol'].split(':')[0],
                'harga': c['last'],
                'change': f"{c['percentage']}%",
                'vol': f"{c['quoteVolume']:,.0f} USDT"
            })
        return summary
    finally:
        await exchange.close()

def ask_ai_agent(data_list):
    # Menggunakan Client baru versi 2026
    client = genai.Client(api_key=GEMINI_KEY)
    
    prompt = f"Analisis data koin top volume ini: {data_list}. Pilih 1 koin terbaik (Long/Short), berikan alasan teknis, dan Trading Plan (Entry, SL, TP). Gunakan bahasa Indonesia santai."
    
    try:
        # Perintah baru untuk generate content
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Waduh, AI lagi pusing: {e}"

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🤖 **HASIL SCANNING AI**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): return
    logging.info("Memulai pemindaian...")
    try:
        data = await get_market_data()
        analysis = ask_ai_agent(data)
        await send_to_telegram(analysis)
        logging.info("Sukses!")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
