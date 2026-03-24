import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt
from google import genai # Import library terbaru 2026

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def get_market_data():
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:15]
        market_summary = []
        for coin in top_coins:
            market_summary.append({
                'koin': coin['symbol'].split(':')[0],
                'harga': coin['last'],
                'perubahan_24j': f"{coin['percentage']}%",
                'vol_24j': f"{coin['quoteVolume']:,.0f} USDT"
            })
        return market_summary
    finally:
        await exchange.close()

def ask_ai_agent(data_list):
    # Inisialisasi Client Gemini versi terbaru
    client = genai.Client(api_key=GEMINI_KEY)
    
    prompt = f"""
    Kamu adalah Senior Trader Crypto. Analisis koin top volume ini: {data_list}. 
    Pilih 1 koin terbaik (Long/Short), berikan alasan teknis, dan Trading Plan (Entry, SL, TP). 
    Gunakan bahasa Indonesia santai.
    """
    
    try:
        # Cara panggil model di library terbaru
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI sedang maintenance: {e}"

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🤖 **HASIL SCANNING AI**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): return
    logging.info("Memulai scanning koin top volume...")
    try:
        data = await get_market_data()
        analysis = ask_ai_agent(data)
        await send_to_telegram(analysis)
        logging.info("Laporan dikirim!")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
