import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def get_market_data():
    # Menggunakan Gate.io Swap (Futures)
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        # Ambil 15 koin dengan volume transaksi tertinggi (Uang paling banyak di sini)
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
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Kamu adalah Senior Trader Crypto. Berikut adalah data 15 koin dengan volume terbesar saat ini:
    {data_list}
    
    Tugasmu:
    1. Analisis koin mana yang menunjukkan momentum paling kuat (bullish) atau paling lemah (bearish).
    2. Pilih HANYA 1 koin terbaik untuk peluang trading (bisa Long atau Short).
    3. Berikan Trading Plan lengkap:
       - Alasan Teknis
       - Entry Area
       - Target Profit (TP)
       - Stop Loss (SL)
    4. Jika pasar benar-benar jelek, katakan 'Wait & See' tapi tetap berikan 1 koin pantauan.
    Gunakan bahasa Indonesia yang santai tapi profesional. Gunakan Markdown.
    """
    response = model.generate_content(prompt)
    return response.text

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
