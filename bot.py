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

async def get_coin_metrics(exchange, coin):
    symbol = coin['symbol']
    try:
        # Bybit menggunakan format yang sedikit berbeda tapi ccxt menyamakannya
        funding_info = await exchange.fetch_funding_rate(symbol)
        
        # Ambil Open Interest khusus untuk Bybit
        oi_info = await exchange.fetch_open_interest(symbol)
        
        oi_value = float(oi_info['openInterestAmount'])
        funding_rate = funding_info['fundingRate'] * 100
        
        price_change = coin['percentage']
        current_price = coin['last']
        high_24h = coin['high']
        
        # Logika tetap sama: OI Naik, Funding Sehat, Dekat High 24h
        if price_change > 1.5 and funding_rate < 0.015:
            if current_price >= (high_24h * 0.98):
                distance_to_high = ((high_24h - current_price) / current_price) * 100
                return {
                    'symbol': symbol,
                    'price': current_price,
                    'change_24h': f"{price_change:.2f}%",
                    'open_interest': f"{oi_value:,.0f}",
                    'funding_rate': f"{funding_rate:.4f}%",
                    'distance_to_high': f"Minus {distance_to_high:.2f}% ke Resistance"
                }
    except Exception as e:
        return None

async def get_high_precision_data():
    # GANTI KE BYBIT agar tidak kena blokir lokasi
    exchange = ccxt.bybit({'options': {'defaultType': 'linear'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        # Filter top 20 koin berdasarkan volume di Bybit
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:20]
        
        tasks = [get_coin_metrics(exchange, coin) for coin in top_coins]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    finally:
        await exchange.close()

def ask_ai_agent(data_list):
    if not data_list:
        return "📉 Belum ada koin di Bybit yang memenuhi kriteria breakout saat ini."
        
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Kamu adalah Crypto Quantitative Analyst. Analisis data Bybit Futures berikut:
    {data_list}
    
    Tugas:
    1. Pilih 1 koin terbaik untuk LONG.
    2. Berikan Trading Plan: Entry Area, SL, dan TP.
    3. Format Markdown rapi.
    """
    response = model.generate_content(prompt)
    return response.text

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🚀 **BYBIT AI SCANNER REPORT**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]):
        return
    logging.info("Memulai pemindaian pasar Bybit...")
    try:
        data = await get_high_precision_data()
        analysis = ask_ai_agent(data)
        await send_to_telegram(analysis)
        logging.info("Sukses mengirim ke Telegram!")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
