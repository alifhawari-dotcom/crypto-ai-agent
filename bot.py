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
    symbol = coin['symbol'] # Format Gate.io: BTC_USDT:USDT
    try:
        # 1. Ambil Funding Rate
        funding_info = await exchange.fetch_funding_rate(symbol)
        funding_rate = funding_info['fundingRate'] * 100
        
        # 2. Ambil Open Interest (OI)
        oi_info = await exchange.fetch_open_interest(symbol)
        oi_value = float(oi_info['baseVolume']) # Gate.io menggunakan baseVolume untuk OI
        
        price_change = coin['percentage']
        current_price = coin['last']
        high_24h = coin['high']
        
        # Logika Screening (OI Naik, Funding Sehat, Dekat Breakout)
        if price_change > 1.2 and funding_rate < 0.02:
            if current_price >= (high_24h * 0.975): # Toleransi 2.5% dari harga tertinggi
                distance_to_high = ((high_24h - current_price) / current_price) * 100
                return {
                    'symbol': symbol.split(':')[0], # Bersihkan nama koin
                    'price': current_price,
                    'change_24h': f"{price_change:.2f}%",
                    'open_interest': f"{oi_value:,.2f}",
                    'funding_rate': f"{funding_rate:.4f}%",
                    'distance_to_high': f"Minus {distance_to_high:.2f}% ke Resistance"
                }
    except Exception:
        return None

async def get_high_precision_data():
    # Menggunakan GATE.IO yang lebih bersahabat dengan IP Cloud
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        # Pilih koin dengan volume USDT terbesar
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:25]
        
        tasks = [get_coin_metrics(exchange, coin) for coin in top_coins]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    finally:
        await exchange.close()

def ask_ai_agent(data_list):
    if not data_list:
        return "📉 Pasar sedang konsolidasi. Belum ada koin di Gate.io yang memenuhi kriteria breakout."
        
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Kamu adalah Crypto Quant Analyst. Analisis data Gate.io Futures ini:
    {data_list}
    
    Tugas:
    1. Pilih 1 koin terbaik untuk LONG (Breakout Setup).
    2. Berikan Trading Plan: Entry Area, SL, dan TP.
    3. Berikan skor keyakinan 1-10.
    4. Gunakan Markdown rapi.
    """
    response = model.generate_content(prompt)
    return response.text

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🚀 **GATE.IO AI SCANNER**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]):
        logging.error("API Keys belum di-setup di Secrets!")
        return
        
    logging.info("Memulai pemindaian di Gate.io...")
    try:
        data = await get_high_precision_data()
        logging.info(f"Ditemukan {len(data)} koin potensial.")
        analysis = ask_ai_agent(data)
        await send_to_telegram(analysis)
    except Exception as e:
        logging.error(f"Sistem Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
