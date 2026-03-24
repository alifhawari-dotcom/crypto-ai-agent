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
        funding_info = await exchange.fetch_funding_rate(symbol)
        funding_rate = funding_info['fundingRate'] * 100
        oi_info = await exchange.fetch_open_interest(symbol)
        oi_value = float(oi_info['baseVolume'])
        
        price_change = coin['percentage']
        current_price = coin['last']
        high_24h = coin['high']
        
        # SYARAT SANGAT LONGGAR (Hanya untuk tes koneksi)
        # Jika koin naik > 0.1% dan jarak ke High 24h < 10%
        if price_change > 0.1 and funding_rate < 0.05:
            if current_price >= (high_24h * 0.90):
                distance_to_high = ((high_24h - current_price) / current_price) * 100
                return {
                    'symbol': symbol.split(':')[0],
                    'price': current_price,
                    'change_24h': f"{price_change:.2f}%",
                    'open_interest': f"{oi_value:,.0f}",
                    'funding_rate': f"{funding_rate:.4f}%",
                    'distance_to_high': f"{distance_to_high:.2f}%"
                }
    except Exception:
        return None

async def get_high_precision_data():
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        # Ambil Top 30 koin volume terbesar
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:30]
        tasks = [get_coin_metrics(exchange, coin) for coin in top_coins]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    finally:
        await exchange.close()

def ask_ai_agent(data_list):
    # Jika tidak ada koin, AI akan membuat pesan status tetap tenang
    if not data_list:
        return "Pasar sedang sangat tenang (sideways). Belum ada setup breakout yang valid untuk saat ini. Saya akan terus memantau setiap 20 menit."
        
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"Analisis data koin potensial ini: {data_list}. Pilih 1 yang terbaik, berikan alasan dan Trading Plan. Gunakan bahasa Indonesia yang santai."
    response = model.generate_content(prompt)
    return response.text

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    # Tambahkan status waktu agar kamu tahu bot sedang aktif
    payload = {"chat_id": TG_CHAT_ID, "text": f"🤖 **LAPORAN AI AGENT**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]):
        print("API KEYS MISSING!")
        return
        
    logging.info("Memulai pemindaian...")
    try:
        # 1. Kirim pesan "Start" ke Telegram sebagai tes koneksi pertama kali
        # await send_to_telegram("Sedang memindai pasar Gate.io... tunggu sebentar.")
        
        data = await get_high_precision_data()
        logging.info(f"Ditemukan {len(data)} koin.")
        
        # 2. Minta AI menganalisis (walaupun data kosong, AI akan lapor market sepi)
        analysis = ask_ai_agent(data)
        
        # 3. Kirim hasil analisis ke Telegram
        await send_to_telegram(analysis)
        logging.info("Laporan terkirim!")
        
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
