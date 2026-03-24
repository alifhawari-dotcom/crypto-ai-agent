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

async def get_market_data():
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
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
    # Kita langsung tembak API Google lewat jalur v1 (STABLE)
    # Ini jauh lebih aman dari error 404
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
    
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"Analisis data koin ini: {data_list}. Pilih 1 koin terbaik untuk trading, berikan alasan teknis, Entry, SL, dan TP. Gunakan Bahasa Indonesia."
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()
        
        # Ambil teks jawaban dari struktur JSON Google
        if 'candidates' in res_json:
            return res_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Waduh, Google bilang: {res_json.get('error', {}).get('message', 'Error tidak diketahui')}"
    except Exception as e:
        return f"Koneksi AI terputus: {e}"

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🚀 **AI TRADING SIGNAL**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): return
    logging.info("Memulai pemindaian pasar...")
    try:
        data = await get_market_data()
        analysis = ask_ai_agent(data)
        await send_to_telegram(analysis)
        logging.info("Sinyal berhasil dikirim!")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
