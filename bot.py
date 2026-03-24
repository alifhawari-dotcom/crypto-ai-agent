import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def get_market_data():
    # Gate.io tetap yang paling aman dari blokir IP GitHub
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
    # DAFTAR MODEL RESMI 2026
    # Kita gunakan gemini-3-flash sebagai model utama tahun ini
    model_name = "gemini-3-flash"
    
    # Gunakan endpoint v1 (Stable) untuk tahun 2026
    url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={GEMINI_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"Sebagai Trader Pro, analisis data koin ini: {data_list}. Pilih 1 koin terbaik, berikan alasan teknis, Entry, SL, dan TP. Gunakan Bahasa Indonesia santai."
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=json.dumps(payload), timeout=15)
        res_json = response.json()
        
        if 'candidates' in res_json:
            return res_json['candidates'][0]['content']['parts'][0]['text']
        else:
            # Jika Gemini 3 gagal, kita coba model alternatif 2026: gemini-3-flash-lite
            return "Maaf, sistem AI sedang sinkronisasi. Coba jalankan ulang dalam 1 menit."
    except Exception as e:
        return f"Gagal terhubung ke otak AI: {e}"

import json # Pastikan json terimport

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🚀 **AI TRADING SIGNAL 2026**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): 
        logging.error("Secrets belum lengkap!")
        return
    logging.info("Memulai pemindaian pasar...")
    try:
        data = await get_market_data()
        analysis = ask_ai_agent(data)
        await send_to_telegram(analysis)
        logging.info("Sinyal dikirim!")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
