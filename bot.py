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
    # DAFTAR MODEL 2026 (Kita coba satu per satu sampai berhasil)
    # Gemini 2.0 dan versi 'latest' adalah yang paling stabil di 2026
    models_to_try = [
        "gemini-2.0-flash", 
        "gemini-1.5-flash-latest", 
        "gemini-1.5-flash",
        "gemini-pro"
    ]
    
    prompt = f"Analisis data koin ini: {data_list}. Pilih 1 koin terbaik untuk trading, berikan alasan teknis, Entry, SL, dan TP. Gunakan Bahasa Indonesia."
    
    for model_name in models_to_try:
        # Kita coba jalur v1beta karena biasanya lebih mendukung model terbaru
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_KEY}"
        headers = {'Content-Type': 'application/json'}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            res_json = response.json()
            
            if 'candidates' in res_json:
                logging.info(f"Berhasil menggunakan model: {model_name}")
                return res_json['candidates'][0]['content']['parts'][0]['text']
            else:
                logging.warning(f"Model {model_name} gagal: {res_json.get('error', {}).get('message')}")
                continue
        except Exception as e:
            logging.error(f"Koneksi ke {model_name} error: {e}")
            continue
            
    return "❌ Semua model AI (1.5 & 2.0) gagal diakses. Pastikan API Key benar dan API Generative Language sudah aktif di Google AI Studio."

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🚀 **AI TRADING SIGNAL 2026**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): 
        logging.error("API Keys belum lengkap di GitHub Secrets!")
        return
    logging.info("Memulai pemindaian pasar...")
    try:
        data = await get_market_data()
        analysis = ask_ai_agent(data)
        await send_to_telegram(analysis)
        logging.info("Proses selesai.")
    except Exception as e:
        logging.error(f"Error fatal: {e}")

if __name__ == "__main__":
    asyncio.run(main())
