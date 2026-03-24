import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def get_coin_metrics(exchange, coin):
    symbol = coin['symbol']
    try:
        # Kita ambil OI dan Funding sebagai data pendukung untuk AI
        oi_info = await exchange.fetch_open_interest(symbol)
        funding = await exchange.fetch_funding_rate(symbol)
        
        return {
            'symbol': symbol.split(':')[0], # Ambil nama koinnya saja
            'price': coin['last'],
            'change_24h': f"{coin['percentage']:.2f}%",
            'vol_usdt': f"{coin['quoteVolume']:,.0f}",
            'oi': f"{float(oi_info['baseVolume']):,.0f}",
            'funding': f"{funding['fundingRate'] * 100:.4f}%",
            'high_24h': coin['high']
        }
    except Exception as e:
        # Silent pass untuk koin yang datanya tidak lengkap
        return None

async def get_high_precision_data():
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        
        # KITA AMBIL TOP 30 KOIN DENGAN VOLUME TERBESAR
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:30]
        
        tasks = [get_coin_metrics(exchange, coin) for coin in top_coins]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    finally:
        # PERBAIKAN: Cara benar menutup koneksi exchange di CCXT
        await exchange.close()

async def ask_ai_agent(data_list):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    
    prompt = f"""
    Kamu adalah Senior Crypto Strategist. Analisis data 30 koin paling aktif di market futures ini:
    {data_list}
    
    Tugasmu:
    1. Dari 30 koin ini, pilih 1 koin yang paling menjanjikan untuk Open Posisi SEKARANG.
    2. Kamu boleh memilih LONG (jika momentum kuat) atau SHORT (jika koin sudah overbought/lemah).
    3. Jelaskan analisismu secara singkat (lihat korelasi antara Harga, Volume, OI, dan Funding Rate).
    4. Berikan Trading Plan:
       - 🎯 Aksi: (LONG / SHORT)
       - 🟢 Entry Area:
       - 🔴 Stop Loss:
       - 🏁 Take Profit:
    Gunakan Bahasa Indonesia yang tajam dan to-the-point. Format menggunakan Markdown.
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    # PERBAIKAN: Gunakan aiohttp agar panggilan ke Gemini tidak memblokir sistem
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    error_msg = await response.text()
                    logging.error(f"Gemini API Error: {error_msg}")
                    return "⚠️ Gagal mengambil respons dari AI (API Error)."
        except asyncio.TimeoutError:
            return "⚠️ Waktu tunggu AI habis (Timeout). Pasar mungkin sedang terlalu bergejolak."
        except Exception as e:
            logging.error(f"Error AI Agent: {e}")
            return "⚠️ Terjadi kesalahan internal saat menghubungi AI."

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🔥 **AI MARKET RADAR (TOP 30 GATE.IO)**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): 
        logging.error("API Keys belum lengkap!")
        return
        
    logging.info("Memulai pemindaian 30 koin teraktif di Gate.io...")
    try:
        data = await get_high_precision_data()
        logging.info(f"Berhasil mengumpulkan data metrik untuk {len(data)} koin.")
        
        logging.info("Meminta analisis ke AI Agent...")
        # Sekarang fungsi ini di-await
        analysis = await ask_ai_agent(data) 
        
        await send_to_telegram(analysis)
        logging.info("Laporan sukses dikirim ke Telegram!")
    except Exception as e:
        logging.error(f"Sistem Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
