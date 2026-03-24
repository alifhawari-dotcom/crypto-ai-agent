import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ambil Keys dari Environment Variables
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def get_coin_metrics(exchange, coin):
    symbol = coin['symbol']
    try:
        # 1. Ambil Funding Rate (Biasanya selalu berhasil)
        funding = await exchange.fetch_funding_rate(symbol)
        funding_val = f"{funding['fundingRate'] * 100:.4f}%"
        
        # 2. Ambil OI dengan Try-Except terpisah
        # Jika OI gagal ditarik di Gate.io, bot TIDAK membuang koin ini, melainkan diisi "N/A"
        try:
            oi_info = await exchange.fetch_open_interest(symbol)
            oi_val = f"{float(oi_info['baseVolume']):,.0f}"
        except:
            oi_val = "N/A" 
            
        return {
            'symbol': symbol.split(':')[0], # Ambil nama koinnya saja
            'price': coin['last'],
            'change_24h': f"{coin['percentage']:.2f}%",
            'vol_usdt': f"{coin['quoteVolume']:,.0f}",
            'oi': oi_val,
            'funding': funding_val,
            'high_24h': coin['high']
        }
    except Exception as e:
        logging.warning(f"Gagal total mengambil data {symbol}: {e}")
        return None

async def get_high_precision_data():
    # KITA GUNAKAN GATE.IO (Aman dari blokir IP Amerika di GitHub Actions)
    exchange = ccxt.gate({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    
    try:
        tickers = await exchange.fetch_tickers()
        
        # Ambil Top 30 Koin dengan Volume Terbesar
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:30]
        
        # Semaphore: Membatasi antrean maksimal 5 request bersamaan agar Gate.io tidak marah
        sem = asyncio.Semaphore(5)
        
        async def safe_get_metrics(coin):
            async with sem:
                return await get_coin_metrics(exchange, coin)
                
        tasks = [safe_get_metrics(coin) for coin in top_coins]
        results = await asyncio.gather(*tasks)
        
        # Saring data yang valid
        valid_results = [r for r in results if r is not None]
        return valid_results
    finally:
        await exchange.close()

async def ask_ai_agent(data_list):
    if not data_list:
        return "⚠️ Peringatan: Data radar kosong. Cek koneksi API Exchange."
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    
    prompt = f"""
    Kamu adalah Senior Crypto Strategist. Analisis data {len(data_list)} koin paling aktif di Gate.io Futures ini:
    {data_list}
    
    Tugasmu:
    1. Pilih 1 koin yang paling menjanjikan untuk Open Posisi SEKARANG.
    2. Kamu boleh memilih LONG atau SHORT.
    3. Jelaskan analisismu secara singkat (Harga, Volume, OI, Funding Rate).
    4. Berikan Trading Plan:
       - 🎯 Aksi: (LONG / SHORT)
       - 🟢 Entry Area:
       - 🔴 Stop Loss:
       - 🏁 Take Profit:
    Gunakan Bahasa Indonesia yang tajam dan to-the-point. Format Markdown.
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    return f"⚠️ API Error: {await response.text()}"
        except Exception as e:
            return f"⚠️ Kesalahan AI: {e}"

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🔥 **AI MARKET RADAR (GATE.IO)**\n\n{text}", "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()

async def main():
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]): 
        logging.error("API Keys belum lengkap di Environment Variables!")
        return
        
    logging.info("Memulai pemindaian koin teraktif di Gate.io...")
    try:
        data = await get_high_precision_data()
        logging.info(f"Berhasil mengumpulkan data metrik untuk {len(data)} koin.")
        
        analysis = await ask_ai_agent(data) 
        await send_to_telegram(analysis)
        logging.info("Laporan sukses dikirim ke Telegram!")
    except Exception as e:
        logging.error(f"Sistem Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
