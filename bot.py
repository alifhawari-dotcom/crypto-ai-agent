import os
import asyncio
import logging
import aiohttp
import ccxt.async_support as ccxt  # Menggunakan versi Async agar 10x lebih cepat
import google.generativeai as genai

# Setup Logging Profesional (Bukan sekadar print)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ambil kunci rahasia dari GitHub Secrets / Environment
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def get_coin_metrics(exchange, coin):
    """Fungsi helper untuk mengambil OI dan Funding secara paralel"""
    symbol = coin['symbol']
    try:
        # Panggil API secara bersamaan untuk setiap koin
        oi_task = exchange.fapiPublicGetOpenInterest({'symbol': symbol.replace('/', '')})
        funding_task = exchange.fetch_funding_rate(symbol)
        
        # Tunggu kedua data selesai diambil
        oi_info, funding = await asyncio.gather(oi_task, funding_task)
        
        oi_value = float(oi_info['openInterest'])
        funding_rate = funding['fundingRate'] * 100
        
        price_change = coin['percentage']
        current_price = coin['last']
        high_24h = coin['high']
        
        # Logika Screening Jitu
        if price_change > 1.5 and funding_rate < 0.015:
            if current_price >= (high_24h * 0.98):
                # Tambahkan metrik jarak persentase agar AI lebih pintar menganalisis
                distance_to_high = ((high_24h - current_price) / current_price) * 100
                return {
                    'symbol': symbol,
                    'price': current_price,
                    'change_24h': f"{price_change:.2f}%",
                    'open_interest': f"{oi_value:,.0f}",
                    'funding_rate': f"{funding_rate:.4f}%",
                    'distance_to_high': f"Minus {distance_to_high:.2f}% ke Resistance High 24h"
                }
    except Exception as e:
        logging.debug(f"Skip {symbol} karena error/data tidak lengkap: {e}")
    return None

async def get_high_precision_data():
    # Aktifkan Rate Limit bawaan ccxt agar tidak kena ban dari Binance
    exchange = ccxt.binanceusdm({'enableRateLimit': True})
    try:
        tickers = await exchange.fetch_tickers()
        
        # Filter 20 koin dengan Volume tertinggi
        top_coins = sorted(tickers.values(), key=lambda x: x['quoteVolume'], reverse=True)[:20]
        
        # Proses 20 koin SECARA PARALEL (Ini yang bikin script kamu 10x lebih cepat)
        tasks = [get_coin_metrics(exchange, coin) for coin in top_coins]
        results = await asyncio.gather(*tasks)
        
        # Bersihkan hasil dari nilai 'None'
        hot_list = [r for r in results if r is not None]
        return hot_list
    finally:
        # Wajib ditutup agar koneksi tidak menggantung (Memory Leak)
        await exchange.close()

def ask_ai_agent(data_list):
    if not data_list:
        return "📉 Pasar sedang *choppy* atau tenang. Belum ada setup *breakout* dengan probabilitas tinggi hari ini."
        
    genai.configure(api_key=GEMINI_KEY)
    # Sebagai Gemini, saya sarankan menggunakan 1.5 Pro untuk reasoning yang lebih dalam jika ada kuota, 
    # namun 1.5 Flash sudah sangat cepat untuk tugas struktural ini.
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Kamu adalah Crypto Quantitative Analyst tingkat Institusi.
    Berikut adalah data live koin Binance Futures yang sedang mendekati harga tertinggi 24 jam dengan struktur funding rate yang sehat:
    
    {data_list}
    
    Tugasmu:
    1. Pilih HANYA 1 koin (Top Pick) dengan probabilitas *breakout* paling tinggi.
    2. Analisis singkat mengapa koin ini dipilih berdasarkan rasio Open Interest dan rendahnya Funding Rate (indikasi belum over-leveraged).
    3. Berikan Trading Plan yang Presisi:
       - 🟢 Entry Area: (Angka spesifik)
       - 🔴 Stop Loss: (Level support terdekat / invalidasi)
       - 🎯 Take Profit 1 & 2: (Gunakan target logis dengan Risk/Reward minimal 1:2)
    4. Format jawaban langsung menggunakan Markdown yang rapi (tanpa basa-basi).
    """
    
    response = model.generate_content(prompt)
    return response.text

async def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID, 
        "text": f"🚀 **AI AGENT SCANNER REPORT**\n\n{text}", 
        "parse_mode": "Markdown"
    }
    
    # Gunakan aiohttp agar pengiriman ke Telegram tidak memblokir sistem
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                logging.error(f"Gagal kirim Telegram: {await response.text()}")

async def main():
    # Validasi Environment Variables di awal (Mencegah error di tengah proses)
    if not all([GEMINI_KEY, TG_TOKEN, TG_CHAT_ID]):
        logging.error("CRITICAL ERROR: API Keys tidak lengkap di Environment Variables!")
        return

    logging.info("Memulai pemindaian pasar Binance Futures...")
    try:
        data = await get_high_precision_data()
        logging.info(f"Ditemukan {len(data)} koin potensial.")
        
        analysis = ask_ai_agent(data)
        await send_to_telegram(analysis)
        logging.info("Sukses! Laporan AI telah dikirim ke Telegram.")
    except Exception as e:
        logging.error(f"Terjadi kegagalan sistem utama: {e}")

# Entry Point untuk menjalankan script Asynchronous
if __name__ == "__main__":
    asyncio.run(main())
