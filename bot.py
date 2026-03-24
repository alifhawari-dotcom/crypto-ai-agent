import os
import ccxt
import google.generativeai as genai
import requests

# Konfigurasi API dari GitHub Secrets
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_TOKEN')
TG_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def get_crypto_data():
    exchange = ccxt.binanceusdm()
    markets = exchange.fetch_markets()
    # Ambil top 15 koin berdasarkan volume
    tickers = exchange.fetch_tickers()
    sorted_tickers = sorted(tickers.values(), key=lambda x: x['quoteVolume'], reverse=True)[:15]
    
    signals = []
    for ticker in sorted_tickers:
        symbol = ticker['symbol']
        try:
            oi_data = exchange.fapiPublicGetOpenInterest({'symbol': symbol.replace('/', '')})
            oi = float(oi_data['openInterest'])
            funding = exchange.fetch_funding_rate(symbol)['fundingRate'] * 100
            price_change = ticker['percentage']
            
            # Filter Sederhana: OI naik & harga stabil/naik
            if price_change > 0:
                signals.append(f"Koin: {symbol}, Harga: {ticker['last']}, Change: {price_change}%, OI: {oi}, Funding: {funding}%")
        except:
            continue
    return "\n".join(signals)

def ask_ai(data):
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Analisis data crypto berikut. Pilih 1-2 koin terbaik untuk LONG/SHORT. Berikan alasan logis, target profit, dan stop loss singkat: \n{data}"
    response = model.generate_content(prompt)
    return response.text

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": f"🤖 AI SCREENER REPORT:\n\n{message}"}
    requests.post(url, json=payload)

# Eksekusi
market_summary = get_crypto_data()
if market_summary:
    ai_analysis = ask_ai(market_summary)
    send_telegram(ai_analysis)
