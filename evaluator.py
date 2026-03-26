import os
import asyncio
import pandas as pd
import ccxt.async_support as ccxt
from datetime import datetime, timezone

LOG_FILE = "trade_history.csv"

async def evaluate_trades():
    if not os.path.exists(LOG_FILE):
        print(f"❌ File {LOG_FILE} tidak ditemukan.")
        return

    df = pd.read_csv(LOG_FILE)
    
    # Cek jika tidak ada yang OPEN
    if 'OPEN' not in df['Result'].values:
        print("✅ Tidak ada posisi OPEN yang perlu dievaluasi.")
        return

    # Kita pakai Gate.io untuk cek pergerakan harga (karena bot utamanya pakai ini)
    exchange = ccxt.gate({'enableRateLimit': True})
    
    print(f"🔍 Memulai evaluasi {len(df[df['Result'] == 'OPEN'])} posisi OPEN...")
    changed_count = 0
    
    for index, row in df.iterrows():
        if row['Result'] != 'OPEN':
            continue
            
        symbol = row['Symbol']
        # Bersihkan format string "DOT/USDT:USDT" menjadi "DOT/USDT" untuk fetch data
        fetch_symbol = symbol.split(':')[0] 
        
        # Konversi waktu sinyal ke Unix Timestamp (Millisecond)
        try:
            trade_time = datetime.strptime(row['Time'], "%Y-%m-%d %H:%M")
            trade_time = trade_time.replace(tzinfo=timezone.utc)
            since_ms = int(trade_time.timestamp() * 1000)
        except Exception as e:
            print(f"⚠️ Format waktu error di baris {index}: {e}")
            continue

        try:
            # Ambil sejarah candle 15m sejak sinyal muncul
            candles = await exchange.fetch_ohlcv(fetch_symbol, '15m', since=since_ms, limit=100)
            if not candles:
                continue
            
            action = row['Action']
            entry  = float(row['Entry'])
            sl     = float(row['SL'])
            tp1    = float(row['TP1'])
            
            is_active = False
            result = "OPEN"
            
            for candle in candles:
                high = candle[2]
                low  = candle[3]
                
                if action == "LONG":
                    if not is_active:
                        if low <= sl:  # Harga kena area SL sebelum Entry tersentuh
                            result = "CANCELLED"
                            break
                        if high >= entry:
                            is_active = True
                    
                    if is_active:
                        if low <= sl:
                            result = "LOSS"
                            break
                        elif high >= tp1:
                            result = "WIN"
                            break
                            
                elif action == "SHORT":
                    if not is_active:
                        if high >= sl: # Harga kena area SL sebelum Entry tersentuh
                            result = "CANCELLED"
                            break
                        if low <= entry:
                            is_active = True
                            
                    if is_active:
                        if high >= sl:
                            result = "LOSS"
                            break
                        elif low <= tp1:
                            result = "WIN"
                            break
            
            # Jika sudah > 24 jam dan masih OPEN -> EXPIRED
            if result == "OPEN":
                last_candle_time = candles[-1][0]
                if (last_candle_time - since_ms) > (24 * 60 * 60 * 1000):
                    result = "EXPIRED"
            
            if result != "OPEN":
                df.at[index, 'Result'] = result
                changed_count += 1
                print(f"🔄 {symbol} ({action}) dievaluasi -> {result}")
                
        except Exception as e:
            print(f"⚠️ Error fetching {symbol}: {e}")
            
        await asyncio.sleep(0.5) # Jaga API rate limit bursa

    await exchange.close()
    
    # Simpan kembali jika ada perubahan
    if changed_count > 0:
        df.to_csv(LOG_FILE, index=False)
        print(f"💾 Berhasil menyimpan {changed_count} pembaruan ke {LOG_FILE}")
    else:
        print("⏳ Belum ada posisi yang mencapai Target (TP) atau Stop Loss (SL).")

if __name__ == "__main__":
    asyncio.run(evaluate_trades())
