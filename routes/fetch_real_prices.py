import json
import yfinance as yf
from routes.seed_companies import COMPANIES

def get_real_prices():
    updated = []
    
    # Map Indian tickers to Yahoo Finance format
    indian_tickers = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'BAJFINANCE', 'HINDUNILVR', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'WIPRO']
    
    for comp in COMPANIES:
        sym = comp['symbol']
        yf_sym = f"{sym}.NS" if sym in indian_tickers else sym
        
        try:
            ticker = yf.Ticker(yf_sym)
            info = ticker.fast_info
            current_price = info.last_price
            
            # Avoid nan or None
            if current_price and current_price > 0:
                comp['current_price'] = round(current_price, 2)
            else:
                pass # fallback to existing
        except Exception as e:
            pass # fallback to existing
            
        updated.append(comp)
        
    print(json.dumps(updated, indent=4))

if __name__ == '__main__':
    get_real_prices()
