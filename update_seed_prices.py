import yfinance as yf
from seed_companies import COMPANIES
import json
import re

def update_companies_file():
    indian_tickers = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'BAJFINANCE', 'HINDUNILVR', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'WIPRO']
    
    with open('seed_companies.py', 'r') as f:
        content = f.read()

    for comp in COMPANIES:
        sym = comp['symbol']
        yf_sym = f"{sym}.NS" if sym in indian_tickers else sym
        
        try:
            ticker = yf.Ticker(yf_sym)
            info = ticker.fast_info
            current_price = info.last_price
            
            if current_price and current_price > 0:
                old_price_str = f"'current_price': {comp['current_price']}"
                new_price = round(current_price, 2)
                new_price_str = f"'current_price': {new_price}"
                
                # Careful replacement to only match the specific company's dictionary
                pattern = f"('symbol': '{sym}'.*?)('current_price': [\\d\\.]+)"
                replacement = f"\\g<1>{new_price_str}"
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        except Exception as e:
            pass

    with open('seed_companies.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    update_companies_file()
