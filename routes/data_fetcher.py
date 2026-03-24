import yfinance as yf
from datetime import datetime
from routes.csv_handler import append_to_csv
from routes.prediction import clear_prediction_cache

def fetch_and_update_hourly(app):
    """
    Fetch YFinance hourly data, update DB, append to CSV, and invalidate cache.
    Called inside an app context.
    """
    from routes.extensions import db
    from routes.models import Company, DailyStockData
    
    app.logger.info("Starting hourly YFinance data fetch.")
    companies = Company.query.all()
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:00')

    updated_count = 0
    for comp in companies:
        try:
            # Format symbol for Indian stocks on Yahoo Finance correctly
            indian_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'BAJFINANCE', 'HINDUNILVR', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'WIPRO']
            yf_symbol = f"{comp.symbol}.NS" if comp.symbol in indian_stocks else comp.symbol
            
            ticker = yf.Ticker(yf_symbol)
            # Fetch recent to ensure we get the latest hour
            hist = ticker.history(period='1d', interval='1h')
            
            if hist.empty:
                continue
            
            latest_row = hist.iloc[-1]
            price = float(latest_row['Close'])
            volume = int(latest_row['Volume'])
            high = float(latest_row['High'])
            low = float(latest_row['Low'])
            open_price = float(latest_row['Open'])
            
            if len(hist) > 1:
                prev_close = float(hist.iloc[-2]['Close'])
                if prev_close > 0:
                    change_pct = ((price - prev_close) / prev_close) * 100
                else:
                    change_pct = 0.0
            else:
                change_pct = 0.0

            # Update Company model
            comp.current_price = price
            
            # Attempt to fetch market cap natively
            try:
                mcap = ticker.fast_info.market_cap
                if mcap and mcap > 0:
                    comp.market_cap = float(mcap)
            except Exception:
                pass
            
            # Add historical snapshot to DB
            new_hourly = DailyStockData(
                company_id=comp.id,
                current_price=price,
                high=high,
                low=low,
                open_price=open_price,
                volume=volume,
                timestamp=datetime.utcnow()
            )
            db.session.add(new_hourly)
            updated_count += 1
            app.logger.info(f"Updated {comp.symbol} to {price}")

        except Exception as e:
            app.logger.error(f"Error fetching data for {comp.symbol}: {e}")
            
    # Commit DB transaction
    db.session.commit()
    app.logger.info(f"Finished fetching data. Updated {updated_count} companies.")

    # Invalidate prediction cache
    clear_prediction_cache()
