import json
import random
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models import Company, DailyStockData
from extensions import db

COMPANIES = [
    # 15 Indian Companies
    {'name': 'Reliance Industries', 'symbol': 'RELIANCE', 'sector': 'Oil & Gas', 'current_price': 1395.6, 'market_cap': 2000000000000, 'pe_ratio': 25.4},
    {'name': 'Tata Consultancy Services', 'symbol': 'TCS', 'sector': 'IT', 'current_price': 2371.3, 'market_cap': 1500000000000, 'pe_ratio': 30.1},
    {'name': 'HDFC Bank', 'symbol': 'HDFCBANK', 'sector': 'Banking', 'current_price': 809.5, 'market_cap': 1100000000000, 'pe_ratio': 15.5},
    {'name': 'Infosys', 'symbol': 'INFY', 'sector': 'IT', 'current_price': 1231.4, 'market_cap': 69000000000, 'pe_ratio': 24.2},
    {'name': 'ICICI Bank', 'symbol': 'ICICIBANK', 'sector': 'Banking', 'current_price': 1266.1, 'market_cap': 74000000000, 'pe_ratio': 17.8},
    {'name': 'State Bank of India', 'symbol': 'SBIN', 'sector': 'Banking', 'current_price': 1048.0, 'market_cap': 67000000000, 'pe_ratio': 9.5},
    {'name': 'Bharti Airtel', 'symbol': 'BHARTIARTL', 'sector': 'Telecom', 'current_price': 1850.6, 'market_cap': 68000000000, 'pe_ratio': 55.4},
    {'name': 'ITC Ltd', 'symbol': 'ITC', 'sector': 'FMCG', 'current_price': 299.6, 'market_cap': 52000000000, 'pe_ratio': 25.6},
    {'name': 'Larsen & Toubro', 'symbol': 'LT', 'sector': 'Engineering', 'current_price': 3458.0, 'market_cap': 50000000000, 'pe_ratio': 35.8},
    {'name': 'Bajaj Finance', 'symbol': 'BAJFINANCE', 'sector': 'Finance', 'current_price': 838.5, 'market_cap': 43000000000, 'pe_ratio': 34.2},
    {'name': 'Hindustan Unilever', 'symbol': 'HINDUNILVR', 'sector': 'FMCG', 'current_price': 2087.5, 'market_cap': 53000000000, 'pe_ratio': 50.1},
    {'name': 'Asian Paints', 'symbol': 'ASIANPAINT', 'sector': 'Paints', 'current_price': 2198.4, 'market_cap': 27000000000, 'pe_ratio': 52.4},
    {'name': 'Maruti Suzuki', 'symbol': 'MARUTI', 'sector': 'Automotive', 'current_price': 12606.0, 'market_cap': 39000000000, 'pe_ratio': 28.5},
    {'name': 'Sun Pharma', 'symbol': 'SUNPHARMA', 'sector': 'Pharma', 'current_price': 1761.1, 'market_cap': 37000000000, 'pe_ratio': 35.6},
    {'name': 'Wipro', 'symbol': 'WIPRO', 'sector': 'IT', 'current_price': 188.76, 'market_cap': 25000000000, 'pe_ratio': 21.4},
    
    # 15 Global Companies
    {'name': 'Apple Inc', 'symbol': 'AAPL', 'sector': 'Technology', 'current_price': 249.94, 'market_cap': 2800000000000, 'pe_ratio': 28.5},
    {'name': 'Microsoft Corporation', 'symbol': 'MSFT', 'sector': 'Technology', 'current_price': 391.79, 'market_cap': 3100000000000, 'pe_ratio': 35.2},
    {'name': 'Tesla Inc', 'symbol': 'TSLA', 'sector': 'Automotive', 'current_price': 392.78, 'market_cap': 580000000000, 'pe_ratio': 45.4},
    {'name': 'Alphabet Inc', 'symbol': 'GOOGL', 'sector': 'Technology', 'current_price': 307.69, 'market_cap': 1850000000000, 'pe_ratio': 24.5},
    {'name': 'Amazon.com Inc', 'symbol': 'AMZN', 'sector': 'E-commerce', 'current_price': 209.87, 'market_cap': 1800000000000, 'pe_ratio': 60.1},
    {'name': 'NVIDIA Corporation', 'symbol': 'NVDA', 'sector': 'Technology', 'current_price': 180.4, 'market_cap': 2100000000000, 'pe_ratio': 75.4},
    {'name': 'Meta Platforms', 'symbol': 'META', 'sector': 'Technology', 'current_price': 615.68, 'market_cap': 1200000000000, 'pe_ratio': 32.1},
    {'name': 'Berkshire Hathaway', 'symbol': 'BRK-B', 'sector': 'Finance', 'current_price': 484.47, 'market_cap': 880000000000, 'pe_ratio': 11.5},
    {'name': 'Visa Inc', 'symbol': 'V', 'sector': 'Finance', 'current_price': 299.02, 'market_cap': 580000000000, 'pe_ratio': 30.2},
    {'name': 'JPMorgan Chase', 'symbol': 'JPM', 'sector': 'Banking', 'current_price': 287.74, 'market_cap': 560000000000, 'pe_ratio': 12.1},
    {'name': 'Walmart Inc', 'symbol': 'WMT', 'sector': 'Retail', 'current_price': 121.98, 'market_cap': 490000000000, 'pe_ratio': 28.4},
    {'name': 'Mastercard', 'symbol': 'MA', 'sector': 'Finance', 'current_price': 488.47, 'market_cap': 440000000000, 'pe_ratio': 35.6},
    {'name': 'Johnson & Johnson', 'symbol': 'JNJ', 'sector': 'Healthcare', 'current_price': 237.28, 'market_cap': 370000000000, 'pe_ratio': 22.4},
    {'name': 'Procter & Gamble', 'symbol': 'PG', 'sector': 'FMCG', 'current_price': 146.71, 'market_cap': 380000000000, 'pe_ratio': 25.1},
    {'name': 'Netflix Inc', 'symbol': 'NFLX', 'sector': 'Entertainment', 'current_price': 94.7, 'market_cap': 270000000000, 'pe_ratio': 45.8}
]

def calculate_sma(data, window):
    if len(data) < window:
        return sum(data) / len(data) if data else 0
    return sum(data[-window:]) / window

def calculate_ema(data, window):
    if not data: return 0
    prices = pd.Series(data)
    ema = prices.ewm(span=window, adjust=False).mean()
    return ema.iloc[-1]

def update_company_data():
    """Update all companies with intelligent simulated data daily and save to CSV"""
    companies = Company.query.all()
    for company in companies:
        historical_data = json.loads(company.historical_data) if company.historical_data else []
        
        # Get historical close prices
        close_prices = [day['Close'] for day in historical_data]
        
        # Intelligent fluctuation
        sma_10 = calculate_sma(close_prices, 10)
        ema_10 = calculate_ema(close_prices, 10)
        
        # Determine trend based on EMA vs SMA
        if ema_10 > sma_10:
            trend = 'bullish'
            change_percent = random.uniform(0.005, 0.03) # +0.5% to +3%
        else:
            trend = 'bearish'
            change_percent = random.uniform(-0.03, -0.005) # -3% to -0.5%
            
        # Add random noise
        change_percent += random.uniform(-0.01, 0.01)
        
        new_price = company.current_price * (1 + change_percent)
        # Avoid extreme drops or spikes from initial
        initial_price = next((c['current_price'] for c in COMPANIES if c['symbol'] == company.symbol), company.current_price)
        new_price = max(new_price, initial_price * 0.5)
        new_price = min(new_price, initial_price * 2.0)
        
        company.current_price = round(new_price, 2)
        
        # Update historical data
        if historical_data:
            historical_data.pop(0)
            
        today = {
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Close': round(company.current_price, 2),
            'High': round(company.current_price * (1 + random.uniform(0, 0.02)), 2),
            'Low': round(company.current_price * (1 - random.uniform(0, 0.02)), 2),
            'Open': round(company.current_price * (1 + random.uniform(-0.01, 0.01)), 2),
            'Volume': random.randint(500000, 5000000)
        }
        historical_data.append(today)
        company.historical_data = json.dumps(historical_data)
        

        # Create a new daily stock data entry
        daily_record = DailyStockData(
            company_id=company.id,
            current_price=today['Close'],
            high=today['High'],
            low=today['Low'],
            open_price=today['Open'],
            volume=today['Volume'],
            timestamp=datetime.utcnow()
        )
        db.session.add(daily_record)

    # Delete DailyStockData older than 24 hours
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    DailyStockData.query.filter(DailyStockData.timestamp < cutoff_time).delete()
    db.session.commit()

def generate_mock_historical_data(current_price, days=60):
    """Generate mock historical data for a company"""
    data = []
    price = current_price * 0.8
    for i in range(days):
        date = (datetime.now() - timedelta(days=days-i)).strftime('%Y-%m-%d')
        change_percent = random.uniform(-0.02, 0.02)
        price = price * (1 + change_percent)
        data.append({
            'Date': date,
            'Close': round(price, 2),
            'High': round(price * 1.015, 2),
            'Low': round(price * 0.985, 2),
            'Open': round(price * 0.99, 2),
            'Volume': random.randint(500000, 5000000)
        })
    return data

def initialize_companies():
    """Insert predefined companies with mock historical data into DB"""
    # Delete old records if we don't have exactly 30 companies, or if running for first time
    if Company.query.count() != 30:
        # Avoid foreign key constraint issues
        from models import InvestmentPrediction, CompanyComparison, DailyStockData
        db.session.query(InvestmentPrediction).delete()
        db.session.query(CompanyComparison).delete()
        db.session.query(DailyStockData).delete()
        db.session.query(Company).delete()
        db.session.commit()

        for company_info in COMPANIES:
            historical_data = generate_mock_historical_data(company_info['current_price'])
            company = Company(
                name=company_info['name'],
                symbol=company_info['symbol'],
                sector=company_info['sector'],
                current_price=company_info['current_price'],
                market_cap=company_info['market_cap'],
                pe_ratio=company_info['pe_ratio'],
                historical_data=json.dumps(historical_data)
            )
            db.session.add(company)
            db.session.flush()


            daily_record = DailyStockData(
                company_id=company.id,
                current_price=company.current_price,
                high=round(company.current_price * 1.02, 2),
                low=round(company.current_price * 0.98, 2),
                open_price=round(company.current_price * 0.99, 2),
                volume=random.randint(500000, 5000000),
                timestamp=datetime.utcnow()
            )
            db.session.add(daily_record)
            
        db.session.commit()
