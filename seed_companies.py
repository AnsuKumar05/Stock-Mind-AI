# company_data.py
import json
import random
from datetime import datetime, timedelta
from models import Company
from extensions import db

# Define all 15 companies
COMPANIES = [
    {
        'name': 'Reliance Industries Ltd',
        'symbol': 'RELIANCE',
        'sector': 'Oil & Gas',
        'current_price': 2450.75,
        'market_cap': 1658000000000,
        'pe_ratio': 15.2
    },
    {
        'name': 'Tata Consultancy Services',
        'symbol': 'TCS',
        'sector': 'Information Technology',
        'current_price': 3420.50,
        'market_cap': 1245000000000,
        'pe_ratio': 26.8
    },
    {
        'name': 'HDFC Bank Ltd',
        'symbol': 'HDFCBANK',
        'sector': 'Banking',
        'current_price': 1580.25,
        'market_cap': 1198000000000,
        'pe_ratio': 19.5
    },
    {
        'name': 'Infosys Ltd',
        'symbol': 'INFY',
        'sector': 'Information Technology',
        'current_price': 1456.80,
        'market_cap': 620000000000,
        'pe_ratio': 23.4
    },
    {
        'name': 'ICICI Bank Ltd',
        'symbol': 'ICICIBANK',
        'sector': 'Banking',
        'current_price': 945.30,
        'market_cap': 664000000000,
        'pe_ratio': 17.8
    },
    {
        'name': 'Hindustan Unilever Ltd',
        'symbol': 'HINDUNILVR',
        'sector': 'FMCG',
        'current_price': 2785.90,
        'market_cap': 653000000000,
        'pe_ratio': 55.2
    },
    {
        'name': 'State Bank of India',
        'symbol': 'SBIN',
        'sector': 'Banking',
        'current_price': 578.45,
        'market_cap': 515000000000,
        'pe_ratio': 12.3
    },
    {
        'name': 'Bharti Airtel Ltd',
        'symbol': 'BHARTIARTL',
        'sector': 'Telecommunications',
        'current_price': 1124.30,
        'market_cap': 678000000000,
        'pe_ratio': 45.6
    },
    {
        'name': 'ITC Ltd',
        'symbol': 'ITC',
        'sector': 'FMCG',
        'current_price': 456.75,
        'market_cap': 567000000000,
        'pe_ratio': 28.9
    },
    {
        'name': 'Larsen & Toubro Ltd',
        'symbol': 'LT',
        'sector': 'Engineering',
        'current_price': 3245.80,
        'market_cap': 456000000000,
        'pe_ratio': 31.2
    },
    {
        'name': 'Asian Paints Ltd',
        'symbol': 'ASIANPAINT',
        'sector': 'Paints',
        'current_price': 3180.50,
        'market_cap': 305000000000,
        'pe_ratio': 52.1
    },
    {
        'name': 'Maruti Suzuki India Ltd',
        'symbol': 'MARUTI',
        'sector': 'Automobile',
        'current_price': 10450.25,
        'market_cap': 315000000000,
        'pe_ratio': 28.7
    },
    {
        'name': 'Wipro Ltd',
        'symbol': 'WIPRO',
        'sector': 'Information Technology',
        'current_price': 456.90,
        'market_cap': 245000000000,
        'pe_ratio': 21.8
    },
    {
        'name': 'Bajaj Finance Ltd',
        'symbol': 'BAJFINANCE',
        'sector': 'Financial Services',
        'current_price': 6780.40,
        'market_cap': 419000000000,
        'pe_ratio': 33.5
    },
    {
        'name': 'Sun Pharmaceutical Industries',
        'symbol': 'SUNPHARMA',
        'sector': 'Pharmaceuticals',
        'current_price': 1234.60,
        'market_cap': 296000000000,
        'pe_ratio': 39.2
    }
]

def update_company_data():
    """Update all companies with new mock data daily"""
    companies = Company.query.all()
    for company in companies:
        # Small random change in price
        change_percent = random.uniform(-0.03, 0.10)  # -3% to +10%
        new_price = company.current_price * (1 + change_percent)

        # Clamp within a reasonable range
        new_price = max(new_price, company.current_price * 0.8)
        new_price = min(new_price, company.current_price * 1.2)

        company.current_price = round(new_price, 2)

        # Update historical data
        historical_data = json.loads(company.historical_data)
        if historical_data:
            historical_data.pop(0)  # drop oldest
        today = {
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Close': round(company.current_price, 2),
            'High': round(company.current_price * 1.02, 2),
            'Low': round(company.current_price * 0.98, 2),
            'Open': round(company.current_price * 0.99, 2),
            'Volume': random.randint(100000, 1000000)
        }
        historical_data.append(today)
        company.historical_data = json.dumps(historical_data)

    db.session.commit()

def generate_mock_historical_data(current_price, days=30):
    """Generate mock historical data for a company"""
    data = []
    price = current_price * 0.9
    for i in range(days):
        date = datetime.now() - timedelta(days=days-i)
        change_percent = random.uniform(-0.05, 0.05)
        price = price * (1 + change_percent)
        price = min(price, current_price*1.2)
        price = max(price, current_price*0.8)
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Close': round(price, 2),
            'High': round(price * 1.02, 2),
            'Low': round(price * 0.98, 2),
            'Open': round(price * 0.99, 2),
            'Volume': random.randint(100000, 1000000)
        })
    return data

def initialize_companies():
    """Insert predefined companies with mock historical data into DB"""
    if Company.query.count() > 0:
        return  # Already initialized
    
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
    db.session.commit()
