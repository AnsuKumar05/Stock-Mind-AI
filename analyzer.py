import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def perform_analysis(company):
    if not company.historical_data:
        return None
        
    historical_data = json.loads(company.historical_data)
    df = pd.DataFrame(historical_data)
    
    if 'Close' not in df.columns or df.empty:
        return None

    close_prices = pd.to_numeric(df['Close'], errors='coerce').dropna()
    if close_prices.empty:
        return None
        
    # Basic analysis
    mean_price = close_prices.mean()
    median_price = close_prices.median()
    std_price = close_prices.std()
    
    # Returns for growth, volatility
    returns = close_prices.pct_change().dropna()
    avg_return = returns.mean() * 100 if len(returns) > 0 else 0
    volatility = returns.std() * 100 if len(returns) > 0 else 0
    growth_rate = ((close_prices.iloc[-1] / (close_prices.iloc[0] + 1e-9)) - 1) * 100 if len(close_prices) > 0 else 0
    
    # Stability: Inverse of volatility
    stability = 100 - volatility if volatility < 100 else 0
    
    # Predict next day using Linear Regression
    df['id'] = np.arange(len(df))
    X = df[['id']].values
    y = close_prices.values
    
    # Needs matching lengths
    if len(X) != len(y):
        X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    next_day_id = np.array([[len(y)]])
    predicted_lr = model.predict(next_day_id)[0]
    
    # Predict next day using SMA and EMA
    df_sma = calculate_sma(close_prices, 10).fillna(close_prices.iloc[0] if len(close_prices) > 0 else 0)
    df_ema = calculate_ema(close_prices, 10).fillna(close_prices.iloc[0] if len(close_prices) > 0 else 0)
    
    sma_pred = df_sma.iloc[-1]
    ema_pred = df_ema.iloc[-1]
    
    # Ensemble prediction (average of LR, SMA, EMA)
    next_day_price = (predicted_lr + sma_pred + ema_pred) / 3
    current_price = close_prices.iloc[-1]
    trend_direction = "Bullish" if next_day_price > current_price else "Bearish"
    
    return {
        "symbol": company.symbol,
        "name": company.name,
        "mean": mean_price,
        "median": median_price,
        "std_dev": std_price,
        "avg_return": avg_return,
        "growth_rate": growth_rate,
        "volatility": volatility,
        "stability": stability,
        "current_price": current_price,
        "next_day_price": next_day_price,
        "trend_direction": trend_direction
    }

def compare_all_companies(companies):
    results = []
    for comp in companies:
        analysis = perform_analysis(comp)
        if analysis:
            results.append(analysis)
            
    if not results:
        return None

    # Ranking logic
    for r in results:
        r['score'] = (r['growth_rate'] * 0.4) + (r['avg_return'] * 0.3) + (r['stability'] * 0.3) - (r['volatility'] * 0.2)
        
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Generate investment recommendations for best
    best_company = results[0]
    risk_level = "Low" if best_company['volatility'] < 2.0 else "Medium" if best_company['volatility'] < 5.0 else "High"
    
    recommendation = {
        "best_company": best_company['name'],
        "best_symbol": best_company['symbol'],
        "risk_level": risk_level,
        "justification": f"Ranked highest due to a strong growth rate of {best_company['growth_rate']:.2f}% and a solid stability score of {best_company['stability']:.2f}% with {risk_level} volatility. The ensemble model predicts a {best_company['trend_direction']} trend with next-day target of {best_company['next_day_price']:.2f}.",
    }
    
    return {
        "rankings": results,
        "recommendation": recommendation
    }
