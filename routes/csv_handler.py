import os
import pandas as pd
import threading
from datetime import datetime, timedelta

CSV_DIR = 'uploads'
os.makedirs(CSV_DIR, exist_ok=True)


csv_lock = threading.Lock()

def get_csv_path(symbol):
    return os.path.join(CSV_DIR, f"{symbol}_data.csv")

def append_to_csv(symbol, data_row):
    """
    Append new hourly data to CSV.
    data_row: dict with Date, Close, Volume, High, Low, Open
    """
    with csv_lock:
        path = get_csv_path(symbol)
        df_new = pd.DataFrame([data_row])
        if not os.path.exists(path):
            df_new.to_csv(path, index=False)
        else:
            df_new.to_csv(path, mode='a', header=False, index=False)

def get_last_n_records(symbol, n=20):
    with csv_lock:
        path = get_csv_path(symbol)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
        return df.tail(n)

def clean_old_csv_data(max_hours=4):
    """Keep only last max_hours of data in CSV"""
    with csv_lock:
        cutoff = datetime.now() - timedelta(hours=max_hours)
        for filename in os.listdir(CSV_DIR):
            if filename.endswith('_data.csv'):
                path = os.path.join(CSV_DIR, filename)
                try:
                    df = pd.read_csv(path)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df[df['Date'] >= cutoff]
                    df.to_csv(path, index=False)
                except Exception as e:
                    print(f"Error cleaning {filename}: {e}")
