import pandas as pd
import ta

def generate_features(df):
    df['sma_5'] = df['close'].rolling(5).mean()
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['volatility'] = df['close'].rolling(5).std()
    df['return'] = df['close'].pct_change()
    df['target'] = df['close'].shift(-1)
    return df.dropna()
