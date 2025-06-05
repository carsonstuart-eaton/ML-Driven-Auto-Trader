import ta  # Technical Analysis library

def add_technical_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['volatility'] = df['Close'].pct_change().rolling(20).std()
    return df
