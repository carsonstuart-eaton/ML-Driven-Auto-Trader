def clean_data(df):
    df = df.dropna()
    df = df.sort_index()
    df = df[df['Volume'] > 0]
    return df
