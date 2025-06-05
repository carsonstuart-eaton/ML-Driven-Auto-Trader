import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.graph_objects as go
from backtester.engine import Backtester

# Load data (you can expand this to load dynamically)
ASSETS = {
    "BTC-USD": "data/processed/BTC-USD_processed.csv",
    "ETH-USD": "data/processed/ETH-USD_processed.csv"
}

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # for deployment (optional)

app.layout = html.Div([
    html.H1("Quant ML Strategy Dashboard", style={'textAlign': 'center'}),

    html.Label("Select Asset:"),
    dcc.Dropdown(
        id='asset-dropdown',
        options=[{"label": k, "value": k} for k in ASSETS.keys()],
        value='BTC-USD'
    ),

    html.Label("Prediction Threshold:"),
    dcc.Slider(
        id='threshold-slider',
        min=0.4, max=0.6, step=0.01,
        marks={i: str(i) for i in [0.4, 0.45, 0.5, 0.55, 0.6]},
        value=0.5
    ),

    dcc.Graph(id='equity-curve'),

    html.H4("Performance Summary"),
    html.Div(id='stats-output', style={'whiteSpace': 'pre-line'})
])

@app.callback(
    [Output('equity-curve', 'figure'),
     Output('stats-output', 'children')],
    [Input('asset-dropdown', 'value'),
     Input('threshold-slider', 'value')]
)
def update_dashboard(asset, threshold):
    df = pd.read_csv(ASSETS[asset], index_col=0, parse_dates=True)
    
    # Example: simulate probabilities → binary prediction (for demo)
    if 'prediction' not in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['prediction'] = (df['Close'].pct_change().shift(-1) > 0).astype(int)  # crude signal

    # Apply threshold
    df['prediction'] = (df['prediction'] > threshold).astype(int)

    # Run backtest
    bt = Backtester(df, prediction_col='prediction', return_col='Close')
    bt.prepare_data()
    results = bt.get_results()
    stats = bt.summary_stats()

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index, y=results['cumulative_market'],
        mode='lines', name='Market Return'
    ))
    fig.add_trace(go.Scatter(
        x=results.index, y=results['cumulative_strategy'],
        mode='lines', name='Strategy Return'
    ))
    fig.update_layout(title=f"Equity Curve: {asset}",
                      xaxis_title="Date", yaxis_title="Cumulative Return",
                      template="plotly_white")

    # Stats display
    stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
    return fig, stats_text

if __name__ == '__main__':
    app.run(debug=True)
