import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def load_data(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # Binary target
    return df

def train_xgboost_model(df):
    features = df.drop(columns=['target', 'Close', 'Open', 'High', 'Low', 'Volume'])
    target = df['target']

    X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    return model, X_test, y_test, y_pred

def save_model(model, path="models/xgb_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

if __name__ == "__main__":
    df = load_data("data/processed/BTC-USD_processed.csv")
    model, X_test, y_test, y_pred = train_xgboost_model(df)
    save_model(model)
