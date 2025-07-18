import joblib
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import json


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse, predictions


def save_model(model, ticker, base_dir="models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{ticker}_model_{timestamp}.pkl"
    path = os.path.join(base_dir, filename)
    os.makedirs(base_dir, exist_ok=True)
    joblib.dump(model, path)
    return path


def log_metrics(ticker, rmse, path, log_file="models/metrics_log.json"):
    log_data = {
        "ticker": ticker,
        "rmse": rmse,
        "model_path": path,
        "timestamp": datetime.now().isoformat()
    }

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_data)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)
