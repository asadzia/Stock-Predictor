from src.utils import (
    train_random_forest,
    evaluate_model,
    save_model,
    log_metrics,
)
from sklearn.model_selection import train_test_split


def train_model(df, ticker="AAPL"):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = train_random_forest(X_train, y_train)
    rmse, _ = evaluate_model(model, X_test, y_test)
    path = save_model(model, ticker)
    log_metrics(ticker, rmse, path)

    return rmse, path
