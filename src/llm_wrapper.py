from src.fetch_data import fetch_daily_stock_data
from src.features import generate_features
from src.predictor import get_latest_model_path, load_model, predict_next
from pandas.tseries.offsets import BDay

def get_prediction_with_context(symbol: str):
    df_raw = fetch_daily_stock_data(symbol)
    df_feat = generate_features(df_raw)
    model_path = get_latest_model_path(symbol)
    if not model_path:
        return f"No model found for {symbol}. Please train one."

    model = load_model(model_path)
    latest_features = df_feat.drop(columns=['target']).iloc[-1:]
    prediction = predict_next(model, latest_features)

    latest_close = df_raw['close'][-1]
    rsi_latest = df_feat['rsi'].iloc[-1]
    next_day = df_raw.index.max() + BDay(1)

    explanation = f"""
    Symbol: {symbol}
    Latest close: ${latest_close:.2f}
    RSI: {rsi_latest:.2f}
    Predicted close for {next_day.date()}: ${prediction:.2f}
    """
    return explanation
