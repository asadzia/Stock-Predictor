import sys
import os

# Add project root to sys.path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fetch_data import fetch_daily_stock_data
from src.features import generate_features
from src.model_train import train_model

if __name__ == "__main__":
    ticker = "AAPL"  # later replace with argparse input
    df_raw = fetch_daily_stock_data(ticker)
    df_feat = generate_features(df_raw)

    rmse, model_path = train_model(df_feat, ticker)
    print(f"✅ Training completed for {ticker}")
    print(f"📦 Model saved to: {model_path}")
    print(f"📉 RMSE: {rmse:.4f}")