import joblib
import os
import glob

def get_latest_model_path(symbol: str, base_dir="models"):
    model_files = glob.glob(f"{base_dir}/{symbol}_model_*.pkl")
    if not model_files:
        return None
    latest = max(model_files, key=os.path.getmtime)
    return latest

def load_model(model_path):
    return joblib.load(model_path)

def predict_next(model, latest_features):
    return model.predict(latest_features)[0]
