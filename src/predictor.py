import joblib

def load_model(model_path):
    return joblib.load(model_path)

def predict_next(model, latest_features):
    return model.predict(latest_features)[0]
