import os
import sys
import glob

# Set project root so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from src.fetch_data import fetch_daily_stock_data
from src.features import generate_features
from src.predictor import load_model, predict_next
from pandas.tseries.offsets import BDay

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("🔮 AI Stock Price Predictor")

symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()

def get_latest_model_path(symbol: str):
    model_files = glob.glob(f"models/{symbol}_model_*.pkl")
    if not model_files:
        return None
    latest = max(model_files, key=os.path.getmtime)
    return latest

def calculate_summary_metrics(df_raw):
    latest_close = df_raw['close'][-1]
    pct_change = df_raw['close'].pct_change().iloc[-1] * 100
    high_52w = df_raw['close'][-252:].max() if len(df_raw) >= 252 else df_raw['close'].max()
    low_52w = df_raw['close'][-252:].min() if len(df_raw) >= 252 else df_raw['close'].min()
    print()
    # avg_volume = df_raw['volume'][-20:].mean() if 'volume' in df_raw.columns else np.nan
    return latest_close, pct_change, high_52w, low_52w

def get_buy_sell_signal(rsi):
    if rsi < 30:
        return "Oversold (Consider Buying)"
    elif rsi > 70:
        return "Overbought (Consider Selling)"
    else:
        return "Neutral"

def plot_correlation_heatmap(df_feat):
    corr = df_feat.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_prediction_vs_actual(df_feat, model):
    # Predict past N days (rolling)
    N = 60
    preds = []
    for i in range(len(df_feat) - N, len(df_feat)):
        X = df_feat.drop(columns=['target']).iloc[i:i+1]
        pred = model.predict(X)[0]
        preds.append(pred)
    actuals = df_feat['target'].iloc[-N:].values

    dates = df_feat.index[-N:]

    df_plot = pd.DataFrame({
        'Date': dates,
        'Actual Close': actuals,
        'Predicted Close': preds
    })

    base = alt.Chart(df_plot).encode(x='Date:T')
    actual_line = base.mark_line(color='blue').encode(y='Actual Close:Q')
    pred_line = base.mark_line(color='red').encode(y='Predicted Close:Q')

    chart = alt.layer(actual_line, pred_line).properties(title="Predicted vs Actual Close Prices (Last 60 days)")
    st.altair_chart(chart, use_container_width=True)

if symbol:
    try:
        df_raw = fetch_daily_stock_data(symbol)
        df_feat = generate_features(df_raw)

        model_path = get_latest_model_path(symbol)

        if model_path is None:
            st.warning(f"No trained model found for {symbol}. Please train one first.")
        else:
            model = load_model(model_path)

            # Summary Metrics
            latest_close, pct_change, high_52w, low_52w = calculate_summary_metrics(df_raw)
            rsi_latest = df_feat['rsi'].iloc[-1]
            signal = get_buy_sell_signal(rsi_latest)

            st.markdown(f"### Summary for {symbol}")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Latest Close", f"${latest_close:.2f}", f"{pct_change:.2f}%")
            col2.metric("52-week High", f"${high_52w:.2f}")
            col3.metric("52-week Low", f"${low_52w:.2f}")
            #col4.metric("Avg Volume (20 days)", f"{avg_volume:,.0f}")
            col4.metric("RSI", f"{rsi_latest:.2f}", signal)

            # Tabs for UI
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Prediction",
                "Feature Trends",
                "Feature Correlations",
                "Model Performance",
                "Historical Predictions"
            ])

            with tab1:
                latest_features = df_feat.drop(columns=['target']).iloc[-1:]
                prediction = predict_next(model, latest_features)
                last_date = df_raw.index.max()
                next_trading_day = last_date + BDay(1)

                st.success(f"Predicted closing price for {symbol} on {next_trading_day.date()}: ${prediction:.2f}")
                st.caption(f"Model used: `{os.path.basename(model_path)}`")

                # Interactive Altair line chart for close price + prediction
                df_chart = df_raw[['close']].tail(60).copy()
                pred_df = pd.DataFrame({
                    'close': [prediction],
                    'date': [next_trading_day]
                }).set_index('date')
                df_chart = pd.concat([df_chart, pred_df])
                df_chart.reset_index(inplace=True)
                df_chart.columns = ['date', 'close']

                line = alt.Chart(df_chart).mark_line(point=True).encode(
                    x='date:T',
                    y='close:Q',
                    tooltip=['date:T', 'close:Q']
                ).properties(title=f"{symbol} Close Price + Prediction")

                prediction_line = alt.Chart(pred_df.reset_index()).mark_point(color='red', size=100).encode(
                    x='date:T',
                    y='close:Q',
                    tooltip=['date:T', 'close:Q']
                )

                st.altair_chart(line + prediction_line, use_container_width=True)

            with tab2:
                available_features = ['close', 'sma_5', 'ema_5', 'rsi', 'macd', 'volatility', 'return']
                selected_features = st.multiselect(
                    "Select features to plot",
                    options=available_features,
                    default=['close', 'sma_5', 'ema_5']
                )
                if selected_features:
                    st.line_chart(df_feat[selected_features].tail(60))
                else:
                    st.info("Please select at least one feature.")

            with tab3:
                st.subheader("Feature Correlation Heatmap")
                plot_correlation_heatmap(df_feat)

            with tab4:
                st.subheader("Model Performance Over Time")
                # Dummy example: show RMSE metric stored alongside model or elsewhere
                # For now, show single RMSE from last training
                # You could load a metrics file or database here
                st.info("Currently showing last training RMSE only.")
                # Placeholder: you can enhance to load RMSE history
                st.write("RMSE: (example) 2.27")

            with tab5:
                st.subheader("Historical Predictions vs Actual")
                plot_prediction_vs_actual(df_feat, model)

            # Download feature data CSV
            csv = df_feat.to_csv().encode('utf-8')
            st.download_button(
                label="Download Feature Data CSV",
                data=csv,
                file_name=f"{symbol}_features.csv",
                mime="text/csv"
            )

            # You can also add News API integration or sentiment analysis here
            # or add custom training UI if you want to extend further.

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please enter a stock symbol to start.")