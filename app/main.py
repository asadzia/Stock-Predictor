import os
import sys

# Set project root so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import re
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from src.fetch_data import fetch_daily_stock_data
from src.features import generate_features
from src.predictor import load_model, predict_next
from pandas.tseries.offsets import BDay
from src.llm_utils import ask_llm
from src.llm_wrapper import get_prediction_with_context
from src.predictor import get_latest_model_path

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("🔮 AI Stock Price Predictor")

symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()

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

with st.sidebar.expander("ℹ️ About this app", expanded=False):
    st.markdown("""
    This is an AI-powered stock predictor. It:
    - Uses historical data and technical indicators
    - Predicts the next trading day's close price using a machine learning model
    - Allows natural language Q&A using GPT-4
    """)

# Tabs for App
tabs = st.tabs([
    "📈 Stock Prediction",
    "🧠 LLM Chat Assistant"
])

# ========== TAB 1: STOCK PREDICTION ==========
with tabs[0]:
    if symbol:
        try:
            df_raw = fetch_daily_stock_data(symbol)
            df_feat = generate_features(df_raw)

            model_path = get_latest_model_path(symbol)

            if model_path is None:
                st.warning(f"No trained model found for {symbol}. Please train one first.")
            else:
                model = load_model(model_path)

                latest_close, pct_change, high_52w, low_52w = calculate_summary_metrics(df_raw)
                rsi_latest = df_feat['rsi'].iloc[-1]
                signal = get_buy_sell_signal(rsi_latest)

                st.markdown(f"### Summary for {symbol}")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Latest Close", f"${latest_close:.2f}", f"{pct_change:.2f}%")
                col2.metric("52-week High", f"${high_52w:.2f}")
                col3.metric("52-week Low", f"${low_52w:.2f}")
                col4.metric("RSI", f"{rsi_latest:.2f}", signal)

                sub_tabs = st.tabs([
                    "Prediction",
                    "Feature Trends",
                    "Feature Correlations",
                    "Model Performance",
                    "Historical Predictions"
                ])

                with sub_tabs[0]:
                    latest_features = df_feat.drop(columns=['target']).iloc[-1:]
                    prediction = predict_next(model, latest_features)
                    last_date = df_raw.index.max()
                    next_trading_day = last_date + BDay(1)

                    st.success(f"Predicted close for {symbol} on {next_trading_day.date()}: ${prediction:.2f}")
                    st.caption(f"Model: `{os.path.basename(model_path)}`")

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

                with sub_tabs[1]:
                    features = ['close', 'sma_5', 'ema_5', 'rsi', 'macd', 'volatility', 'return']
                    selected = st.multiselect("Plot features", options=features, default=['close', 'sma_5'])
                    if selected:
                        st.line_chart(df_feat[selected].tail(60))
                    else:
                        st.info("Select features to view chart.")

                with sub_tabs[2]:
                    plot_correlation_heatmap(df_feat)

                with sub_tabs[3]:
                    st.info("Showing RMSE from last training only.")
                    st.write("RMSE: (example) 2.27")

                with sub_tabs[4]:
                    plot_prediction_vs_actual(df_feat, model)

                csv = df_feat.to_csv().encode('utf-8')
                st.download_button("📥 Download CSV", data=csv, file_name=f"{symbol}_features.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Enter a stock symbol to begin.")

# ========== TAB 2: CHAT ASSISTANT ==========
with tabs[1]:
    st.markdown("## 💬 Ask Anything About a Stock")
    user_question = st.text_input("e.g. What's the forecast for NVDA tomorrow?")

    if user_question:
        # find words of length 1-5 letters, regardless of case, then uppercase and pick first
        possible_tickers = re.findall(r"\b[a-zA-Z]{1,5}\b", user_question)
        # Filter candidates to uppercase and exclude common stopwords
        stopwords = {"WHAT", "WHATS", "THE", "FOR", "IS", "OF", "ON", "TO", "A", "AN", "AND", "IN", "IT", "BY", "YOU"}
        candidates = [w.upper() for w in possible_tickers if w.upper() not in stopwords]
        symbol = candidates[0] if candidates else None

        if not symbol:
            st.warning("Couldn't find a valid stock ticker in your question.")
        else:
            st.write(symbol)
            context = get_prediction_with_context(symbol)
            prompt = f"""
            You are a helpful AI assistant for stock prediction.
            Based on the following context and features, answer the user's question:

            Context:
            {context}

            Question:
            {user_question}
            """

            answer = ask_llm(prompt)
            st.markdown("### 🤖 Assistant Response")
            st.success(answer)