#!/bin/bash
echo "Training model..."
python src/model_train.py

echo "Starting app..."
streamlit run app/main.py