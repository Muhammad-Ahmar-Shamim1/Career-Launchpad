@echo off
REM Script to run Streamlit app for Sentiment Analysis

echo.
echo ================================
echo Sentiment Analysis - Streamlit App
echo ================================
echo.

REM Check if model exists
if not exist "models\bert_sentiment_model" (
    echo ERROR: Model not found!
    echo Please train the model first by running:
    echo   python -m src.train_model
    echo.
    pause
    exit /b 1
)

echo Starting Streamlit app...
echo.

REM Run Streamlit
python -m streamlit run src/streamlit_app.py

pause
