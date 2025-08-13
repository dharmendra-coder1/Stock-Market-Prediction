# Stock Price Prediction Web App (Flask + LSTM)
**Predict next trading day’s close price using an LSTM model.**  
**Outputs**: interactive chart (HTML) + today/predicted CSV.

🔗 **View Project** (https://drive.google.com/file/d/1_RlTxRA1_32yvMaFLcpnLk4idh0l7MiY/view?usp=sharing)

## What it does
- Downloads stock data from Yahoo Finance
- Trains an LSTM on the Close price (60-day window)
- Predicts the next trading day (skips weekends)
- Saves:
  - `static/interactive_chart.html`
  - `static/forecast.csv`

## Why this project
Quick way to forecast and visualize stock price movement, perfect for learning ML + Flask and showcasing skills.


## Why LSTM is the Best Choice
Handles Sequential Data – LSTM networks are specifically designed to process and predict time-series data like stock prices.

Remembers Long-Term Patterns – Unlike regular neural networks, LSTMs retain past information for longer periods, which is crucial for understanding market trends.

Reduces Data Loss – LSTMs solve the vanishing gradient problem, making them more accurate for longer historical datasets.

Proven Accuracy – Widely used in financial forecasting because they adapt well to market fluctuations.

## Tech Stack
Flask, yfinance, pandas, numpy, scikit-learn, TensorFlow/Keras (LSTM), Plotly.

| Module                 | Purpose                                                |
| ---------------------- | ------------------------------------------------------ |
| **Flask**              | Backend framework to create the web application.       |
| **yfinance**           | Fetches real-time and historical stock market data.    |
| **pandas**             | Data manipulation and preprocessing for analysis.      |
| **numpy**              | Numerical operations for LSTM model input preparation. |
| **matplotlib**         | Static data visualization for stock price charts.      |
| **plotly**             | Interactive data visualization for better insights.    |
| **scikit-learn**       | Data scaling and accuracy score calculation.           |
| **tensorflow / keras** | Building and training the LSTM model.                  |
| **datetime**           | Handling and formatting date inputs.                   |



## Features
Input stock symbol and custom date range.

Predict tomorrow’s stock price with visual and text output.

Interactive charts for historical and predicted data.

Displays prediction accuracy percentage.

Automatically saves forecast results (forecast.csv) and chart (interactive_chart.html) in the static folder.

## Conclusion
This project demonstrates how AI and deep learning can be used to forecast stock prices and help in decision-making. The integration of LSTM with Flask makes it interactive, user-friendly, and deployable on the web for real-world use.
