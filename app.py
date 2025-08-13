from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime, timedelta
import os

app = Flask(__name__)

def prepare_data(data, window_size=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    x, y = [], []
    for i in range(window_size, len(scaled_data)):
        x.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler, scaled_data

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol']
        from_date = request.form['from_date']
        to_date_input = request.form['to_date']

        try:
            to_date_obj = pd.to_datetime(to_date_input)
            adjusted_to_date = (to_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')

            stock = yf.download(symbol, start=from_date, end=adjusted_to_date)

            if stock.empty:
                return render_template('index.html', error="Invalid stock symbol or date range.")
        except Exception as e:
            return render_template('index.html', error=str(e))

        data = stock[['Close']].copy()
        x, y, scaler, scaled_data = prepare_data(data)

        model = build_model((x.shape[1], 1))
        model.fit(x, y, epochs=5, batch_size=32, verbose=0)

        # Predict next trading day
        last_60 = scaled_data[-60:]
        input_seq = last_60.reshape(1, 60, 1)
        pred_scaled = model.predict(input_seq)[0][0]
        tomorrow_price = scaler.inverse_transform([[pred_scaled]])[0][0]

        predicted_scaled = model.predict(x)
        predicted_prices = scaler.inverse_transform(predicted_scaled)
        actual_prices = scaler.inverse_transform(y.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        accuracy = 100 - (rmse / np.mean(actual_prices) * 100)

        today_price = data['Close'].iloc[-1]
        today_date = data.index[-1]

        # ✅ Skip weekends
        tomorrow_date = today_date + timedelta(days=1)
        while tomorrow_date.weekday() >= 5:
            tomorrow_date += timedelta(days=1)

        # ✅ Final fixed code: prevent tolist() error
        stock_df = stock.to_frame() if isinstance(stock, pd.Series) else stock
        prices = stock_df['Close'].values.flatten().tolist()  # safe and robust
        dates = stock_df.index.tolist()

        # ✅ Interactive Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers', name='Historical'))

        fig.add_trace(go.Scatter(
            x=[tomorrow_date], y=[tomorrow_price],
            mode='markers+text',
            name='Predicted Tomorrow',
            marker=dict(color='red', size=10),
            text=[f"{tomorrow_price:.2f}"],
            textposition="top center"
        ))

        fig.update_layout(
            title=f"{symbol.upper()} Price Forecast (Zoom & Hover Enabled)",
            xaxis_title="Date",
            yaxis_title="Price",
            autosize=True,
            template="plotly_white"
        )

        os.makedirs('static', exist_ok=True)
        chart_path = os.path.join(app.root_path, 'static', 'interactive_chart.html')
        pio.write_html(fig, file=chart_path, auto_open=False)

        # Save CSV
        csv_data = pd.DataFrame({
            'Date': [today_date.strftime('%Y-%m-%d'), tomorrow_date.strftime('%Y-%m-%d')],
            'Price': [round(today_price, 2), round(tomorrow_price, 2)],
            'Type': ['Actual (Today)', 'Predicted (Next Trading Day)'],
            'Accuracy (%)': [round(accuracy, 2), '']
        })
        csv_path = os.path.join(app.root_path, 'static', 'forecast.csv')
        csv_data.to_csv(csv_path, index=False)

        return render_template('result.html',
                               symbol=symbol.upper(),
                               today_price=round(today_price, 2),
                               tomorrow_price=round(tomorrow_price, 2),
                               today_date=today_date.strftime("%Y-%m-%d"),
                               tomorrow_date=tomorrow_date.strftime("%Y-%m-%d"),
                               accuracy=round(accuracy, 2),
                               csv_file='forecast.csv',
                               chart_file='interactive_chart.html')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
