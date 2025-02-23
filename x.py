import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from tensorflow.keras.callbacks import EarlyStopping

# Streamlit App Title
st.title("📊  Stock Price Prediction & Trend Clustering Using GRU and K-Means")

# Select Stock Ticker
stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")

# Fetch Stock Data
def fetch_stock_data(stock_ticker):
    try:
        data = yf.download(stock_ticker, period='10y', interval='1d')
        if data.empty:
            st.error("No data found! Please check the stock symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Add Technical Indicators
def add_features(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).mean() / data['Close'].pct_change().rolling(14).std()))
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()
    data['Volume_Change'] = data['Volume'].pct_change()
    data.dropna(inplace=True)
    return data

# Load Data
data = fetch_stock_data(stock)
if data is not None:
    data = add_features(data)
    st.write("### Stock Data (Last 10 Days)")
    st.dataframe(data.tail(10))
    fig4 = px.line(data, x=data.index, y='Volatility', title="📊 Volatility Over Time")
    st.plotly_chart(fig4)
    # Clustering Market Trends using K-Means
    data['Close_1D'] = data['Close'].values.flatten()  

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data['Trend_Cluster'] = kmeans.fit_predict(data[['Close_1D']])  

    # Convert Trend_Cluster to categorical for Plotly
    data['Trend_Cluster'] = data['Trend_Cluster'].astype(str)

    # Create Scatter Plot for Clustering
    fig6 = px.scatter(
        data, x=data.index, y='Close_1D', 
        color='Trend_Cluster',  
        title="📊 K-Means Clustering of Stock Trends",
        labels={"Trend_Cluster": "Cluster"},
        template="plotly_dark"
    )
    st.plotly_chart(fig6)

    # Data Preprocessing
    features = ['Close', 'SMA_50', 'SMA_200', 'EMA_50', 'RSI', 'MACD', 'Volatility', 'Volume_Change']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    # Prepare Sequences for GRU
    seq_length = 40  # Reduced from 60 to 40 for faster training
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i][0])
    X, y = np.array(X), np.array(y)

    # Train-Test Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Optimized GRU Model
    model = Sequential([
        GRU(64, activation='tanh', return_sequences=True, input_shape=(seq_length, X.shape[2])),  # CuDNN optimization
        Dropout(0.05),  # Reduced dropout for faster convergence
        GRU(64, activation='tanh'),  
        Dropout(0.05),
        Dense(1)
    ])

    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse')  # RMSprop is better for RNNs
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stop])  # Reduced epochs & increased batch size

    # Evaluate Model
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], X.shape[2] - 1)))))[:, 0]
    y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X.shape[2] - 1)))))[:, 0]

    mae = mean_absolute_error(y_test_actual, predictions)
    mse = mean_squared_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, predictions)

    st.write("### Model Evaluation")
    st.write(f"📌 **Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"📌 **Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"📌 **Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"📌 **R² Score:** {r2:.2f}")

    # Predict Next Day Price
    last_sequence = scaled_data[-seq_length:]
    prediction = model.predict(last_sequence.reshape(1, seq_length, X.shape[2]))
    predicted_price = scaler.inverse_transform(np.hstack((prediction, np.zeros((1, X.shape[2] - 1)))))[:, 0][0]

    st.write(f"## 📌 Predicted Next-Day {stock} Price: **${predicted_price:.2f}**")

    # Plot Actual vs Predicted Prices
    fig = px.line(title=f'📈 Actual vs Predicted {stock} Prices')
    fig.add_scatter(x=data.index[-len(y_test_actual):], y=y_test_actual, mode='lines', name='Actual')
    fig.add_scatter(x=data.index[-len(predictions):], y=predictions, mode='lines', name='Predicted')
    st.plotly_chart(fig)

else:
    st.error("⚠️ Could not retrieve data. Please check your internet connection or try again later.")
