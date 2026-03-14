import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#Configuring app page
st.set_page_config(page_title="Predicting XRP Price", page_icon="📈", layout="wide")
st.title("Predicting Ripple (XRP) Price")
st.markdown("""
This web app pulls the latest data for **Ripple (XRP)**, calculates key technical indicators, 
and utilizes a Deep Learning **LSTM Neural Network** to predict tomorrow's closing price.
""")

#cache function for 1 hour
@st.cache_data(ttl=3600)
def load_data():
    #fetching price for Ripple, Bitcoin, and S&P500
    xrp = yf.download('XRP-USD', period='5y')[['Close']]
    btc = yf.download('BTC-USD', period='5y')[['Close']]
    sp500 = yf.download('^GSPC', period='5y')[['Close']]
    
    #mergeing columns
    combined = xrp.rename(columns={'Close': 'XRP_Close'})
    combined['BTC'] = btc['Close']
    combined['SP500'] = sp500['Close']

    #forwardfill weekends for S&P 500
    combined = combined.ffill().dropna()
    
    #feature engineering -moving avareages and MACD
    combined['SMA_7'] = combined['XRP_Close'].rolling(window=7).mean()
    combined['SMA_30'] = combined['XRP_Close'].rolling(window=30).mean()
    
    ema_12 = combined['XRP_Close'].ewm(span=12, adjust=False).mean()
    ema_26 = combined['XRP_Close'].ewm(span=26, adjust=False).mean()
    combined['MACD'] = ema_12 - ema_26
    
    combined.dropna(inplace=True)
    return combined

#LSTM
@st.cache_resource
def train_and_predict(df):
    #scaling data
    dataset = df[['XRP_Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    #looking back 14 days
    time_steps = 14
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:(i + time_steps)])
        y.append(scaled_data[i + time_steps])
        
    X, y = np.array(X), np.array(y)
    
    #LSTM optimized for UI on app
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    #train recent data
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    
    #predict from last 14 days
    last_14_days = scaled_data[-time_steps:]
    last_14_days_reshaped = np.reshape(last_14_days, (1, time_steps, 1))
    
    predicted_scaled = model.predict(last_14_days_reshaped)
    predicted_price = scaler.inverse_transform(predicted_scaled)
    
    return predicted_price[0][0]

with st.spinner('Fetching live market data and updating model...'):
    df = load_data()
    tomorrow_prediction = train_and_predict(df)

#configuring UI interactivity
#Sidebar Toggles
st.sidebar.header("Market Correlation")
show_btc = st.sidebar.checkbox("Compare with Bitcoin", value=False)
show_sp500 = st.sidebar.checkbox("Compare with S&P 500", value=False)

st.sidebar.header("Chart Controls")
show_sma7 = st.sidebar.checkbox("Show 7-Day SMA", value=True)
show_sma30 = st.sidebar.checkbox("Show 30-Day SMA", value=False)
days_to_show = st.sidebar.slider("Days of History to Display", min_value=30, max_value=365, value=90)

#filtering dataframe on slider
df_display = df.tail(days_to_show)

# displaying prediction metric
#forcing values into standard floats to by pass yfinance MultiIndex formatting errors
current_price = float(df['XRP_Close'].values.flatten()[-1])
tomorrow_prediction = float(tomorrow_prediction)
price_diff = tomorrow_prediction - current_price

st.metric(
    label="Tomorrow's Predicted Price (LSTM)", 
    value=f"${tomorrow_prediction:.4f}", 
    delta=f"${price_diff:.4f} from today's close"
)

fig = go.Figure()

#actual rpice
fig.add_trace(go.Scatter(x=df_display.index, y=df_display['XRP_Close'], mode='lines', name='Actual Price', line=dict(color='black', width=2)))

if show_btc:
    #show secondary Y-axis
    btc_norm = df_display['BTC'] / df_display['BTC'].iloc[0] * df_display['XRP_Close'].iloc[0]
    fig.add_trace(go.Scatter(x=df_display.index, y=btc_norm, name='BTC (Scaled)', line=dict(color='orange', dash='dash')))

if show_sp500:
    sp_norm = df_display['SP500'] / df_display['SP500'].iloc[0] * df_display['XRP_Close'].iloc[0]
    fig.add_trace(go.Scatter(x=df_display.index, y=sp_norm, name='S&P 500 (Scaled)', line=dict(color='green', dash='dash')))

if show_sma7:
    fig.add_trace(go.Scatter(x=df_display.index, y=df_display['SMA_7'], mode='lines', name='7-Day SMA', line=dict(color='blue', dash='dot')))
if show_sma30:
    fig.add_trace(go.Scatter(x=df_display.index, y=df_display['SMA_30'], mode='lines', name='30-Day SMA', line=dict(color='orange', dash='dot')))

fig.update_layout(
    title=f"XRP Price History (Last {days_to_show} Days)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)