import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'symbol' not in st.session_state:
    st.session_state.symbol = "TSLA"
if 'start_date' not in st.session_state:
    st.session_state.start_date = pd.to_datetime("2025-01-01")
if 'end_date' not in st.session_state:
    st.session_state.end_date = pd.to_datetime("today")

st.title("Alan's Trading Strategy Optimizer")
st.markdown("Fetch historical stock data, analyze trends, and optimize your trading strategy using AI!")

# User input for stock symbol and date range
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", st.session_state.symbol)
start_date = st.date_input("Start Date", st.session_state.start_date)
end_date = st.date_input("End Date", st.session_state.end_date)

# Update session state with user inputs
st.session_state.symbol = symbol
st.session_state.start_date = start_date
st.session_state.end_date = end_date

# Fetch data using yfinance
if st.button("Show Data"):
    st.session_state.df = yf.download(symbol, start=start_date, end=end_date)
    if st.session_state.df.empty:
        st.error("No data found for the given stock symbol and date range.")
        st.session_state.df = None
    else:
        st.write("## Raw Data Preview")
        st.dataframe(st.session_state.df.tail(31))

        # Save CSV
        csv = st.session_state.df.to_csv().encode('utf-8')
        st.download_button("Download CSV", csv, f"{symbol}_data.csv", "text/csv")

        # Plot closing price using Matplotlib
        st.write("## Price Chart")
        fig, ax = plt.subplots()
        ax.plot(st.session_state.df.index, st.session_state.df['Close'], label='Closing Price', color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=90)
        ax.grid(True)
        st.pyplot(fig)

# Strategy selection and calculations
if st.session_state.df is not None and not st.session_state.df.empty:
    st.write("## Choose Trading Strategy")
    strategy = st.selectbox("Select a strategy", [
        "移動平均交叉 (Moving Average Crossover)", "相對強弱指數 (RSI)", "布林帶 (Bollinger Bands)", "移動平均匯聚背離 (MACD)",
        "隨機震盪器 (Stochastic Oscillator)", "斐波那契回撤 (Fibonacci Retracement)", "平均真實範圍 (Average True Range)"
    ])

    df = st.session_state.df.copy()  # Work on a copy

    if strategy == "移動平均交叉 (Moving Average Crossover)":
        short_window = st.number_input("Short Moving Average Window", min_value=5, max_value=50, value=10)
        long_window = st.number_input("Long Moving Average Window", min_value=20, max_value=200, value=50)
        df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
        df['Long_MA'] = df['Close'].rolling(window=long_window).mean()

        st.write("**說明**: 使用短期和長期移動平均線（MA）來識別趨勢變化。短期MA（如10天）反映短期價格動態，長期MA（如50天）顯示長期趨勢。")
        st.write("**買入信號**: 當短期MA從下方穿越長期MA時，表明上升趨勢，建議買入。")
        st.write("**賣出信號**: 當短期MA從上方穿越長期MA時，表明下降趨勢，建議賣出。")
        st.write("## Moving Average Crossover Plot")
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'], label='Closing Price', color='blue')
        ax.plot(df.index, df['Short_MA'], label=f'{short_window}-day MA', color='red')
        ax.plot(df.index, df['Long_MA'], label=f'{long_window}-day MA', color='green')
        ax.legend()
        plt.xticks(rotation=90)
        ax.grid(True)
        st.pyplot(fig)

    elif strategy == "相對強弱指數 (RSI)":
        period = st.number_input("RSI Period", min_value=5, max_value=50, value=14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        st.write("**說明**: RSI衡量價格動量，範圍0-100，通常使用14天周期。RSI高於70表示超買，低於30表示超賣。")
        st.write("**買入信號**: 當RSI跌破30（超賣）時，價格可能反彈，建議買入。")
        st.write("**賣出信號**: 當RSI突破70（超買）時，價格可能回落，建議賣出。")
        st.write("## RSI Plot")
        fig, ax = plt.subplots()
        ax.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax.axhline(70, linestyle='--', color='red', alpha=0.5, label='Overbought (70)')
        ax.axhline(30, linestyle='--', color='green', alpha=0.5, label='Oversold (30)')
        ax.set_xlabel("Date")
        ax.set_ylabel("RSI")
        ax.legend()
        plt.xticks(rotation=90)
        ax.grid(True)
        st.pyplot(fig)

    elif strategy == "布林帶 (Bollinger Bands)":
        window = st.number_input("Bollinger Bands Window", min_value=5, max_value=50, value=20)
        num_std = st.number_input("Number of Standard Deviations", min_value=1.0, max_value=3.0, value=2.0)
        df['Middle_Band'] = df['Close'].rolling(window=window).mean()
        df['Std_Dev'] = df['Close'].rolling(window=window).std()
        df['Upper_Band'] = df['Middle_Band'] + (df['Std_Dev'] * num_std)
        df['Lower_Band'] = df['Middle_Band'] - (df['Std_Dev'] * num_std)

        st.write("**說明**: 布林帶由20天移動平均線（中間帶）和上下兩條標準差帶（±2σ）組成，反映價格波動性。")
        st.write("**買入信號**: 當價格觸及下帶時，表示可能超賣，建議買入。")
        st.write("**賣出信號**: 當價格觸及上帶時，表示可能超買，建議賣出。")
        st.write("## Bollinger Bands Plot")
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'], label='Closing Price', color='blue')
        ax.plot(df.index, df['Middle_Band'], label='Middle Band (SMA)', color='black')
        ax.plot(df.index, df['Upper_Band'], label='Upper Band', color='red')
        ax.plot(df.index, df['Lower_Band'], label='Lower Band', color='green')
        ax.fill_between(df.index, df['Upper_Band'], df['Lower_Band'], color='gray', alpha=0.1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=90)
        ax.grid(True)
        st.pyplot(fig)

    elif strategy == "移動平均匯聚背離 (MACD)":
        short_ema = st.number_input("Short EMA Period", min_value=5, max_value=50, value=12)
        long_ema = st.number_input("Long EMA Period", min_value=10, max_value=100, value=26)
        signal_period = st.number_input("Signal Line Period", min_value=5, max_value=20, value=9)
        df['Short_EMA'] = df['Close'].ewm(span=short_ema, adjust=False).mean()
        df['Long_EMA'] = df['Close'].ewm(span=long_ema, adjust=False).mean()
        df['MACD'] = df['Short_EMA'] - df['Long_EMA']
        df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()

        st.write("**說明**: MACD由12天和26天指數移動平均線（EMA）差值及9天信號線組成，反映趨勢動量。")
        st.write("**買入信號**: 當MACD線從下方穿越信號線時，表示看漲，建議買入。")
        st.write("**賣出信號**: 當MACD線從上方穿越信號線時，表示看跌，建議賣出。")
        st.write("## MACD Plot")
        fig, ax = plt.subplots()
        ax.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax.plot(df.index, df['Signal_Line'], label='Signal Line', color='red')
        ax.bar(df.index, df['MACD'] - df['Signal_Line'], label='Histogram', color='gray', alpha=0.3)
        ax.set_xlabel("Date")
        ax.set_ylabel("MACD")
        ax.legend()
        plt.xticks(rotation=90)
        ax.grid(True)
        st.pyplot(fig)

    elif strategy == "隨機震盪器 (Stochastic Oscillator)":
        period = st.number_input("Stochastic Period", min_value=5, max_value=50, value=14)
        smooth_k = st.number_input("Smooth %K", min_value=1, max_value=10, value=3)
        smooth_d = st.number_input("Smooth %D", min_value=1, max_value=10, value=3)
        low = df['Low'].rolling(window=period).min()
        high = df['High'].rolling(window=period).max()
        df['%K'] = 100 * (df['Close'] - low) / (high - low)
        df['%K'] = df['%K'].rolling(window=smooth_k).mean()
        df['%D'] = df['%K'].rolling(window=smooth_d).mean()

        st.write("**說明**: 隨機震盪器比較收盤價與14天價格範圍，計算%K和%D線，範圍0-100。")
        st.write("**買入信號**: 當%K在20以下從下方穿越%D時，表示超賣，建議買入。")
        st.write("**賣出信號**: 當%K在80以上從上方穿越%D時，表示超買，建議賣出。")
        st.write("## Stochastic Oscillator Plot")
        fig, ax = plt.subplots()
        ax.plot(df.index, df['%K'], label='%K', color='blue')
        ax.plot(df.index, df['%D'], label='%D', color='red')
        ax.axhline(80, linestyle='--', color='red', alpha=0.5, label='Overbought (80)')
        ax.axhline(20, linestyle='--', color='green', alpha=0.5, label='Oversold (20)')
        ax.set_xlabel("Date")
        ax.set_ylabel("Stochastic")
        ax.legend()
        plt.xticks(rotation=90)
        ax.grid(True)
        st.pyplot(fig)

    elif strategy == "斐波那契回撤 (Fibonacci Retracement)":
        high_price = df['Close'].max().item()  # Convert to scalar
        low_price = df['Close'].min().item()   # Convert to scalar
        diff = high_price - low_price
        levels = {
            '100%': high_price,
            '61.8%': high_price - 0.618 * diff,
            '50%': high_price - 0.5 * diff,
            '38.2%': high_price - 0.382 * diff,
            '23.6%': high_price - 0.236 * diff,
            '0%': low_price
        }

        st.write("**說明**: 根據價格區間繪製回撤水平（23.6%、38.2%、50%、61.8%、100%），識別支撐與阻力。")
        st.write("**買入信號**: 當價格接近支撐水平（如38.2%、50%）時，建議買入。")
        st.write("**賣出信號**: 當價格接近阻力水平（如61.8%、100%）時，建議賣出。")
        st.write("## Fibonacci Retracement Plot")
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'], label='Closing Price', color='blue')
        for level, price in levels.items():
            ax.axhline(price, linestyle='--', label=f'{level} ({price:.2f})', alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=90)
        ax.grid(True)
        st.pyplot(fig)

    elif strategy == "平均真實範圍 (Average True Range)":
        period = st.number_input("ATR Period", min_value=5, max_value=50, value=14)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=period).mean()

        st.write("**說明**: ATR衡量14天內價格波動性，幫助設定止損點。高ATR表示高波動。")
        st.write("**買入信號**: ATR不直接提供買賣信號，需結合其他指標；高ATR建議設置較寬止損。")
        st.write("**賣出信號**: 用於風險管理，調整買賣策略的止損範圍。")
        st.write("## ATR Plot")
        fig, ax = plt.subplots()
        ax.plot(df.index, df['ATR'], label='ATR', color='orange')
        ax.set_xlabel("Date")
        ax.set_ylabel("ATR")
        ax.legend()
        plt.xticks(rotation=90)
        ax.grid(True)
        st.pyplot(fig)

    st.success("Strategy applied successfully!")
else:
    st.info("Please fetch data by clicking 'Show Data' to proceed with strategy analysis.")