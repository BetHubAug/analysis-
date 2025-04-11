# Combined Market Analysis Tool (Gradio App)
# Integrates yfinance data, Prophet forecasting, basic local ML,
# placeholder sentiment, and optional ArcheanVision API data.

import os
import warnings
import logging
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import cloudscraper # For ArcheanVision API
from dotenv import load_dotenv # To load API key from .env file
import gradio as gr

# --- Environment & Configuration ---

# Load environment variables from .env file (if it exists)
load_dotenv()
ARCHEANVISION_API_KEY = os.environ.get("ARCHEANVISION_API_KEY") # Optional

# Suppress warnings (optional, adjust as needed)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("prophet").setLevel(logging.ERROR) # Silence Prophet logs
logging.getLogger("cmdstanpy").setLevel(logging.ERROR) # Silence cmdstanpy logs
logging.getLogger("numexpr").setLevel(logging.ERROR) # Silence numexpr logs


# --- Constants ---
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD"]
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
# Note: yfinance interval support varies. '1h' often requires fetching more data.
# Stocks generally only have daily ('1d') reliable historical data via yfinance.
INTERVAL_OPTIONS = ["1d", "1h", "4h"] # Added 4h as example, adjust based on yfinance limitations

# --- Data Fetching Functions ---

def fetch_yfinance_data(symbol, interval="1d", period="1y"):
    """Fetch market data using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        # Adjust period based on interval for better data coverage
        if "h" in interval:
            # Fetch more granular data for hourly intervals, yfinance has limitations here
             # Max period for 1h is 730 days, use less for speed
            df = ticker.history(period="60d", interval=interval)
        elif interval == "1d":
            df = ticker.history(period=period, interval=interval)
        else: # Default to daily if interval is unknown
             df = ticker.history(period=period, interval="1d")

        if df.empty:
           raise ValueError("No data returned from yfinance.")

        df.reset_index(inplace=True)
        # Standardize column names
        df.rename(columns={
            df.columns[0]: "timestamp", # Handle 'Date' or 'Datetime'
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
            }, inplace=True)
        # Ensure correct dtypes
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True) # Ensure timezone aware
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Select and order columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        return df.dropna()
    except Exception as e:
        print(f"Error fetching yfinance data for {symbol} ({interval}): {e}")
        raise Exception(f"yfinance Error: {e}")


# --- ArcheanVision API Functions (Optional) ---
# Uses cloudscraper from the Streamlit example

def get_archean_signals_cloudscraper(api_key, market):
    """Retrieves market signals for the given market using cloudscraper."""
    if not api_key:
        return None, "ArcheanVision API Key not configured."

    # Attempt to map yfinance symbol to ArcheanVision symbol (simple example)
    # This might need adjustment based on ArcheanVision's exact naming convention
    av_market = market.split('-')[0].upper() # e.g., "BTC-USD" -> "BTC"

    print(f"Attempting to fetch ArcheanVision signals for market: {av_market}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/115.0.0.0 Safari/537.36")
    }
    url = f"https://archeanvision.com/api/signals/{av_market}/signals"
    scraper = cloudscraper.create_scraper(delay=10, browser='chrome') # Add delay, specify browser

    try:
        response = scraper.get(url, headers=headers)
        print(f"ArcheanVision API Status Code: {response.status_code}")
        # print(f"ArcheanVision API Response Text: {response.text[:200]}...") # Debug: print start of response

        response.raise_for_status() # Raises an exception for HTTP errors (4xx or 5xx)
        signals_data = response.json()
        if not signals_data:
             return pd.DataFrame(), f"No signals found from ArcheanVision for {av_market}."
        df_signals = pd.DataFrame(signals_data)
        # Basic processing matching the Streamlit example
        if 'date' in df_signals.columns:
             df_signals['date'] = pd.to_datetime(df_signals['date'], unit='s', errors='coerce').dt.tz_localize('UTC')
             df_signals = df_signals.sort_values('date', ascending=False)
        # Convert dict columns to string for display in Gradio DataFrame
        for col in df_signals.columns:
            if df_signals[col].apply(lambda x: isinstance(x, dict)).any():
                df_signals[col] = df_signals[col].astype(str)
        return df_signals, f"Successfully fetched {len(df_signals)} signals for {av_market}."

    except requests.exceptions.RequestException as e:
        error_msg = f"Network error fetching ArcheanVision signals: {e}"
        print(error_msg)
        # Attempt to get more specific Cloudflare info if possible
        if hasattr(response, 'status_code') and 500 <= response.status_code < 600 :
             error_msg += f" (Server Error {response.status_code})"
        elif hasattr(response, 'status_code') and response.status_code == 403:
             error_msg += " (Forbidden - Possible Cloudflare challenge or IP block)"
        elif hasattr(response, 'status_code'):
             error_msg += f" (Status: {response.status_code})"

        # Include response text if available and seems like Cloudflare page
        if hasattr(response, 'text') and ('cloudflare' in response.text.lower() or 'checking your browser' in response.text.lower()):
             error_msg += " - Cloudflare protection likely active."
             # print(f"Cloudflare Response Snippet: {response.text[:500]}") # Debugging

        return pd.DataFrame(), error_msg
    except Exception as e:
        error_msg = f"Error processing ArcheanVision response: {e}"
        print(error_msg)
        return pd.DataFrame(), error_msg


# --- Sentiment Analysis Function (Placeholder) ---

def fetch_sentiment_data(keyword):
    """Analyze sentiment from social media (BASIC PLACEHOLDER)."""
    if not keyword:
        return 0, "No keyword provided."
    try:
        # **Replace this with a real sentiment analysis API or method**
        # This is just a simple demo using TextBlob on hardcoded examples
        tweets = [
            f"{keyword} looks promising, might buy soon!",
            f"I think {keyword} is overvalued right now.",
            f"{keyword} to the moon! 🚀 #crypto",
            f"Uncertainty around {keyword} regulations.",
            f"Just sold my {keyword}, taking profits.",
            f"Feeling bullish about {keyword} long term.",
        ]
        sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        return round(avg_sentiment, 3), f"Analyzed {len(tweets)} sample texts."
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return 0, f"Sentiment Error: {e}"

# --- Technical Analysis Functions ---

def calculate_technical_indicators(df):
    """Calculates RSI, MACD, and Bollinger Bands."""
    if df.empty or len(df) < 20: # Need enough data for rolling windows
        print("Warning: Not enough data for full TA calculation.")
        # Return df with potentially missing TA columns
        df['RSI'] = np.nan
        df['MACD'] = np.nan
        df['Signal_Line'] = np.nan
        df['MA20'] = np.nan
        df['BB_upper'] = np.nan
        df['BB_lower'] = np.nan
        return df

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
    std_dev = df['close'].rolling(window=20, min_periods=1).std()
    df['BB_upper'] = df['MA20'] + 2 * std_dev
    df['BB_lower'] = df['MA20'] - 2 * std_dev

    return df

def create_technical_charts(df):
    """Creates technical analysis charts (Price/BB, RSI, MACD)."""
    if df.empty:
        return go.Figure(), go.Figure(), go.Figure() # Return empty figures

    # Price Chart with Candlesticks and Bollinger Bands
    fig_price = go.Figure()
    try:
        fig_price.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Price'
        ))
        fig_price.add_trace(go.Scatter(
            x=df['timestamp'], y=df['BB_upper'], name='Upper BB',
            line=dict(color='rgba(150, 150, 150, 0.5)', dash='dash'), showlegend=False))
        fig_price.add_trace(go.Scatter(
            x=df['timestamp'], y=df['BB_lower'], name='Lower BB', fill='tonexty',
            fillcolor='rgba(150, 150, 150, 0.1)', line=dict(color='rgba(150, 150, 150, 0.5)', dash='dash'), showlegend=False))
        fig_price.add_trace(go.Scatter(
            x=df['timestamp'], y=df['MA20'], name='MA20',
            line=dict(color='rgba(255, 165, 0, 0.6)', width=1), showlegend=False))

    except Exception as e:
         print(f"Error creating candlestick/BB chart: {e}")
         # Fallback to simple line chart if candlestick fails
         fig_price.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], name='Close Price'))

    fig_price.update_layout(
        title='Price and Bollinger Bands', xaxis_title='Date', yaxis_title='Price',
        xaxis_rangeslider_visible=False, template="plotly_white", height=350
    )

    # RSI Chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], name='RSI', line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig_rsi.update_layout(
        title='RSI Indicator', xaxis_title='Date', yaxis_title='RSI',
        template="plotly_white", height=250
    )

    # MACD Chart
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD'], name='MACD', line=dict(color='blue')))
    fig_macd.add_trace(go.Scatter(x=df['timestamp'], y=df['Signal_Line'], name='Signal Line', line=dict(color='orange')))
    # Optional: MACD Histogram
    macd_hist = df['MACD'] - df['Signal_Line']
    colors = ['green' if val >= 0 else 'red' for val in macd_hist]
    fig_macd.add_trace(go.Bar(x=df['timestamp'], y=macd_hist, name='Histogram', marker_color=colors))

    fig_macd.update_layout(
        title='MACD', xaxis_title='Date', yaxis_title='Value',
        template="plotly_white", height=250
    )

    return fig_price, fig_rsi, fig_macd

# --- Prophet Forecasting Functions ---

def prepare_data_for_prophet(df):
    """Prepares data for Prophet."""
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    df_prophet = df.rename(columns={"timestamp": "ds", "close": "y"})
    # Ensure ds is datetime and timezone-naive for Prophet
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
    return df_prophet[["ds", "y"]]

def prophet_forecast(df_prophet, periods=10, freq="D", daily_seasonality="auto", weekly_seasonality="auto", yearly_seasonality="auto", seasonality_mode="additive", changepoint_prior_scale=0.05):
    """Performs Prophet forecasting."""
    if df_prophet.empty or len(df_prophet) < 5: # Need minimal data for Prophet
        return pd.DataFrame(), "Not enough data for Prophet forecasting (need >= 5 rows)."

    try:
        model = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        return forecast, ""
    except Exception as e:
        print(f"Prophet forecast error: {e}")
        return pd.DataFrame(), f"Prophet Forecast Error: {e}"

def create_forecast_plot(df_hist, forecast_df):
    """Creates the forecast plot showing history and prediction."""
    if forecast_df.empty:
        return go.Figure().update_layout(title="Forecast Error: No forecast data")

    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df_hist["ds"]), # Use original df_prophet for history
        y=df_hist["y"],
        mode="lines",
        name="Historical Price",
        line=dict(color="black", width=1)
    ))

    # Add forecast line
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(forecast_df["ds"]),
        y=forecast_df["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color="blue", width=2)
    ))

    # Add uncertainty interval
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(forecast_df["ds"]),
        y=forecast_df["yhat_lower"],
        fill=None,
        mode="lines",
        line=dict(width=0),
        showlegend=False # Usually cluttering
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(forecast_df["ds"]),
        y=forecast_df["yhat_upper"],
        fill="tonexty", # Fill area between lower and upper bounds
        mode="lines",
        line=dict(width=0),
        fillcolor='rgba(0, 0, 255, 0.1)', # Light blue fill
        name="Uncertainty Interval"
    ))

    fig.update_layout(
        title="Price Forecast (Historical & Predicted)",
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Basic ML Model Training and Prediction ---
# Global model instance (simplistic approach)
model_rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced') # Use fewer estimators for speed, balanced weights
model_is_trained = False # Flag

def train_predict_model(df):
    """Train the simple RandomForest model and predict trend for the latest point."""
    global model_rf, model_is_trained
    if df.empty or len(df) < 20: # Need some data to train
        print("Warning: Not enough data for ML model training.")
        return "N/A: Insufficient Data", 0.0

    # --- Feature Engineering (VERY Basic) ---
    # Use log returns and lagged features might be better, but keep it simple here.
    df_model = df[['timestamp', 'close', 'volume']].copy()
    df_model['return'] = df_model['close'].pct_change()
    # Simple Target: Price increases significantly in the NEXT period (requires shifting)
    # Target: 1 if NEXT close > current close * 1.01 (e.g., > 1% increase)
    df_model['target'] = (df_model['close'].shift(-1) > df_model['close'] * 1.01).astype(int)

    # Add some lagged features (e.g., previous return, previous volume change)
    df_model['prev_return'] = df_model['return'].shift(1)
    df_model['volume_change'] = df_model['volume'].pct_change().shift(1)
    df_model['rsi_model'] = calculate_technical_indicators(df_model.copy())['RSI'].shift(1) # Add RSI as feature

    # Drop rows with NaN created by diff/shift
    df_model.dropna(inplace=True)

    if len(df_model) < 10:
        print("Warning: Not enough valid rows after feature engineering for ML.")
        model_is_trained = False
        return "N/A: Insufficient Data", 0.0

    # Define features (X) and target (y)
    # Use current values (or lagged) to predict the NEXT period's target
    features = ['close', 'volume', 'prev_return', 'volume_change', 'rsi_model']
    X = df_model[features]
    y = df_model['target']

    # Split data (simple split for demo - TimeseriesSplit is better)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) == 0 or len(y_train) == 0:
         print("Warning: Training data is empty after split.")
         model_is_trained = False
         return "N/A: Training Error", 0.0


    # --- Training ---
    try:
        print(f"Training Random Forest on {len(X_train)} samples...")
        model_rf.fit(X_train, y_train)
        model_is_trained = True
        print("Model trained.")
        # Evaluate (optional, basic accuracy)
        accuracy = model_rf.score(X_test, y_test)
        print(f"Model accuracy on test set: {accuracy:.2f}")
    except Exception as e:
        print(f"Error training model: {e}")
        model_is_trained = False
        return "N/A: Training Error", 0.0

    # --- Prediction ---
    # Predict for the *next* period based on the *latest available* features
    if model_is_trained:
        try:
            latest_features = X.iloc[-1:].values # Get the last row of features used for training
            prediction = model_rf.predict(latest_features)[0]
            proba = model_rf.predict_proba(latest_features)[0][1] # Probability of class 1 (increase)
            trend_label = "Predicted Increase (>1%)" if prediction == 1 else "Predicted No Increase"
            return trend_label, round(proba, 3)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "N/A: Prediction Error", 0.0
    else:
        return "N/A: Model Not Trained", 0.0

# --- Main Analysis Function ---

def analyze_market(market_type, symbol, interval, forecast_steps, daily_seasonality, weekly_seasonality, yearly_seasonality, seasonality_mode, changepoint_prior_scale, sentiment_keyword):
    """Main function to orchestrate data fetching, analysis, and prediction."""
    print(f"\n--- Starting Analysis for {symbol} ({interval}) ---")
    status_messages = []
    df = pd.DataFrame()
    df_prophet = pd.DataFrame()
    forecast_df = pd.DataFrame()
    forecast_plot = go.Figure().update_layout(title="No Data Yet")
    tech_plot = go.Figure().update_layout(title="No Data Yet")
    rsi_plot = go.Figure().update_layout(title="No Data Yet")
    macd_plot = go.Figure().update_layout(title="No Data Yet")
    forecast_data_display = pd.DataFrame()
    growth_label = "N/A"
    growth_proba = 0.0
    sentiment_score = 0.0
    sentiment_msg = ""
    av_signals_df = pd.DataFrame()
    av_signals_msg = "ArcheanVision not checked (no API key or different market)."

    # --- 1. Data Fetching (yfinance) ---
    try:
        df = fetch_yfinance_data(symbol, interval)
        status_messages.append(f"Successfully fetched {len(df)} data points for {symbol} from yfinance.")
        print(f"Fetched {len(df)} rows from yfinance.")
        if df.empty: raise ValueError("DataFrame is empty after fetching.")
    except Exception as e:
        status_messages.append(f"ERROR fetching yfinance data: {e}")
        print(f"yfinance fetch failed: {e}")
        # Return empty plots and dataframes with error message
        error_output = "\n".join(status_messages)
        return (forecast_plot, tech_plot, rsi_plot, macd_plot, forecast_data_display,
                growth_label, growth_proba, sentiment_score, sentiment_msg, av_signals_df,
                av_signals_msg, error_output)

    # --- 2. Technical Analysis ---
    try:
        df = calculate_technical_indicators(df)
        status_messages.append("Calculated Technical Indicators (RSI, MACD, BBands).")
        tech_plot, rsi_plot, macd_plot = create_technical_charts(df)
        print("Created TA charts.")
    except Exception as e:
        status_messages.append(f"ERROR calculating TA or creating charts: {e}")
        print(f"TA calculation/charting failed: {e}")
        # Continue if possible, but charts might be empty


    # --- 3. Prophet Forecasting ---
    try:
        df_prophet = prepare_data_for_prophet(df)
        status_messages.append("Prepared data for Prophet.")
        # Determine frequency for Prophet (H for hourly, D for daily)
        freq = "H" if "h" in interval.lower() else "D"
        if not df_prophet.empty:
            forecast_df, prophet_error = prophet_forecast(
                df_prophet, periods=forecast_steps, freq=freq,
                daily_seasonality=daily_seasonality, weekly_seasonality=weekly_seasonality,
                yearly_seasonality=yearly_seasonality, seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale
            )
            if prophet_error:
                status_messages.append(f"Prophet Warning: {prophet_error}")
                print(f"Prophet issue: {prophet_error}")
            else:
                status_messages.append(f"Generated Prophet forecast for {forecast_steps} periods.")
                forecast_plot = create_forecast_plot(df_prophet, forecast_df)
                # Prepare forecast data for display (future points only)
                forecast_data_display = forecast_df.loc[len(df_prophet):, ["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                forecast_data_display.rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"}, inplace=True)
                forecast_data_display['Date'] = pd.to_datetime(forecast_data_display['Date']).dt.strftime('%Y-%m-%d %H:%M') # Format date
                print("Created Prophet forecast plot and data.")
        else:
            status_messages.append("Skipped Prophet: Not enough data after preparation.")
            print("Skipped Prophet forecasting.")

    except Exception as e:
        status_messages.append(f"ERROR during Prophet forecasting: {e}")
        print(f"Prophet failed: {e}")


    # --- 4. Basic ML Prediction ---
    try:
        growth_label, growth_proba = train_predict_model(df.copy()) # Train on a copy
        status_messages.append(f"Basic ML Prediction: {growth_label} (Prob: {growth_proba})")
        print(f"ML Prediction done: {growth_label} ({growth_proba})")
    except Exception as e:
        status_messages.append(f"ERROR during ML prediction: {e}")
        print(f"ML failed: {e}")
        growth_label = "N/A: Error"
        growth_proba = 0.0

    # --- 5. Sentiment Analysis (Placeholder) ---
    try:
        sentiment_score, sentiment_msg = fetch_sentiment_data(sentiment_keyword)
        status_messages.append(f"Placeholder Sentiment Score ({sentiment_keyword if sentiment_keyword else 'N/A'}): {sentiment_score}. {sentiment_msg}")
        print(f"Sentiment done: {sentiment_score}")
    except Exception as e:
        status_messages.append(f"ERROR during Sentiment analysis: {e}")
        print(f"Sentiment failed: {e}")
        sentiment_score = 0.0
        sentiment_msg = "Error"


    # --- 6. ArcheanVision Signals (Optional) ---
    if ARCHEANVISION_API_KEY:
        try:
            # Only attempt if it's a crypto symbol likely supported by AV
            if market_type == "Crypto":
                av_signals_df, av_signals_msg = get_archean_signals_cloudscraper(ARCHEANVISION_API_KEY, symbol)
                status_messages.append(f"ArcheanVision API: {av_signals_msg}")
                print(f"ArcheanVision check done: {av_signals_msg}")
                if not av_signals_df.empty:
                     # Format date for display if present
                    if 'date' in av_signals_df.columns:
                         av_signals_df['date'] = pd.to_datetime(av_signals_df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                av_signals_msg = "ArcheanVision API not checked (selected market is Stock)."
                status_messages.append(av_signals_msg)
        except Exception as e:
             av_signals_msg = f"Error fetching ArcheanVision data: {e}"
             status_messages.append(f"ERROR: {av_signals_msg}")
             print(f"ArcheanVision fetch failed: {e}")
    else:
        av_signals_msg = "ArcheanVision API Key not found in environment variables (.env)."
        status_messages.append(av_signals_msg)

    # --- Compile final status message ---
    final_status = "\n".join(status_messages)

    print("--- Analysis Complete ---")

    return (forecast_plot, tech_plot, rsi_plot, macd_plot, forecast_data_display,
            growth_label, growth_proba, sentiment_score, sentiment_msg, av_signals_df,
            av_signals_msg, final_status)


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"), title="Market Analysis Tool") as demo:
    gr.Markdown("# Comprehensive Market Analysis & Forecasting Tool")
    gr.Markdown(
        """
        This tool fetches market data using **yfinance**, performs technical analysis (RSI, MACD, Bollinger Bands),
        generates price forecasts using **Facebook Prophet**, trains a **basic local Random Forest model** to predict short-term trends,
        shows a **placeholder sentiment score** based on a keyword, and optionally attempts to fetch signals from the **ArcheanVision API**
        if an `ARCHEANVISION_API_KEY` is configured in your `.env` file.

        **Note:** The local ML model is very basic and for demonstration only. Sentiment analysis is a placeholder. ArcheanVision integration depends on API access and `cloudscraper`.
        """
    )

    with gr.Row():
        # --- Input Column ---
        with gr.Column(scale=1):
            gr.Markdown("## Analysis Configuration")
            market_type_dd = gr.Radio(label="Market Type", choices=["Crypto", "Stock"], value="Crypto")
            symbol_dd = gr.Dropdown(label="Symbol", choices=CRYPTO_SYMBOLS, value="BTC-USD", interactive=True)
            interval_dd = gr.Dropdown(label="Interval", choices=INTERVAL_OPTIONS, value="1d", interactive=True)

            with gr.Accordion("Prophet Forecast Settings", open=False):
                 forecast_steps_slider = gr.Slider(label="Forecast Steps (Periods)", minimum=1, maximum=100, value=30, step=1)
                 daily_box = gr.Checkbox(label="Daily Seasonality", value="auto") # Use auto for Prophet
                 weekly_box = gr.Checkbox(label="Weekly Seasonality", value="auto")
                 yearly_box = gr.Checkbox(label="Yearly Seasonality", value="auto")
                 seasonality_mode_dd = gr.Dropdown(label="Seasonality Mode", choices=["additive", "multiplicative"], value="additive")
                 changepoint_scale_slider = gr.Slider(label="Changepoint Prior Scale (Flexibility)", minimum=0.001, maximum=0.5, step=0.005, value=0.05) # Adjusted range

            with gr.Accordion("Sentiment (Placeholder)", open=False):
                 sentiment_keyword_txt = gr.Textbox(label="Keyword for Sentiment Analysis (e.g., Bitcoin)")

            analyze_button = gr.Button("Run Analysis", variant="primary")
            status_output = gr.Textbox(label="Analysis Status Log", lines=8, interactive=False)


        # --- Output Column ---
        with gr.Column(scale=3):
            gr.Markdown("## Analysis Results")
            # Main Forecast Plot
            forecast_plot_output = gr.Plot(label="Price Forecast (History & Prediction)")

            with gr.Tabs():
                 with gr.TabItem("Technical Analysis Charts"):
                      tech_plot_output = gr.Plot(label="Price & Bollinger Bands")
                      with gr.Row():
                           rsi_plot_output = gr.Plot(label="RSI Indicator", scale=1)
                           macd_plot_output = gr.Plot(label="MACD Indicator", scale=1)
                 with gr.TabItem("Forecast Data"):
                      forecast_df_output = gr.DataFrame(label="Prophet Forecast Data (Future Periods)", interactive=False)
                 with gr.TabItem("Basic ML & Sentiment"):
                       with gr.Row():
                           growth_label_output = gr.Label(label="Basic ML Trend Prediction")
                           growth_proba_output = gr.Number(label="Prediction Probability (Class 1)")
                       with gr.Row():
                           sentiment_label_output = gr.Number(label="Placeholder Sentiment Score")
                           sentiment_msg_output = gr.Textbox(label="Sentiment Status", interactive=False)
                 with gr.TabItem("ArcheanVision Signals (Optional)"):
                      av_status_output = gr.Textbox(label="ArcheanVision API Status", interactive=False)
                      av_signals_output = gr.DataFrame(label="Latest ArcheanVision Signals", interactive=False, height=300)


    # --- Event Listeners ---

    # Update symbol dropdown based on market type
    def update_symbol_choices(market_type):
        if market_type == "Crypto":
            choices = CRYPTO_SYMBOLS
            value = choices[0] if choices else None
        elif market_type == "Stock":
            choices = STOCK_SYMBOLS
            value = choices[0] if choices else None
             # Optionally disable incompatible intervals for stocks
            # interval_choices = ["1d"] # Restrict intervals for stocks if needed
            # return gr.Dropdown(choices=choices, value=value), gr.Dropdown(choices=interval_choices, value="1d")
        else:
            choices = []
            value = None
        return gr.Dropdown(choices=choices, value=value) # Just update symbols for now

    market_type_dd.change(fn=update_symbol_choices, inputs=market_type_dd, outputs=symbol_dd) # Simplified update


    # Connect button click to the main analysis function
    analyze_button.click(
        fn=analyze_market,
        inputs=[
            market_type_dd,
            symbol_dd,
            interval_dd,
            forecast_steps_slider,
            daily_box,
            weekly_box,
            yearly_box,
            seasonality_mode_dd,
            changepoint_scale_slider,
            sentiment_keyword_txt,
        ],
        outputs=[
            forecast_plot_output,
            tech_plot_output,
            rsi_plot_output,
            macd_plot_output,
            forecast_df_output,
            growth_label_output,
            growth_proba_output,
            sentiment_label_output,
            sentiment_msg_output,
            av_signals_output,
            av_status_output,
            status_output # Output for the main status log
        ]
    )

# --- Launch the App ---
if __name__ == "__main__":
    print("Starting Gradio App...")
    if ARCHEANVISION_API_KEY:
        print("ArcheanVision API Key found in environment variables.")
    else:
        print("ArcheanVision API Key NOT found. AV features will be skipped.")
    demo.launch(debug=False) # Set debug=True for more detailed console output during development