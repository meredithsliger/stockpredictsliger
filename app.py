import os
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import streamlit as st

# --- MUST MATCH your notebook feature set from training ---
feature_cols = [
    "return_1d", "return_5d", "return_10d",
    "vol_5d",
    "ma_5", "ma_20", "ma_ratio",
    "rsi_14",
    "vol_ma_10", "vol_ratio",
    "spy_ret_5d", "spy_ret_10d",
    "rel_strength_5d", "rel_strength_10d",
]

# ---------- Session state defaults ----------
if "prediction" not in st.session_state:
    # will store dict like:
    # {"ticker": str, "p1": float, "p5": float, "live_date": pd.Timestamp}
    st.session_state.prediction = None


# ---------- Helper: build live features for a ticker ----------
def prepare_live_features(ticker):
    """
    Build the latest feature row for a given ticker,
    using the SAME logic as training.
    """
    # 1) Download recent history for this ticker
    df = yf.download(ticker, period="200d")
    df = df.sort_index()

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- price-based features ---
    df["return_1d"]  = df["Close"].pct_change()
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    daily = df["Close"].pct_change()
    df["vol_5d"] = daily.rolling(5).std()
    df["ma_5"]   = df["Close"].rolling(5).mean()
    df["ma_20"]  = df["Close"].rolling(20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]

    # --- RSI(14) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- volume features ---
    df["vol_ma_10"] = df["Volume"].rolling(10).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma_10"]

    # --- SP500 context ---
    spy = yf.download("^GSPC", period="200d").sort_index()
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy5  = spy["Close"].pct_change(5)
    spy10 = spy["Close"].pct_change(10)

    df["spy_ret_5d"]  = spy5.reindex(df.index)
    df["spy_ret_10d"] = spy10.reindex(df.index)
    df["rel_strength_5d"]  = df["return_5d"]  - df["spy_ret_5d"]
    df["rel_strength_10d"] = df["return_10d"] - df["spy_ret_10d"]

    # Drop rows with NaNs in feature columns
    df = df.dropna(subset=feature_cols)

    if df.empty:
        raise ValueError("Not enough recent data to build features.")

    # Take the most recent row of features
    latest_feats = df[feature_cols].iloc[-1:]
    latest_date  = df.index[-1]
    return latest_feats, latest_date


# ---------- Helper: load models + make prediction ----------
def predict_live(ticker):
    """
    Use the saved XGBoost models for this ticker
    to get 1-day and 5-day up probabilities.
    """
    X_live, live_date = prepare_live_features(ticker)

    model_1d_path = os.path.join("models_1d", f"{ticker}.pkl")
    model_5d_path = os.path.join("models_5d", f"{ticker}.pkl")

    if not os.path.exists(model_1d_path):
        raise FileNotFoundError(f"No 1-day model for {ticker}: {model_1d_path}")
    if not os.path.exists(model_5d_path):
        raise FileNotFoundError(f"No 5-day model for {ticker}: {model_5d_path}")

    model_1d = joblib.load(model_1d_path)
    model_5d = joblib.load(model_5d_path)

    p1 = model_1d.predict_proba(X_live)[0, 1]
    p5 = model_5d.predict_proba(X_live)[0, 1]

    return p1, p5, live_date


# ---------- Helper: historical price data for chart ----------
@st.cache_data
def get_price_history(ticker, period="6mo"):
    """
    Download recent price history for plotting.
    Cached to avoid repeated downloads.
    """
    if not ticker:
        return pd.DataFrame()

    hist = yf.download(ticker, period=period)
    hist = hist.sort_index()

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    return hist


# ---------- Streamlit UI ----------
st.title("📈 Toy Stock Direction Predictor")
st.caption("For fun + learning only. Not financial advice. 🙃")

# Find all tickers we have models for
if not os.path.exists("models_1d") or not os.path.exists("models_5d"):
    st.error(
        "models_1d/ or models_5d/ folder not found. "
        "Make sure app.py is in the same folder as your model folders."
    )
else:
    tickers_available = sorted(
        f.replace(".pkl", "") for f in os.listdir("models_1d") if f.endswith(".pkl")
    )

    if not tickers_available:
        st.error("No model files found in models_1d/. Train and save models first.")
    else:
        # Choose ticker
        ticker = st.selectbox("Choose a ticker:", tickers_available, index=0)

        # Button: when clicked, we compute and STORE the prediction in session_state
        if st.button("🔮 Predict direction"):
            with st.spinner("Downloading data and running models..."):
                try:
                    p1, p5, live_date = predict_live(ticker)

                    # save to session_state so it persists when other widgets change
                    st.session_state.prediction = {
                        "ticker": ticker,
                        "p1": p1,
                        "p5": p5,
                        "live_date": live_date,
                    }

                except Exception as e:
                    st.session_state.prediction = None
                    st.error(f"Something went wrong: {e}")

        # --- Show prediction results if we have them stored ---
        pred = st.session_state.prediction

        if pred is not None:
            # Optionally check that prediction ticker matches current selection
            if pred["ticker"] != ticker:
                st.info(
                    f"Last prediction is for {pred['ticker']}. "
                    f"Change ticker and click 'Predict' again to update."
                )

            st.success(f"Latest data date: {pred['live_date'].date()}")

            col1, col2 = st.columns(2)
            col1.metric("Prob UP Tomorrow", f"{pred['p1']:.1%}")
            col2.metric("Prob UP in 5 Days", f"{pred['p5']:.1%}")

            st.write(
                "If probabilities are close to 50%, the model isn't very confident. "
                "This is totally normal for stock direction models."
            )

        # --- Always show price history section for the currently selected ticker ---
        st.markdown("---")
        st.subheader(f"{ticker} price history")

        period = st.selectbox(
            "History window",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2,  # default "6mo"
            key="hist_period",
        )

        hist = get_price_history(ticker, period=period)

        if hist.empty:
            st.warning("No price data returned for this ticker/period.")
        else:
            st.line_chart(hist["Close"])

            with st.expander("Show raw historical data"):
                st.dataframe(
                    hist[["Open", "High", "Low", "Close", "Volume"]]
                    .reset_index()
                    .rename(columns={"Date": "date"})
                )
