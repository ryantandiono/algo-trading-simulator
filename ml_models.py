import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a given price series.
    RSI is a momentum oscillator that measures the speed and change of price movements.
    Parameters:
        series (pd.Series): A pandas Series of price data (e.g., closing prices).
        period (int, optional): The number of periods to use for calculating RSI. Defaults to 14.
    Returns:
        pd.Series: A pandas Series containing the RSI values.
    """
    # Calculate the difference in price from the previous period
    delta = series.diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate the Exponential Moving Average for gains and losses
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI based on RS
    rsi = 100 - (100 / (1 + rs))

    return rsi

def add_technical_indicators(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add common technical analysis indicators to the dataframe.
    Indicators added:
        - Simple Moving Average (SMA)
        - Standard Deviation (STD)
        - Bollinger Bands (Upper and Lower)
        - Relative Strength Index (RSI)
    Parameters:
        df (pd.DataFrame): DataFrame containing at least a 'close' column.
        period (int, optional): The number of periods to use for calculating indicators. Defaults to 14.
    Returns:
        pd.DataFrame: The input DataFrame enriched with technical indicators.
    """
    if 'close' not in df.columns:
        raise KeyError("DataFrame must contain a 'close' column.")

    # Calculate Simple Moving Average
    df['ma'] = df['close'].rolling(window=period, min_periods=period).mean()

    # Calculate Standard Deviation
    df['std'] = df['close'].rolling(window=period, min_periods=period).std()

    # Calculate Bollinger Bands
    df['upper_band'] = df['ma'] + (df['std'] * 2)
    df['lower_band'] = df['ma'] - (df['std'] * 2)

    # Calculate Relative Strength Index
    df['rsi'] = compute_rsi(df['close'], period=period)

    # Drop rows with NaN values resulting from indicator calculations
    df.dropna(inplace=True)

    return df

def train_ml_model(data: pd.DataFrame, features: Optional[List[str]] = None, test_size: float = 0.25) -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Train a Random Forest classifier to predict future price movements.
    The model predicts whether the closing price will increase the next day.
    Parameters:
        data (pd.DataFrame): DataFrame containing technical indicators and 'close' price.
        features (Optional[List[str]], optional): List of feature column names to use for training.
                                               If None, defaults to ['ma', 'std', 'upper_band', 'lower_band', 'rsi'].
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.25.
    Returns:
        Tuple[RandomForestClassifier, StandardScaler]: The trained Random Forest model and the fitted StandardScaler.
    """
    # Define default features if none are provided
    if features is None:
        features = ['ma', 'std', 'upper_band', 'lower_band', 'rsi']

    # Ensure all required feature columns are present
    missing_cols = [col for col in features if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required feature column(s): {missing_cols}")

    # Define target variable: 1 if next day's close is higher than today's close, else 0
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    data.dropna(inplace=True)  # Drop the last row with NaN target

    X = data[features]
    y = data['target']

    # Split the data into training and testing sets (No shuffling for time series data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Initialize and fit the StandardScaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate model accuracy on the test set
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model Accuracy on Test Set: {accuracy:.2f}")

    return model, scaler
