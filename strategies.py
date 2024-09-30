import backtrader as bt
import numpy as np

class MeanReversion(bt.Strategy):
    """
    Mean Reversion Strategy using Bollinger Bands.
    Buys when the price drops below the lower Bollinger Band and sells when it rises above the middle band (SMA).
    """
    params = dict(
        period=20,        # Period for Bollinger Bands
        devfactor=2.0,    # Deviation factor for Bollinger Bands
    )

    def __init__(self):
        """
        Initializes the Bollinger Bands indicator.
        """
        # Initialize Bollinger Bands indicator with specified period and deviation factor
        self.bb = bt.indicators.BollingerBands(
            period=self.p.period, 
            devfactor=self.p.devfactor
        )

    def next(self):
        """
        Executes the strategy logic on each new data point.
        - Buys when the price is below the lower Bollinger Band.
        - Sells when the price is above the middle Bollinger Band (SMA).
        """
        if not self.position:
            # If not in position and price is below the lower Bollinger Band, BUY
            if self.data.close[0] < self.bb.lines.bot[0]:
                self.buy()
                self.log(f'Buy Executed at {self.data.close[0]:.2f}')
        else:
            # If in position and price is above the middle Bollinger Band, SELL
            if self.data.close[0] > self.bb.lines.mid[0]:
                self.sell()
                self.log(f'Sell Executed at {self.data.close[0]:.2f}')

    def log(self, txt, dt=None):
        """
        Logs a message with an optional timestamp.
        Parameters:
            txt (str): The message to log.
            dt (datetime.date, optional): The date to associate with the log message.
                                           Defaults to the current data point's date.
        """
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')


class Momentum(bt.Strategy):
    """
    Momentum Strategy using Relative Strength Index (RSI).
    Buys when RSI falls below 30 (indicating oversold conditions) and sells when RSI rises above 70 (indicating overbought conditions).
    """
    params = dict(
        period=14,    # Period for RSI
    )

    def __init__(self):
        """
        Initializes the RSI indicator.
        """
        # Initialize RSI indicator with specified period
        self.rsi = bt.indicators.RSI_SMA(
            period=self.p.period
        )

    def next(self):
        """
        Executes the strategy logic on each new data point.
        - Buys when RSI is below 30.
        - Sells when RSI is above 70.
        """
        if not self.position:
            # If not in position and RSI is below 30, BUY
            if self.rsi[0] < 30:
                self.buy()
                self.log(f'Buy Executed at {self.data.close[0]:.2f}')
        else:
            # If in position and RSI is above 70, SELL
            if self.rsi[0] > 70:
                self.sell()
                self.log(f'Sell Executed at {self.data.close[0]:.2f}')

    def log(self, txt, dt=None):
        """
        Logs a message with an optional timestamp.
        Parameters:
            txt (str): The message to log.
            dt (datetime.date, optional): The date to associate with the log message.
                                           Defaults to the current data point's date.
        """
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')


class MachineLearning(bt.Strategy):
    """
    Machine Learning-Based Strategy.
    Utilizes a trained machine learning model to predict price movements.
    Buys when the model predicts an upward movement and sells when it predicts a downward movement.
    """
    params = (
        ('model', None),   # Trained ML model
        ('scaler', None),  # Scaler used during model training
    )

    def __init__(self, *args, **kwargs):
        """
        Initializes the strategy by setting up technical indicators and verifying model parameters.
        """
        super().__init__(*args, **kwargs)
        print("Machine Learning Strategy initialized")

        # Verify that both model and scaler are provided
        if self.params.model is None or self.params.scaler is None:
            raise ValueError("Model and Scaler must be provided.")

        # Initialize technical indicators to match those used during model training
        self.ma = bt.indicators.SimpleMovingAverage(
            self.data.close, 
            period=14
        )
        self.std = bt.indicators.StandardDeviation(
            self.data.close, 
            period=14
        )
        self.upper_band = self.ma + (self.std * 2)
        self.lower_band = self.ma - (self.std * 2)
        self.rsi = bt.indicators.RSI_SMA(
            self.data.close, 
            period=14
        )

    def next(self):
        """
        Executes the strategy logic on each new data point.
        - Gathers technical indicators as features.
        - Scales the features using the provided scaler.
        - Predicts the price movement using the trained model.
        - Buys or sells based on the model's prediction.
        """
        # Ensure sufficient data points are available for indicators
        if len(self) < 15:
            return

        # Gather feature values from indicators
        features = [
            self.ma[0],          # Moving Average
            self.std[0],         # Standard Deviation
            self.upper_band[0],  # Upper Bollinger Band
            self.lower_band[0],  # Lower Bollinger Band
            self.rsi[0],         # RSI
        ]

        # Convert features to a 2D NumPy array for the scaler
        features_array = np.array(features).reshape(1, -1)

        # Scale the features using the provided scaler
        try:
            features_scaled = self.params.scaler.transform(features_array)
        except Exception as e:
            self.log(f"Scaler transformation error: {e}")
            return

        # Predict the price movement using the trained model
        try:
            prediction = self.params.model.predict(features_scaled)
            print(f"Prediction: {prediction}")
        except Exception as e:
            self.log(f"Model prediction error: {e}")
            return

        # Execute trade based on the prediction
        if not self.position and prediction == 1:
            # If not in position and prediction is 1, BUY
            self.buy()
            self.log(f"Buy Executed at {self.data.close[0]:.2f}")
        elif self.position and prediction == 0:
            # If in position and prediction is 0, SELL
            self.sell()
            self.log(f"Sell Executed at {self.data.close[0]:.2f}")

    def log(self, txt, dt=None):
        """
        Logs a message with an optional timestamp.
        
        Parameters:
            txt (str): The message to log.
            dt (datetime.date, optional): The date to associate with the log message.
                                           Defaults to the current data point's date.
        """
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')
