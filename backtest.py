import backtrader as bt
from data_acquisition import get_stock_data, get_crypto_data
from strategies import MeanReversion, Momentum, MachineLearning
import pandas as pd
from analyzers import PortfolioValueAnalyzer
from typing import Optional, Tuple
import logging
import joblib

def run_backtest(
    strategy: bt.Strategy,
    ticker: str,
    start_date: str,
    end_date: str,
    initial_cash: float = 10000.0,
    model: Optional[object] = None,
    scaler: Optional[object] = None
) -> Tuple[float, pd.DataFrame]:
    """
    Executes a backtest for a given trading strategy and financial instrument.
    Parameters:
        strategy (bt.Strategy): The trading strategy class to be used.
        ticker (str): The ticker symbol of the financial instrument (e.g., 'AAPL').
        start_date (str): The start date for the backtest in 'YYYY-MM-DD' format.
        end_date (str): The end date for the backtest in 'YYYY-MM-DD' format.
        initial_cash (float, optional): The starting cash for the portfolio. Defaults to 10000.0.
        model (Optional[object], optional): Trained machine learning model for ML strategies. Required if strategy is MachineLearning.
        scaler (Optional[object], optional): Scaler used during model training for ML strategies. Required if strategy is MachineLearning.
    Returns:
        Tuple[float, pd.DataFrame]: A tuple containing the final portfolio value and a DataFrame of portfolio values over time.
    """
    # Configure logging
    logging.basicConfig(
        filename='backtest.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # Display initial backtest information
    logging.info(f"Running backtest for {ticker} from {start_date} to {end_date} with initial cash ${initial_cash}.")

    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)

    # Fetch historical data using data acquisition functions
    data_df = get_stock_data(ticker, start_date, end_date)
    if data_df.empty:
        logging.error(f"Failed to fetch stock data for '{ticker}'. Aborting backtest.")
        print(f"Failed to fetch stock data for '{ticker}'. Aborting backtest.")
        return initial_cash, pd.DataFrame()

    # Convert pandas DataFrame to Backtrader data feed
    data_feed = bt.feeds.PandasData(
        dataname=data_df,
        datetime='datetime',   # Ensure 'datetime' is correctly named in data_df
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=None
    )
    cerebro.adddata(data_feed)

    # Add the selected trading strategy
    if strategy == MachineLearning:
        if model is None or scaler is None:
            logging.error("Model and Scaler must be provided for Machine Learning Strategy.")
            raise ValueError("Model and Scaler must be provided for Machine Learning Strategy.")
        cerebro.addstrategy(strategy, model=model, scaler=scaler)
    else:
        cerebro.addstrategy(strategy)

    # Add custom analyzer to track portfolio value over time
    cerebro.addanalyzer(PortfolioValueAnalyzer, _name='portfolio_value')

    # Run the backtest
    results = cerebro.run()
    first_strategy = results[0]

    # Retrieve portfolio values from the analyzer
    portfolio_values = first_strategy.analyzers.portfolio_value.get_analysis()

    # Correctly extract datetime objects using num2date
    data_dates = data_df['datetime'].tolist()

    len_data = len(data_dates)
    len_portfolio = len(portfolio_values)

    if len_portfolio > len_data:
        # Trim excess portfolio values
        portfolio_values = portfolio_values[:len_data]
    elif len_portfolio < len_data:
        # Extend portfolio values with the last known value
        last_value = portfolio_values[-1] if len_portfolio > 0 else initial_cash
        padding = [last_value] * (len_data - len_portfolio)
        portfolio_values.extend(padding)

    # Create a DataFrame for portfolio values with corresponding dates
    portfolio_df = pd.DataFrame({
        'datetime': data_dates[:len(portfolio_values)],
        'Portfolio Value': portfolio_values
    })

    # Get the final portfolio value
    final_portfolio_value = cerebro.broker.getvalue()
    logging.info(f"Backtest completed. Final Portfolio Value: ${final_portfolio_value:.2f}")

    return final_portfolio_value, portfolio_df


if __name__ == '__main__':
    """
    Example usage of the run_backtest function.
    """
    # Select the trading strategy
    strategy = Momentum  # Options: MeanReversion, Momentum, MachineLearning

    # Define backtest parameters
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    initial_cash = 10000.0

    # For MachineLearning strategy, load the trained model and scaler
    # Example:
    # model = joblib.load('model.joblib')
    # scaler = joblib.load('scaler.joblib')

    model = None
    scaler = None

    # Execute the backtest
    final_value, portfolio = run_backtest(
        strategy=strategy,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        model=model,
        scaler=scaler
    )

    # Display the results
    print(f"\nFinal Portfolio Value: ${final_value:.2f}")
    if not portfolio.empty:
        print("\nSample Portfolio Values:")
        print(portfolio.head())
    else:
        print("\nNo portfolio data to display.")
