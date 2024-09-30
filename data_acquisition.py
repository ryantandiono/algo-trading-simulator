import yfinance as yf
import pandas as pd
import ccxt

def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance.
    Parameters:
        ticker (str): Stock ticker symbol ('AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame containing historical stock data 
    """
    try:
        print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"No data found for ticker '{ticker}' between {start_date} and {end_date}.")
            return pd.DataFrame()

        # Convert column names to lowercase for consistency
        data.reset_index(inplace=True)  # Ensure 'Date' is a column, not the index
        data = data.rename(columns=lambda x: x.lower())
        if 'date' in data.columns:
            data = data.rename(columns = {'date':'datetime'})
        else:
            print("Column date not found. Please check the DataFrame columns.")
            return pd.DataFrame()
        print(f"Successfully fetched stock data for {ticker}.")
        return data

    except Exception as e:
        print(f"An error occurred while fetching stock data for '{ticker}': {e}")
        return pd.DataFrame()


def get_crypto_data(symbol: str, exchange: str = 'binance', timeframe: str = '1d') -> pd.DataFrame:
    """
    Fetches historical cryptocurrency data from a specified exchange.
    Parameters:
        symbol (str): Cryptocurrency symbol pair ('BTC-USD').
        exchange (str, optional): Exchange name (default: 'binance').
        timeframe (str, optional): Timeframe for OHLCV data (default: '1d').
    Returns:
        pd.DataFrame: DataFrame containing historical cryptocurrency data
    """
    try:
        print(f"Fetching crypto data for {symbol} from exchange '{exchange}' with timeframe '{timeframe}'...")
        
        # Check if the specified exchange exists in ccxt
        if exchange not in ccxt.exchanges:
            print(f"Exchange '{exchange}' is not supported by ccxt.")
            return pd.DataFrame()

        exchange_class = getattr(ccxt, exchange)()
        bars = exchange_class.fetch_ohlcv(symbol, timeframe=timeframe)

        if not bars:
            print(f"No OHLCV data found for symbol '{symbol}' on exchange '{exchange}' with timeframe '{timeframe}'.")
            return pd.DataFrame()

        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.drop('timestamp', axis=1, inplace=True)  # Remove raw timestamp

        print(f"Successfully fetched crypto data for {symbol}.")
        return df

    except ccxt.BaseError as e:
        print(f"CCXT error while fetching crypto data for '{symbol}' on '{exchange}': {e}")
        return pd.DataFrame()

    except Exception as e:
        print(f"An error occurred while fetching crypto data for '{symbol}': {e}")
        return pd.DataFrame()
