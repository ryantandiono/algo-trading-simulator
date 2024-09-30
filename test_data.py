from data_acquisition import get_stock_data, get_crypto_data
import sys

def main():
    """
    Main function to test data acquisition functions.
    Fetches and displays sample stock and cryptocurrency data.
    """
    # Define test parameters
    STOCK_TICKER = 'AAPL'
    STOCK_START_DATE = '2020-01-01'
    STOCK_END_DATE = '2021-01-01'
    CRYPTO_SYMBOL = 'BTC/USDT'
    
    # Test fetching stock data
    print(f"Fetching stock data for {STOCK_TICKER} from {STOCK_START_DATE} to {STOCK_END_DATE}...")
    stock_data = get_stock_data(STOCK_TICKER, STOCK_START_DATE, STOCK_END_DATE)
    
    if not stock_data.empty:
        print("\nSample Stock Data:")
        print(stock_data.head())
    else:
        print(f"\nFailed to fetch stock data for '{STOCK_TICKER}'. Please check the ticker symbol and date range.")
    
    print("\n" + "-"*60 + "\n")
    
    # Test fetching cryptocurrency data
    print(f"Fetching cryptocurrency data for '{CRYPTO_SYMBOL}'...")
    crypto_data = get_crypto_data(CRYPTO_SYMBOL)
    
    if not crypto_data.empty:
        print("\nSample Cryptocurrency Data:")
        print(crypto_data.head())
    else:
        print(f"\nFailed to fetch cryptocurrency data for '{CRYPTO_SYMBOL}'. Please check the symbol and exchange settings.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
