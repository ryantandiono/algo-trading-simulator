"""
Train and evaluate a machine learning model for stock price prediction.

Steps:
1. Fetch historical stock data.
2. Add technical indicators.
3. Perform Exploratory Data Analysis (EDA).
4. Train the model.
5. Save the trained model and scaler.
"""
import argparse
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_acquisition import get_stock_data
from ml_models import train_ml_model, add_technical_indicators


def perform_eda(data: pd.DataFrame, save_plots: bool = False, output_dir: str = 'plots') -> None:
    """
    Generate plots for EDA: Closing Prices with Bollinger Bands and Correlation Heatmap.
    Args:
        data (pd.DataFrame): DataFrame with technical indicators.
        save_plots (bool): Whether to save the plots as PNG files.
        output_dir (str): Directory to save plots if `save_plots` is True.
    """
    required_cols = {'close', 'upper_band', 'lower_band'}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"Missing columns for EDA: {required_cols - set(data.columns)}")

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    # Plot Closing Price with Bollinger Bands
    plt.figure(figsize=(14, 8))
    plt.plot(data['close'], label='Close Price', color='blue')
    plt.plot(data['upper_band'], label='Upper Band', color='red', linestyle='--')
    plt.plot(data['lower_band'], label='Lower Band', color='green', linestyle='--')
    plt.title('Closing Price with Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'closing_prices.png'))
    plt.show()

    # Plot Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = data[['ma', 'std', 'upper_band', 'lower_band', 'rsi', 'close']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.show()

def main(args) -> None:
    """
    Execute the training pipeline.
    Args:
        args: Command-line arguments.
    """
    # Fetch data
    print(f"Fetching data for {args.ticker} from {args.start_date} to {args.end_date}...")
    data = get_stock_data(args.ticker, args.start_date, args.end_date)
    if data.empty:
        print("No data fetched. Exiting.")
        return

    # Add technical indicators
    print("Adding technical indicators...")
    try:
        data = add_technical_indicators(data, period=args.period)
    except KeyError as e:
        print(f"Error adding indicators: {e}")
        return

    # Perform EDA
    print("Performing EDA...")
    try:
        perform_eda(data, save_plots=args.save_plots, output_dir=args.output_dir)
    except ValueError as e:
        print(f"EDA Error: {e}")

    # Train model
    print("Training the model...")
    try:
        model, scaler = train_ml_model(data)
    except KeyError as e:
        print(f"Training Error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error during training: {e}")
        return

    # Save model and scaler
    print("Saving model and scaler...")
    try:
        joblib.dump(model, args.model_save_path)
        joblib.dump(scaler, args.scaler_save_path)
        print("Model and scaler saved successfully.")
    except Exception as e:
        print(f"Error saving artifacts: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ML model for stock prediction.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol.')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, default='2024-01-01', help='End date (YYYY-MM-DD).')
    parser.add_argument('--period', type=int, default=14, help='Period for technical indicators.')
    parser.add_argument('--save_plots', action='store_true', help='Save EDA plots as PNG files.')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots.')
    parser.add_argument('--model_save_path', type=str, default='model.joblib', help='Path to save the trained model.')
    parser.add_argument('--scaler_save_path', type=str, default='scaler.joblib', help='Path to save the scaler.')

    # Example input in terminal
    # python train_model.py --ticker AAPL --start_date 2020-01-01 --end_date 2024-01-01 --period 14 --save_plots --output_dir plots/AAPL --model_save_path models/AAPL_model.joblib --scaler_save_path models/AAPL_scaler.joblib

    args = parser.parse_args()
    main(args)
