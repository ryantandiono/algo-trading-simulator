"""
Algorithmic Trading Strategy Simulator using Streamlit.
Allows users to select a trading strategy, input parameters, run backtests, and visualize results.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from backtest import run_backtest
from strategies import MeanReversion, Momentum, MachineLearning
from ml_models import train_ml_model, add_technical_indicators
from data_acquisition import get_stock_data

def perform_backtest(
    strategy_choice: str,
    ticker: str,
    start_date: str,
    end_date: str,
    initial_cash: float
):
    """
    Execute the backtesting process based on user inputs.
    Args:
        strategy_choice (str): Selected trading strategy.
        ticker (str): Stock ticker symbol.
        start_date (str): Backtest start date (YYYY-MM-DD).
        end_date (str): Backtest end date (YYYY-MM-DD).
        initial_cash (float): Initial portfolio cash.
    Returns:
        Tuple[float, pd.DataFrame]: Final portfolio value and portfolio value over time.
    """
    # Fetch data
    data = get_stock_data(ticker, start_date, end_date)
    if data.empty:
        st.error("No data fetched. Please check the ticker symbol and date range.")
        return None, None

    # Add technical indicators
    try:
        data = add_technical_indicators(data.copy())
        st.success("Technical indicators added successfully.")
    except KeyError as e:
        st.error(f"Error adding indicators: {e}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while adding technical indicators: {e}")
        return None, None

    # Ensure 'datetime' column exists
    if 'date' in data.columns:
        data.rename(columns={'date': 'datetime'}, inplace=True)
    elif 'Date' in data.columns:
        data.rename(columns={'Date': 'datetime'}, inplace=True)

    # Reset index if 'datetime' is the index
    if data.index.name == 'date':
        data.reset_index(inplace=True)

    # Convert 'datetime' to datetime type
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
    else:
        st.error("No 'datetime' column found after processing.")
        return None, None

    # Select strategy
    if strategy_choice == "Mean Reversion":
        strategy = MeanReversion
        model, scaler = None, None
    elif strategy_choice == "Momentum":
        strategy = Momentum
        model, scaler = None, None
    elif strategy_choice == "Machine Learning":
        try:
            model, scaler = train_ml_model(data)
            strategy = MachineLearning
        except KeyError as e:
            st.error(f"Feature Engineering Error: {e}")
            return None, None
        except Exception as e:
            st.error(f"Model Training Error: {e}")
            return None, None
    else:
        st.error("Invalid strategy selected.")
        return None, None

    # Run backtest
    try:
        final_value, portfolio_df = run_backtest(
            strategy,
            ticker,
            start_date,
            end_date,
            initial_cash,
            model=model,
            scaler=scaler
        )
    except KeyError as e:
        st.error(f"Backtesting Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Backtesting Error: {e}")
        return None, None

    return final_value, portfolio_df


def display_results(final_value: float, portfolio_df: pd.DataFrame):
    """
    Display backtest results and performance metrics.
    Args:
        final_value (float): Final portfolio value.
        portfolio_df (pd.DataFrame): Portfolio value over time.
    """
    # Convert datetime to datetime type if not already
    portfolio_df['datetime'] = pd.to_datetime(portfolio_df['datetime'], errors='coerce')
    
    if portfolio_df['datetime'].isnull().any():
        st.error("Some 'datetime' entries could not be converted. Please check the data.")
        return

    # Check unique dates
    unique_dates = portfolio_df['datetime'].nunique()
    if unique_dates < 2:
        st.error("Not enough unique dates to plot. Please check the backtest logic")
        return
    
    st.subheader("Backtest Results")
    st.write(f"**Final Portfolio Value:** ${final_value:,.2f}")

    st.subheader("Portfolio Value Over Time")
    fig = px.line(
        portfolio_df,
        x='datetime',
        y='Portfolio Value',
        labels={'datetime': 'Date', 'Portfolio Value': 'Value ($)'},
        title='Portfolio Value Over Time'
    )
    fig.update_xaxes(type='date')
    st.plotly_chart(fig)

    # Performance Metrics
    st.subheader("Performance Metrics")
    initial_value = portfolio_df['Portfolio Value'].iloc[0]
    total_return = (final_value - initial_value) / initial_value * 100
    st.write(f"**Total Return:** {total_return:.2f}%")

    portfolio_df['Daily Return'] = portfolio_df['Portfolio Value'].pct_change()
    sharpe_ratio = (portfolio_df['Daily Return'].mean() / portfolio_df['Daily Return'].std()) * (252**0.5)
    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

    portfolio_df['Cumulative Max'] = portfolio_df['Portfolio Value'].cummax()
    portfolio_df['Drawdown'] = (portfolio_df['Portfolio Value'] - portfolio_df['Cumulative Max']) / portfolio_df['Cumulative Max']
    max_drawdown = portfolio_df['Drawdown'].min() * 100
    st.write(f"**Maximum Drawdown:** {max_drawdown:.2f}%")

    # Daily Returns Plot
    st.subheader("Daily Returns Over Time")
    fig2 = px.line(
        portfolio_df,
        x=portfolio_df.index,
        y='Daily Return',
        labels={'x': 'Date', 'Daily Return': 'Daily Return'},
        title='Daily Returns Over Time'
    )
    st.plotly_chart(fig2)


def main():
    """
    Main function to render the Streamlit app.
    """
    st.title("Algorithmic Trading Strategy Simulator")
    # My Linkedin
    linkedin_url = 'https://www.linkedin.com/in/ryan-tandiono-331584207/'
    linkedin_icon = 'https://cdn-icons-png.flaticon.com/512/174/174857.png'

    # Sidebar inputs
    with st.sidebar:
        st.markdown(
            f'''
            <div style="display: flex; align-items: center;">
                <a href="{linkedin_url}" target="_blank">
                    <img src="{linkedin_icon}" width="30" style="margin-right: 10px;">
                </a>
            </div>
            ''',
            unsafe_allow_html=True
        )
        st.sidebar.header("Backtest Parameters")
        strategy_choice = st.sidebar.selectbox("Select Strategy", ['Mean Reversion', 'Momentum', 'Machine Learning'])
        ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")
        start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2021-01-01"))
        initial_cash = st.sidebar.number_input("Initial Cash ($)", value=10000.0, min_value=100.0, step=100.0)

    if st.sidebar.button("Run Backtest"):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        if start_str >= end_str:
            st.error("Start date must be before end date.")
            return

        final_value, portfolio_df = perform_backtest(
            strategy_choice,
            ticker,
            start_str,
            end_str,
            initial_cash
        )

        if final_value and portfolio_df is not None:
            display_results(final_value, portfolio_df)
    
    # Add footer at bottom
    st.markdown(
        """
        <hr>
        <div style="text-align: center; font-size: 14px; color: lightgrey; font-family: Arial, sans-serif;">
            Developed by Ryan Tandiono, for educational purposes only. Not financial advice.
        </div>
        """,
        unsafe_allow_html=True
    )
    
if __name__ == '__main__':
    main()
