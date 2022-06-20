# Data handling
import time
import pandas as pd

# Linear algebra
import numpy as np

# Date handling
from datetime import datetime

# Signal processing
from scipy.signal import savgol_filter

# Machine learning
from sklearn.linear_model import LinearRegression

# Warnings
import warnings 
warnings.filterwarnings('ignore')

# Logger
import logging
logging.basicConfig(filename='logger.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',level=logging.DEBUG)

# Market Exchange API
from kucoin.client import Market
client = Market(url='https://api.kucoin.com')


def getLastestData(symbol="BTC",timeframe='15min', indicators=False)-> pd.DataFrame:
    """Function that uses Kucoin API to get the latest data for a specific symbol and timeframe.

    Args:
        symbol (str, optional): The symbol for the data we want to extract. Defaults to "BTC".
        indicators (bool, optional): Whether we want to add indicators to the dataframe or not. Defaults to False.

    Returns:
        pd.DataFrame: The dataframe containing history.
    """
    klines = client.get_kline(f'{symbol}-USDT','15min', startAt=round(datetime.now().timestamp())-100000, endAt=round(datetime.now().timestamp()))
    df = pd.DataFrame(klines,columns=['Date','Open','High','Low','Close','Transaction volume','Transaction amount'],dtype=float)
    df = df.sort_values(by='Date')
    df['Timestamp'] = df['Date'].astype(int)
    df['Date'] = df['Date'].astype(int).apply(datetime.fromtimestamp)
    df = df.set_index('Date')
    logging.debug(f'Data loaded : {len(df)} rows')
    return df


def computeLaggingLinearRegression(df, col="Close",window=15)->pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): The dataframe containing features.
        col (str, optional): The column we apply Linear regression on. Defaults to "Close.
        window (int, optional): The window we apply linear regression on. Defaults to 15.

    Returns:
        pd.DataFrame: The entry DataFrame we another column called B_MLR_coefs
    """
    coefs_2 = []
    intercepts_2 = []
    logging.debug(f'Before computing LR')
    for i in range(window,len(df)):
        ys = df[col].values[i-window:i]
        xs = df.Timestamp.values[i-window:i].reshape(-1, 1)
        
        model = LinearRegression().fit(xs,ys)
        
        coefs_2.append(model.coef_[0])
        intercepts_2.append(model.intercept_)
    logging.debug(f'LR computed !')
    df['B_MLR_coefs'] = np.nan
    df['B_MLR_coefs'].iloc[window:] = coefs_2    
    return df.dropna()

def computeBuySellSignal(df)-> pd.DataFrame:
    """Generate all BUY and SELL signals.

    Args:
        df (pd.DataFrame): The Dataframe we add buy sell signal.

    Returns:
        pd.DataFrame: The final dataframe with 2 new columns : B_MLR_coefs_filtered and BUY_SELL.
    """
    
    df['B_MLR_coefs_filtered'] = savgol_filter(df['B_MLR_coefs'].values, 40, 2)
    logging.debug(f'Data filtered !')
    df['B_MLR_coefs_filtered_diff'] = df['B_MLR_coefs_filtered'].diff()
    logging.debug(f'Data differentiated !')
    df.dropna(inplace=True)
    df['BUY_SELL'] = np.nan
    df['BUY_SELL'].iloc[1:] = ["HOLD" if np.sign(df['B_MLR_coefs_filtered_diff'][i-1])==np.sign(df['B_MLR_coefs_filtered_diff'][i]) else "BUY" if df['B_MLR_coefs_filtered_diff'][i-1]<0 and df['B_MLR_coefs_filtered_diff'][i]>0 else "SELL" for i in range(1,len(df))]
    logging.debug(f'BUY SELL signal added !')
    return df.dropna().drop(columns=['B_MLR_coefs_filtered_diff'])
        
if __name__ == "__main__":
    cryptos = ['ETH','BTC','SOL','KDA']
    
    POSITION_OPENED = {crypto:False for crypto in cryptos}
    
    while True:
        for crypto in cryptos:
            df = getLastestData(crypto, '15min', False)
            computeLaggingLinearRegression(df)
            computeBuySellSignal(df)
            if df['BUY_SELL'].iloc[-1]=="BUY" and POSITION_OPENED[crypto]==False:
                logging.debug(f'Time to buy {crypto} at {df["Close"].iloc[-1]} USDT !')
                POSITION_OPENED[crypto] = True
            elif df['BUY_SELL'].iloc[-1]=="SELL" and POSITION_OPENED[crypto]==True:
                logging.debug(f'Time to sell {crypto} at {df["Close"].iloc[-1]} USDT !')
                POSITION_OPENED[crypto] = False
            else:
                logging.debug(f'Nothing to do with {crypto}')
        logging.debug(f'State of trades : \n{POSITION_OPENED}')
        time.sleep(900)