from datetime import datetime
import pandas as pd
# Market Exchange API
from kucoin.client import Market

from .filters import filterData
from .indicators import addIndicators
import statsmodels as sm


client = Market(url='https://api.kucoin.com')


def getLastestData(symbol:str="BTC",timeframe:str='15min', indicators:bool=False, filter_close:bool=True, filter_method:str='savgol')-> pd.DataFrame:
    """Function that uses Kucoin API to get the latest data for a specific symbol and timeframe.

    Args:
        symbol (str, optional): The symbol for the data we want to extract. Defaults to "BTC".
        indicators (bool, optional): Whether we want to add indicators to the dataframe or not. Defaults to False.

    Returns:
        pd.DataFrame: The dataframe containing history.
    """
    klines = client.get_kline(f'{symbol}-USDT',timeframe, startAt=round(datetime.now().timestamp())-100000, endAt=round(datetime.now().timestamp()))
    df = pd.DataFrame(klines,columns=['Date','Open','High','Low','Close','Volume','Amount'],dtype=float)
    df = df.sort_values(by='Date')
    df['Timestamp'] = df['Date'].astype(int)
    df['Date'] = df['Date'].astype(int).apply(datetime.fromtimestamp)
    df = df.set_index('Date')
    if indicators==True:
        df = addIndicators(df)
    if filter_close==True:
        df['Close_filtered'] = filterData(df.Close.values,filter_method)
    return df