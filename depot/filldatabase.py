#!/usr/bin/python

from datetime import datetime
import pandas as pd
import time
#pip3 install 'pymongo[srv]' pymongo  python-dateutil pandas numpy python-kucoin
from pymongo import MongoClient
import pymongo

from dateutil import parser


CONNECTION_STRING = "mongodb+srv://BaptisteZloch:TM8dR9QnfnHo3Ge8@baptistecluster.tirvo.mongodb.net/test"
mongo_client = MongoClient(CONNECTION_STRING)
db = mongo_client['cryptos']
#db.create_collection('ohlcv', timeseries={ 'timeField': 'timestamp' })

from kucoin.client import Market
kc_market_client = Market(url='https://api.kucoin.com')


def writeData(row_to_write,symbol):
    crypto_db = db[u'ohlcv']
    
    crypto_db.insert_one({
        "symbol":symbol,
        "open":row_to_write['Open'],
        "high":row_to_write['High'],
        "low":row_to_write['Low'],
        "close":row_to_write['Close'],
        "transaction volume":row_to_write['Transaction volume'],
        "transaction amount":row_to_write['Transaction amount'],
        "timestamp":datetime.today().replace(microsecond=0)
    })
    print('row stored !')
    
def getData(symbol) -> pd.DataFrame:
    klines = kc_market_client.get_kline(symbol,'1min', startAt=round(datetime.now().timestamp())-20000, endAt=round(datetime.now().timestamp()))
    klines_dataframe = pd.DataFrame(klines,columns=['Date','Open','High','Low','Close','Transaction volume','Transaction amount'],dtype=float)
    klines_dataframe = klines_dataframe.sort_values(by='Date')
    klines_dataframe.drop(columns=['Transaction volume','Transaction amount'])
    klines_dataframe['Timestamp'] = klines_dataframe['Date'].astype(int)
    klines_dataframe['Date'] = klines_dataframe['Date'].astype(int).apply(datetime.fromtimestamp)
    klines_dataframe = klines_dataframe.set_index('Date')
    return klines_dataframe

if __name__ == "__main__":
    while True:
        symbol = 'AVAX-USDT'
        df = getData(symbol)
        print(df.tail())
        writeData(df.iloc[-1],symbol)
        time.sleep(60)