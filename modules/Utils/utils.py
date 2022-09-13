import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
from .filters import filterData
import requests
import json

import pylab as pl
from numpy import fft

def loadFromDB(path:str, keep_timestamp:bool=True)->pd.DataFrame:
    df = pd.read_csv(path,names=['Date','Open','High','Low','Close','Volume'])
    df = df.iloc[1:]
    df['Date'] = df['Date'].apply(lambda x: int(str(x)[:-3]))
    df = df.astype(float)
    if keep_timestamp==True:
        df['Timestamp'] = df['Date'].astype(int)
    df['Date'] = df['Date'].astype(int).apply(datetime.fromtimestamp)
    df = df.set_index('Date')
    #print(f"Total records : {len(df)} rows")
    return df

def computeStochasticLinearRegression(df:pd.DataFrame, metric:str="Close", )->pd.DataFrame:
    df_copy = df.copy()
    a = list(df_copy[metric])
    b = df_copy[metric].diff().fillna(0)
    df_copy['stepsize']=b
    c=list(df_copy.stepsize.abs())
    import statistics as s
    harmean=[0,0]
    i=1
    j=2
    while j<len(c):
        harsteps=s.harmonic_mean(c[i:j])
        harmean.append(harsteps)
        j+=1
    df_copy['harmean of magnitude'] = harmean
    j=1
    countofup=[0,0]
    probabilityofup = [0,0]
    while j<len(b)-1:
        if b[j]<0:
            countofup.append(0)
            a = sum(countofup)/j
            probabilityofup.append(a)
            j+=1
        else: 
            countofup.append(1)
            p = sum(countofup)/j
            probabilityofup.append(p)
            j+=1
        
    df_copy['probabilityofup'] = probabilityofup
    a=list(df_copy[metric])
    b=list(df_copy['harmean of magnitude'])
    c=list(df_copy['probabilityofup'])
    import random
    pred = [0,0]
    i=1
    while i<len(df_copy)-1:
        if random.uniform(0,1)>c[i]:
            prediction = a[i]+(b[i+1]*-1)
            pred.append(prediction)
            i+=1
        else:
            prediction = a[i]+(b[i+1]*1)
            pred.append(prediction)
            i+=1
        
    df['Stochastic_prediction']=pred
    return  df.iloc[3:]

def getFearAndGreedIndicator(df:pd.DataFrame,keep_na:bool=True):
    """Generate the fear and gred indicator throught alternative.me API. It only works for hourly df data.

    Args:
        df (pd.DataFrame): _description_
        keep_na (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    df_fng = pd.DataFrame(json.loads(requests.get('https://api.alternative.me/fng/?limit=0&format=json').text)['data'])
    df_fng.drop(columns=['time_until_update'],inplace=True)
    df_fng.rename(columns={'timestamp':'date'},inplace=True)
    df_fng['date'] = df_fng['date'].astype(int).apply(lambda x: datetime.fromtimestamp(x))
    df_fng.sort_values('date',inplace=True)
    df_fng.set_index('date',inplace=True)
    def addFNG(x):
        try:
            return int(df_fng.loc[pd.to_datetime(x.name).date().strftime("%Y-%m-%d")].value.values[0])
        except:
            return np.nan
    df['FnG'] = df.apply(addFNG,axis=1)
    if keep_na==False:
        return df.dropna()
    else:
        return df

def computeFutureLinearRegression(df, col="Close",window=15,filter_ceof:bool=True, filter_method:str='savgol',derivative:bool=True, stratify:bool=True)->pd.DataFrame:
    """Compute a lagging moving regression on a column with a window.

    Args:
        df (pd.DataFrame): The dataframe containing features.
        col (str, optional): The column we apply Linear regression on. Defaults to "Close.
        window (int, optional): The window we apply linear regression on. Defaults to 15.

    Returns:
        pd.DataFrame: The entry DataFrame we another column called B_MLR_coefs
    """  
    def computeLinearRegression(x,y)->float:
        """Compute simple linear regression between 2 vectors x and y

        Args:
            x (np.array): x vector
            y (np.array): y vector

        Returns:
            float: The coefficient a corresponding to the linear regression y=ax+b.
        """
        model = LinearRegression().fit(x,y)
        return model.coef_[0]
    
    df['F_MLR_coefs'] = np.nan
    df['F_MLR_coefs'].iloc[:-window] = [computeLinearRegression(df.Timestamp.values[i:i+window].reshape(-1, 1), df[col].values[i:i+window]) for i in range(len(df)-window)] 
    df = df.dropna()
    if filter_ceof==True:
        df['F_MLR_coefs_filtered'] = filterData(df.F_MLR_coefs.values,filter_method)
        if derivative==True:
            df['F_MLR_coefs_filtered_diff'] = df['F_MLR_coefs_filtered'].diff()
        if stratify==True:
            quantiles = np.linspace(0,1,11)
            df['F_MLR_coefs_filtered_strat'] = pd.qcut(df.F_MLR_coefs_filtered,quantiles,labels=[i+1 for i in range(len(quantiles)-1)])
    else:
        if derivative==True:
            df['F_MLR_coefs_diff'] = df['F_MLR_coefs'].diff()
        if stratify==True:
            quantiles = np.linspace(0,1,11)
            df['F_MLR_coefs_filtered_strat'] = pd.qcut(df.F_MLR_coefs_filtered,quantiles,labels=[i+1 for i in range(len(quantiles)-1)])
    return df.dropna()

def strategyTester(df:pd.DataFrame,buyConditonFunc, sellConditionFunc, equity:int=1000, optimization_process:bool=False, stop_loss:bool=False, take_profit:bool=False, tp:int=0,sl:int=0)->pd.DataFrame:
    dfTest = df.copy()

    # -- Definition of dt, that will be the dataset to do your trades analyses --
    dt = None
    dt = pd.DataFrame(columns = ['date','position', 'reason', 'price', 'frais' ,'fiat', 'coins', 'wallet', 'drawBack'])

    # -- You can change variables below --
    usdt = equity
    makerFee = 0.0002
    takerFee = 0.0007

    # -- Do not touch these values --
    initalWallet = usdt
    wallet = usdt
    coin = 0
    lastAth = 0
    previousRow = dfTest.iloc[0]
    stopLoss = 0
    takeProfit = 500000
    buyReady = True
    sellReady = True
    
        
    # -- Iteration on all your price dataset (df) --
    for index, row in dfTest.iterrows():
        # -- Buy market order --
        if buyConditonFunc(row, previousRow) and usdt > 0 and buyReady == True:
            # -- You can define here at what price you buy --
            buyPrice = row['Close']

            # -- Define the price of you SL and TP or comment it if you don't want a SL or TP --
            if take_profit == True:
                takeProfit = buyPrice + tp * buyPrice
                
            if stop_loss == True:
                stopLoss = buyPrice - sl * buyPrice

            coin = usdt / buyPrice
            fee = takerFee * coin
            coin = coin - fee
            usdt = 0
            wallet = coin * row['Close']

            # -- Check if your wallet hit a new ATH to know the drawBack --
            if wallet > lastAth:
                lastAth = wallet

            # -- You can uncomment the line below if you want to see logs --
            # print("Buy COIN at",buyPrice,'$ the', index)

            # -- Add the trade to DT to analyse it later --
            myrow = {'date': index, 'position': "Buy", 'reason':'Buy Market Order','price': buyPrice,'frais': fee * row['Close'],'fiat': usdt,'coins': coin,'wallet': wallet,'drawBack':(wallet-lastAth)/lastAth}
            dt = dt.append(myrow,ignore_index=True)
        
        # -- Stop Loss --
        elif row['Low'] < stopLoss and coin > 0:
            sellPrice = stopLoss
            usdt = coin * sellPrice
            fee = makerFee * usdt
            usdt = usdt - fee
            coin = 0
            buyReady = True
            wallet = usdt

            # -- Check if your wallet hit a new ATH to know the drawBack --
            if wallet > lastAth:
                lastAth = wallet
            
            # -- You can uncomment the line below if you want to see logs --
            # print("Sell COIN at Stop Loss",sellPrice,'$ the', index)

            # -- Add the trade to DT to analyse it later --
            myrow = {'date': index,'position': "Sell", 'reason':'Sell Stop Loss','price': sellPrice,'frais': fee,'fiat': usdt,'coins': coin,'wallet': wallet,'drawBack':(wallet-lastAth)/lastAth}
            dt = dt.append(myrow,ignore_index=True)    

        elif row['High'] > takeProfit and coin > 0:
            sellPrice = takeProfit

            usdt = coin * sellPrice
            fee = makerFee * usdt
            usdt = usdt - fee
            coin = 0
            buyReady = True
            wallet = usdt
            if wallet > lastAth:
                lastAth = wallet
            # print("Sell COIN at Take Profit Loss",sellPrice,'$ the', index)
            myrow = {'date': index,'position': "Sell", 'reason': 'Sell Take Profit', 'price': sellPrice, 'frais': fee, 'fiat': usdt, 'coins': coin, 'wallet': wallet, 'drawBack':(wallet-lastAth)/lastAth}
            dt = dt.append(myrow,ignore_index=True) 
            
        # -- Sell Market Order --
        elif sellConditionFunc(row, previousRow) and coin > 0 and sellReady == True:

            # -- You can define here at what price you buy --
            sellPrice = row['Close']
            usdt = coin * sellPrice
            fee = takerFee * usdt
            usdt = usdt - fee
            coin = 0
            buyReady = True
            wallet = usdt

            # -- Check if your wallet hit a new ATH to know the drawBack --
            if wallet > lastAth:
                lastAth = wallet

            # -- You can uncomment the line below if you want to see logs --  
            # print("Sell COIN at",sellPrice,'$ the', index)

            # -- Add the trade to DT to analyse it later --
            myrow = {'date': index,'position': "Sell", 'reason':'Sell Market Order','price': sellPrice,'frais': fee,'fiat': usdt,'coins': coin,'wallet': wallet,'drawBack':(wallet-lastAth)/lastAth}
            dt = dt.append(myrow,ignore_index=True)
        
        previousRow = row

    # -- BackTest Analyses --
    dt = dt.set_index(dt['date'])
    dt.index = pd.to_datetime(dt.index)
    dt['resultat'] = dt['wallet'].diff()
    dt['resultat%'] = dt['wallet'].pct_change()*100
    dt.loc[dt['position']=='Buy','resultat'] = None
    dt.loc[dt['position']=='Buy','resultat%'] = None

    dt['tradeIs'] = ''
    dt.loc[dt['resultat']>0,'tradeIs'] = 'Good'
    dt.loc[dt['resultat']<=0,'tradeIs'] = 'Bad'

    iniClose = dfTest.iloc[0]['Close']
    lastClose = dfTest.iloc[len(dfTest)-1]['Close']
    holdPercentage = ((lastClose - iniClose)/iniClose) * 100
    algoPercentage = ((wallet - initalWallet)/initalWallet) * 100
    vsHoldPercentage = ((algoPercentage - holdPercentage)/holdPercentage) * 100

    try:
        tradesPerformance = round(dt.loc[(dt['tradeIs'] == 'Good') | (dt['tradeIs'] == 'Bad'), 'resultat%'].sum()
                / dt.loc[(dt['tradeIs'] == 'Good') | (dt['tradeIs'] == 'Bad'), 'resultat%'].count(), 2)
    except:
        tradesPerformance = 0
        print("/!\ There is no Good or Bad Trades in your BackTest, maybe a problem...")

    try:
        totalGoodTrades = dt.groupby('tradeIs')['date'].nunique()['Good']
        AveragePercentagePositivTrades = round(dt.loc[dt['tradeIs'] == 'Good', 'resultat%'].sum()
                                            / dt.loc[dt['tradeIs'] == 'Good', 'resultat%'].count(), 2)
        idbest = dt.loc[dt['tradeIs'] == 'Good', 'resultat%'].idxmax()
        bestTrade = str(
            round(dt.loc[dt['tradeIs'] == 'Good', 'resultat%'].max(), 2))
    except:
        totalGoodTrades = 0
        AveragePercentagePositivTrades = 0
        idbest = ''
        bestTrade = 0
        print("/!\ There is no Good Trades in your BackTest, maybe a problem...")

    try:
        totalBadTrades = dt.groupby('tradeIs')['date'].nunique()['Bad']
        AveragePercentageNegativTrades = round(dt.loc[dt['tradeIs'] == 'Bad', 'resultat%'].sum()
                                            / dt.loc[dt['tradeIs'] == 'Bad', 'resultat%'].count(), 2)
        idworst = dt.loc[dt['tradeIs'] == 'Bad', 'resultat%'].idxmin()
        worstTrade = round(dt.loc[dt['tradeIs'] == 'Bad', 'resultat%'].min(), 2)
    except:
        totalBadTrades = 0
        AveragePercentageNegativTrades = 0
        idworst = ''
        worstTrade = 0
        print("/!\ There is no Bad Trades in your BackTest, maybe a problem...")

    totalTrades = totalBadTrades + totalGoodTrades
    winRateRatio = (totalGoodTrades/totalTrades) * 100

    reasons = dt['reason'].unique()
    if optimization_process==False:
        print("Period : [" + str(dfTest.index[0]) + "] -> [" +
            str(dfTest.index[len(dfTest)-1]) + "]")
        print("Starting balance :", initalWallet, "$")

        print("\n----- General Informations -----")
        print("Final balance :", round(wallet, 2), "$")
        print("Performance vs US Dollar :", round(algoPercentage, 2), "%")
        print("Buy and Hold Performence :", round(holdPercentage, 2), "%")
        print("Performance vs Buy and Hold :", round(vsHoldPercentage, 2), "%")
        print("Best trade : +"+bestTrade, "%, the", idbest)
        print("Worst trade :", worstTrade, "%, the", idworst)
        print("Worst drawBack :", str(100*round(dt['drawBack'].min(), 2)), "%")
        print("Total fees : ", round(dt['frais'].sum(), 2), "$")

        print("\n----- Trades Informations -----")
        print("Total trades on period :",totalTrades)
        print("Number of positive trades :", totalGoodTrades)
        print("Number of negative trades : ", totalBadTrades)
        print("Trades win rate ratio :", round(winRateRatio, 2), '%')
        print("Average trades performance :",tradesPerformance,"%")
        print("Average positive trades :", AveragePercentagePositivTrades, "%")
        print("Average negative trades :", AveragePercentageNegativTrades, "%")
        dt[['wallet', 'price']].plot(subplots=True, figsize=(20, 10))
        print("\n----- Plot -----")
    else:
        return round(wallet, 2)

def fourrierExtrapolation(data_to_predict:np.array, n_predict:int,has_trend:bool=True):
    n = data_to_predict.size
    n_harm = 50                     # number of harmonics in model
    t = np.arange(0, n)
    if has_trend==True:
        p = np.polyfit(t, data_to_predict, 1)         # find linear trend in x
        x_notrend = data_to_predict - p[0] * t        # detrended x
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        f = fft.fftfreq(n)              # frequencies
        indexes = list(range(n))
        #indexes = range(n)
        # sort indexes by frequency, lower -> higher
        indexes.sort(key = lambda i: np.absolute(f[i]))
    
        t = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n   # amplitude
            phase = np.angle(x_freqdom[i])          # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        return restored_sig + p[0] * t   
    else:
        x_notrend = data_to_predict      # detrended x
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        f = fft.fftfreq(n)              # frequencies
        indexes = list(range(n))
        #indexes = range(n)
        # sort indexes by frequency, lower -> higher
        indexes.sort(key = lambda i: np.absolute(f[i]))
    
        t = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n   # amplitude
            phase = np.angle(x_freqdom[i])          # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        return restored_sig
