import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
from .filters import filterData

def loadFromDB(path:str, keep_timestamp:bool=True)->pd.DataFrame:
    df = pd.read_csv(path,names=['Date','Open','High','Low','Close','Volume'])
    df = df.iloc[1:]
    df['Date'] = df['Date'].apply(lambda x: int(str(x)[:-3]))
    df = df.astype(float)
    if keep_timestamp==True:
        df['Timestamp'] = df['Date'].astype(int)
    df['Date'] = df['Date'].astype(int).apply(datetime.fromtimestamp)
    df = df.set_index('Date')
    print(f"Total records : {len(df)} rows")
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


def strategyTester(df:pd.DataFrame,buyConditonFunc, sellConditionFunc, equity:int=1000, ):
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
            # stopLoss = buyPrice - 0.02 * buyPrice
            # takeProfit = buyPrice + 0.04 * buyPrice

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
            buyReady = False
            wallet = usdt

            # -- Check if your wallet hit a new ATH to know the drawBack --
            if wallet > lastAth:
                lastAth = wallet
            
            # -- You can uncomment the line below if you want to see logs --
            # print("Sell COIN at Stop Loss",sellPrice,'$ the', index)

            # -- Add the trade to DT to analyse it later --
            myrow = {'date': index,'position': "Sell", 'reason':'Sell Stop Loss','price': sellPrice,'frais': fee,'fiat': usdt,'coins': coin,'wallet': wallet,'drawBack':(wallet-lastAth)/lastAth}
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