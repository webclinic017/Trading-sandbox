import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

# Technical indicators
from ta.momentum import stochrsi,rsi
from ta.trend import ema_indicator, macd_diff, vortex_indicator_neg, vortex_indicator_pos, adx, cci, sma_indicator
from ta.volatility import bollinger_hband, bollinger_lband
from ta.volume import volume_weighted_average_price, ease_of_movement

from .filters import filterData

def addIndicators(df:pd.DataFrame,b_engulfings:bool=False, derivative:bool=False, double_derivative:bool=False,heikin_ashi:bool=False,chandelier_exit:bool=False, **kwargs) -> pd.DataFrame:
    """Apply indicators to the DataFrame.

    Args:
        df (pd.DataFrame): The dataframe you want to add indicators on.
        b_engulfings (bool, optional): Add bearish and bullish engulfing indicators. Defaults to False.
        derivative (bool, optional): Add the first derivative of the Close price. Defaults to False.
        double_derivative (bool, optional): Add the second derivative of the Close price. Defaults to False.
        heikin_ashi (bool, optional): Add the heikin_ashi candle to the dataframe. Defaults to False.
        chandelier_exit (bool, optional): Add the CE signal short and long to the dataframe. Defaults to False.
        **kwargs

    Returns:
        pd.DataFrame: The same dataframe with indicators
    """
    df['High_Low_diff'] = df.High-df.Low
    df['EMA20'] = ema_indicator(df.Close,20)
    df['EMA50'] = ema_indicator(df.Close,50)
    df['EMA100'] = ema_indicator(df.Close,100)
    df['EMA200'] = ema_indicator(df.Close,200)
    df['MACD'] = macd_diff(df.Close,kwargs.get('macd_length',14))
    df['Stoch_RSI'] = stochrsi(df.Close, kwargs.get('rsi_length',14), smooth1=3, smooth2=3)
    df['Vortex'] = (vortex_indicator_pos(df.High,df.Low,df.Close,kwargs.get('vortex_length',20),fillna=True)-1)-(vortex_indicator_neg(df.High,df.Low,df.Close,kwargs.get('vortex_length',20),fillna=True)-1)
    df['Bollinger_low'] = bollinger_hband(df.Close,kwargs.get('bollinger_length',20),fillna=True)
    df['Bollinger_high'] = bollinger_lband(df.Close,kwargs.get('bollinger_length',20),fillna=True)
    df['ADX'] = adx(df.High,df.Low,df.Close,kwargs.get('adx_length',14))
    df['ATR'] = adx(df.High,df.Low,df.Close,kwargs.get('atr_length',22))
    df['CCI'] = cci(df.High,df.Low,df.Close,kwargs.get('cci_length',14))
    df['OVB'] = (np.sign(df.Close.diff())*df.Volume).fillna(0).cumsum()
    df['OVB_EMA200'] = ema_indicator(df.OVB,200)
    df['EVM'] = ease_of_movement(df.High,df.Low,df.Volume,kwargs.get('evm_length',14))
    if chandelier_exit==True:
        df['CE_long'] = np.nan
        df['CE_short'] = np.nan
        df['CE_long'].iloc[kwargs.get('atr_length',22):] = [df['High'].iloc[i-kwargs.get('atr_length',22):i].max()-df['ATR'].iloc[i]*3 for i in range(kwargs.get('atr_length',22),len(df))]
        df['CE_short'].iloc[kwargs.get('atr_length',22):] = [df['Low'].iloc[i-kwargs.get('atr_length',22):i].min()+df['ATR'].iloc[i]*3 for i in range(kwargs.get('atr_length',22),len(df))]
    if b_engulfings==True:   
        def isBearishCandleStick(candle) -> bool:
            """Check whether a candle is a bearish candle or not

            Args:
                candle (pd.Series): The current candle that contains OHLC

            Returns:
                bool: A boolean representing if the candle is bearish candle (True) or not (False)
            """
            return candle['Close']<candle['Open']

        def isBullishCandleStick(candle) -> bool:
            """Check whether a candle is a bullish candle or not

            Args:
                candle (pd.Series): The current candle that contains OHLC

            Returns:
                bool: A boolean representing if the candle is bullish candle (True) or not (False)
            """
            return candle['Close']>candle['Open']

        def isBullishEngulfing(previous_candle,current_candle) -> int:
            """A function that check for bullish engulfing pattern through candle stick

            Args:
                previous_candle (pd.Series): The previous candle that contains OHLC
                current_candle (pd.Series): The current candle that contains OHLC

            Returns:
                int: represent the pattern spotting : 1 bullish engulfing, 0 not.
            """
            return 1 if isBearishCandleStick(previous_candle) and isBullishCandleStick(current_candle) and previous_candle['Open']<current_candle['Close'] and previous_candle['Close']>current_candle['Open'] else 0
            
        def isBearishEngulfing(previous_candle,current_candle) -> int:
            """A function that check for bearish engulfing pattern through candle stick

            Args:
                previous_candle (pd.Series): The previous candle that contains OHLC
                current_candle (pd.Series): The current candle that contains OHLC

            Returns:
                int: represent the pattern spotting : 1 bearish engulfing, 0 not.
            """
            return 1 if isBullishCandleStick(previous_candle) and isBearishCandleStick(current_candle) and previous_candle['Close']<current_candle['Open'] and previous_candle['Open']>current_candle['Close'] else 0    
                
        df['Bullish_engulfing'] = np.nan
        df['Bullish_engulfing'].iloc[1:] = [isBullishEngulfing(df.iloc[i-1],df.iloc[i]) for i in range(1,len(df))]
        df['Bearish_engulfing'] = np.nan
        df['Bearish_engulfing'].iloc[1:] = [isBearishEngulfing(df.iloc[i-1],df.iloc[i]) for i in range(1,len(df))]
    if derivative==True:
        df['Slope'] = df.Close.diff()
    if double_derivative==True:
        df['Acceleration'] = df.Close.diff().diff()
    if heikin_ashi==True:
        df['HA_Close'] = (df.Open + df.High + df.Low + df.Close)/4
        ha_open = [(df.Open[0] + df.Close[0]) / 2]
        [ha_open.append((ha_open[i] + df.HA_Close.values[i]) / 2) for i in range(0, len(df)-1)]
        df['HA_Open'] = ha_open
        df['HA_High'] = df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
        df['HA_Low'] = df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)
    return df.dropna()

def computeTrixIndicator(df,trix_length=9,trix_signal=21,col='Close',histo=True)->pd.DataFrame:
    """_summary_

    Args:
        df (_type_): _description_
        trix_length (_type_): _description_
        trix_signal (_type_): _description_
        col (str, optional): _description_. Defaults to 'Close'.
        histo (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DateFrame: _description_
    """
    df['Trix'] = ema_indicator(ema_indicator(ema_indicator(df[col], window=trix_length), window=trix_length), window=trix_length)
    trix_pct = df['Trix'].pct_change()*100
    trix_sig = sma_indicator(trix_pct,trix_signal)
    if histo==True:
        df['Trix_histo'] = trix_pct - trix_sig
    return df.dropna()


def computeLaggingLinearRegression(df, col="Close",window=15,filter_ceof:bool=True, filter_method:str='dwt',derivative:bool=True)->pd.DataFrame:
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
    
    df['B_MLR_coefs'] = np.nan
    df['B_MLR_coefs'].iloc[window:] = [computeLinearRegression(df.Timestamp.values[i-window:i].reshape(-1, 1), df[col].values[i-window:i]) for i in range(window,len(df))] 
    df = df.dropna()
    if filter_ceof==True:
        df['B_MLR_coefs_filtered'] = filterData(df.B_MLR_coefs.values,filter_method)
        if derivative==True:
            df['B_MLR_coefs_filtered_diff'] = df['B_MLR_coefs_filtered'].diff()
    else:
        if derivative==True:
            df['B_MLR_coefs_diff'] = df['B_MLR_coefs'].diff()
    return df.dropna()


def computeRSI_VWAP(df, rsi_window=25,vwap_window=19)->pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): The dataframe containing features.
        rsi_window (int, optional): The window we apply VWAP indicator on. Defaults to 25.
        vwap_window (int, optional): The window we apply RSI indicator on. Defaults to 19.

    Returns:
        pd.DataFrame: The entry DataFrame with RSI_VWAP indicator column added.
    """  
    df['RSI_VWAP'] = rsi(volume_weighted_average_price(df.High,df.Low,df.Close,df['Volume'],vwap_window),rsi_window)
    return df.dropna()


def computeVMCCB(df:pd.DataFrame)->pd.DataFrame:
    class VMC():
        """ VuManChu Cipher B + Divergences 
            Args:
                High(pandas.Series): dataset 'High' column.
                Low(pandas.Series): dataset 'Low' column.
                Close(pandas.Series): dataset 'Close' column.
                wtChannelLen(int): n period.
                wtAverageLen(int): n period.
                wtMALen(int): n period.
                rsiMFIperiod(int): n period.
                rsiMFIMultiplier(int): n period.
                rsiMFIPosY(int): n period.
        """

        def __init__(
            self: pd.Series,
            Open: pd.Series,
            High: pd.Series,
            Low: pd.Series,
            Close: pd.Series,
            wtChannelLen: int = 9,
            wtAverageLen: int = 12,
            wtMALen: int = 3,
            rsiMFIperiod: int = 60,
            rsiMFIMultiplier: int = 150,
            rsiMFIPosY: int = 2.5
        ) -> None:
            self._high = High
            self._low = Low
            self._close = Close
            self._open = Open
            self._wtChannelLen = wtChannelLen
            self._wtAverageLen = wtAverageLen
            self._wtMALen = wtMALen
            self._rsiMFIperiod = rsiMFIperiod
            self._rsiMFIMultiplier = rsiMFIMultiplier
            self._rsiMFIPosY = rsiMFIPosY

            self._run()
            self.wave_1()

        def _run(self) -> None:
            self.hlc3 = (self._close + self._high + self._low)
            self._esa = ema_indicator(
                Close=self.hlc3, window=self._wtChannelLen)
            self._de = ema_indicator(
                Close=abs(self.hlc3 - self._esa), window=self._wtChannelLen)
            self._rsi = sma_indicator(self._close, self._rsiMFIperiod)
            self._ci = (self.hlc3 - self._esa) / (0.015 * self._de)

        def wave1(self) -> pd.Series:
            """VMC Wave 1 
            Returns:
                pandas.Series: New feature generated.
            """
            wt1 = ema_indicator(self._ci, self._wtAverageLen)
            return pd.Series(wt1, name="wt1")

        def wave2(self) -> pd.Series:
            """VMC Wave 2
            Returns:
                pandas.Series: New feature generated.
            """
            wt2 = sma_indicator(self.wave_1(), self._wtMALen)
            return pd.Series(wt2, name="wt2")

        def moneyFlow(self) -> pd.Series:
            """VMC Money Flow
            Returns:
                pandas.Series: New feature generated.
            """
            mfi = ((self._close - self._open) /
                (self._high - self._low)) * self._rsiMFIMultiplier
            rsi = sma_indicator(mfi, self._rsiMFIperiod)
            money_flow = rsi - self._rsiMFIPosY
            return pd.Series(money_flow, name="money_flow")
        
    vmc = VMC(df.Open, df.High, df.Low, df.Close)
    df['Money_Flow'] = vmc.moneyFlow()
    df['Wave1'] = vmc.wave1()
    df['Wave2'] = vmc.wave2()
    def chop(high, low, close, window=14):
        """ Chopiness index
            Args:
                high(pd.Series): dataframe 'high' columns,
                low(pd.Series): dataframe 'low' columns,
                close(pd.Series): dataframe 'close' columns,
                window(int): the window length for the chopiness index,
            Returns:
                pd.Series: Chopiness index
        """
        tr1 = pd.DataFrame(high - low).rename(columns = {0:'tr1'})
        tr2 = pd.DataFrame(abs(high - close.shift(1))).rename(columns = {0:'tr2'})
        tr3 = pd.DataFrame(abs(low - close.shift(1))).rename(columns = {0:'tr3'})
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').dropna().max(axis = 1)
        atr = tr.rolling(1).mean()
        highh = high.rolling(window).max()
        lowl = low.rolling(window).min()
        return 100 * np.log10((atr.rolling(window).sum()) / (highh - lowl)) / np.log10(window)
    return df

def computeSuperTrend(df:pd.DataFrame,upper=False, lower=False)->pd.DataFrame:
    """Compute the super trend indicator for a dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The resulting dataframe with 3 new columns
    """
    class SuperTrend():
        def __init__(
            self,
            High,
            Low,
            Close,
            atr_window=10,
            atr_multi=3
        ):
            self.High = High
            self.Low = Low
            self.Close = Close
            self.atr_window = atr_window
            self.atr_multi = atr_multi
            self._run()
            
        def _run(self):
            # calculate ATR
            price_diffs = [self.High - self.Low, 
                        self.High - self.Close.shift(), 
                        self.Close.shift() - self.Low]
            true_range = pd.concat(price_diffs, axis=1)
            true_range = true_range.abs().max(axis=1)
            # default ATR calculation in supertrend indicator
            atr = true_range.ewm(alpha=1/self.atr_window,min_periods=self.atr_window).mean() 
            # atr = ta.volatility.average_true_range(High, Low, Close, atr_period)
            # df['atr'] = df['tr'].rolling(atr_period).mean()
            
            # HL2 is simply the average of High and Low prices
            hl2 = (self.High + self.Low) / 2
            # upperband and lowerband calculation
            # notice that final bands are set to be equal to the respective bands
            final_upperband = upperband = hl2 + (self.atr_multi * atr)
            final_lowerband = lowerband = hl2 - (self.atr_multi * atr)
            
            # initialize Supertrend column to True
            supertrend = [True] * len(self.Close)
            
            for i in range(1, len(self.Close)):
                curr, prev = i, i-1
                
                # if current Close price crosses above upperband
                if self.Close[curr] > final_upperband[prev]:
                    supertrend[curr] = True
                # if current Close price crosses below lowerband
                elif self.Close[curr] < final_lowerband[prev]:
                    supertrend[curr] = False
                # else, the trend continues
                else:
                    supertrend[curr] = supertrend[prev]
                    
                    # adjustment to the final bands
                    if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                        final_lowerband[curr] = final_lowerband[prev]
                    if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                        final_upperband[curr] = final_upperband[prev]

                # to remove bands according to the trend direction
                if supertrend[curr] == True:
                    final_upperband[curr] = np.nan
                else:
                    final_lowerband[curr] = np.nan
                    
            self.st = pd.DataFrame({
                'Supertrend': supertrend,
                'Final Lowerband': final_lowerband,
                'Final Upperband': final_upperband
            })
            
        def superTrendUpper(self):
            return self.st['Final Upperband']
            
        def superTrendLower(self):
            return self.st['Final Lowerband']
            
        def superTrendDirection(self):
            return self.st['Supertrend']
        
    st = SuperTrend(df.High, df.Low, df.Close)
    
    if upper==True:
        df['ST_Upper']  = st.superTrendUpper()
    if lower==True:
        df['ST_Lower'] = st.superTrendLower()
    df['ST_Direction'] = st.superTrendDirection()
    return df


def computeMASlope(df_true:pd.DataFrame)->pd.DataFrame:
    """Compute MA Slope indicator on a dataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The output dataframe with 2 new columns.
    """
    class MaSlope():
        """ Slope adaptative moving average
        """

        def __init__(
            self,
            Close: pd.Series,
            High: pd.Series,
            Low: pd.Series,
            long_ma: int = 200,
            major_length: int = 14,
            minor_length: int = 6,
            slope_period: int = 34,
            slope_ir: int = 25
        ):
            self.Close = Close
            self.High = High
            self.Low = Low
            self.long_ma = long_ma
            self.major_length = major_length
            self.minor_length = minor_length
            self.slope_period = slope_period
            self.slope_ir = slope_ir
            self._run()

        def _run(self):
            minAlpha = 2 / (self.minor_length + 1)
            majAlpha = 2 / (self.major_length + 1)
            # df = pd.DataFrame(data = [self.Close, self.High, self.Low], columns = ['Close','High','Low'])
            df = pd.DataFrame(data = {"Close": self.Close, "High": self.High, "Low":self.Low})
            df['hh'] = df['High'].rolling(window=self.long_ma+1).max()
            df['ll'] = df['Low'].rolling(window=self.long_ma+1).min()
           
            df = df.fillna(0)
            df['mult'] = 0
            df['mult'].iloc[self.long_ma:] = abs(2 * df['Close'].iloc[self.long_ma:] - df['ll'].iloc[self.long_ma:] - df['hh'].iloc[self.long_ma:]) / (df['hh'].iloc[self.long_ma:] - df['ll'].iloc[self.long_ma:])
            df['final'] = df['mult'] * (minAlpha - majAlpha) + majAlpha

            ma_first = (df.iloc[0]['final']**2) * df.iloc[0]['Close']

            col_ma = [ma_first]
            for i in range(1, len(df)):
                ma1 = col_ma[i-1]
                col_ma.append(ma1 + (df.iloc[i]['final']**2) * (df.iloc[i]['Close'] - ma1))

            df['ma'] = col_ma
            pi = math.atan(1) * 4
            df['hh1'] = df['High'].rolling(window=self.slope_period).max()
            df['ll1'] = df['Low'].rolling(window=self.slope_period).min()
            df['slope_range'] = self.slope_ir / (df['hh1'] - df['ll1']) * df['ll1']
            df['dt'] = (df['ma'].shift(2) - df['ma']) / df['Close'] * df['slope_range'] 
            df['c'] = (1+df['dt']*df['dt'])**0.5
            df['xangle'] = round(180*np.arccos(1/df['c']) / pi)
            df[df['dt']>0]['xangle'] = - df[df['dt']>0]['xangle']
            self.df = df
            # print(df)

        def ma_line(self) -> pd.Series:
            """ ma_line
                Returns:
                    pd.Series: ma_line
            """
            return self.df['ma']

        def x_angle(self) -> pd.Series:
            """ x_angle
                Returns:
                    pd.Series: x_angle
            """
            return self.df['xangle']
            
        
    ms = MaSlope(df_true.High, df_true.Low, df_true.Close)
    df_true['Angle'] = ms.x_angle()
    df_true['MA_Slope'] = ms.ma_line()
    return df_true