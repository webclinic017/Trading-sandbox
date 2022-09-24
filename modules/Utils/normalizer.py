from typing import Union
from joblib import dump, load
import pandas as pd
import numpy as np

class StandardNormalizer():
    def __init__(self)->None:
        self._mean = None
        self._std = None

    def fit_transform(self,df:pd.DataFrame)->pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def fit(self, df:pd.DataFrame)->None:
        self._mean = df.mean()
        self._std = df.std()

    def transform(self, df:pd.DataFrame)->pd.DataFrame:
        return (df - self._mean) / self._std

    def inverse_transform(self,df:pd.DataFrame)->pd.DataFrame:
        return df * self._std + self._mean

    def inverse_transform_single(self,data:Union[pd.DataFrame,np.ndarray],corresponding_column:str='Close')->pd.DataFrame:
        return data * self._std[corresponding_column] + self._mean[corresponding_column]

class VectorNormalizer():
    def __init__(self)->None:
        self._denominator = None

    def fit_transform(self,df:pd.DataFrame)->pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def fit(self, df:pd.DataFrame)->None:
        self._denominator = np.sqrt(df.pow(2).sum())

    def transform(self, df:pd.DataFrame)->pd.DataFrame:
        return df / self._denominator

    def inverse_transform(self,df:pd.DataFrame)->pd.DataFrame:
        return df * self._denominator

    def inverse_transform_single(self,data:Union[pd.DataFrame,np.ndarray],corresponding_column:str='Close')->pd.DataFrame:
        return data * self._denominator[corresponding_column] 

        
class MedianNormalizer():
    def __init__(self)->None:
        self._median = None

    def fit_transform(self,df:pd.DataFrame)->pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def fit(self, df:pd.DataFrame)->None:
        self._median = df.median()

    def transform(self, df:pd.DataFrame)->pd.DataFrame:
        return df / self._median

    def inverse_transform(self,df:pd.DataFrame)->pd.DataFrame:
        return df * self._median

    def inverse_transform_single(self,data:Union[pd.DataFrame,np.ndarray],corresponding_column:str='Close')->pd.DataFrame:
        return data * self._median[corresponding_column] 