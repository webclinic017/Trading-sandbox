from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import load, dump

class CustomStandardScaler(StandardScaler):
    def load(self, path:str)-> StandardScaler:
        """Method that extends the StandardScaler class with the aim to load the scaler to a python object.

        Args:
            path (str): The path where the StandardScaler trained is located.

        Returns:
            StandardScaler: The loaded scaler.
        """
        return load(path)
    
    def save(self,scaler:StandardScaler, path:str) -> None:
        """Method that extends the StandardScaler class with the aim to save the scaler to a .save file.

        Args:
            scaler (StandardScaler): The scaler to save.
            path (str): The path where to save the StandardScaler trained.
        """
        dump(scaler,path)
        
        
class CustomMinMaxScaler(MinMaxScaler):
    def load(self, path:str)-> MinMaxScaler:
        """Method that extends the MinMaxScaler class with the aim to load the scaler to a python object.

        Args:
            path (str): The path where the MinMaxScaler trained is located.

        Returns:
            MinMaxScaler: The loaded scaler.
        """
        return load(path)
    
    def save(self,scaler:MinMaxScaler, path:str) -> None:
        """Method that extends the MinMaxScaler class with the aim to save the scaler to a .save file.

        Args:
            scaler (MinMaxScaler): The scaler to save.
            path (str): The path where to save the MinMaxScaler trained.
        """
        dump(scaler,path)