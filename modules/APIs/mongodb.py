from curses import echo
import json
from pymongo import MongoClient
import pymongo

class mongoDB:
    def __init__(self,env:str='server') -> None:
        """Constructor of the mongoDB class.

        Args:
            env (str, optional): The current environnement. Defaults to 'server'.
        """
        with open('./modules/APIs/slack_channels.json') as json_file:
            mongo_credentials = json.load(json_file)
        if env=='cloud':
            CONNECTION_STRING = f"mongodb+srv://{mongo_credentials['cloud']['user']}:{mongo_credentials['cloud']['password']}@{mongo_credentials['cloud']['host']}" #cloud
        elif env=='local':
            CONNECTION_STRING = f"mongodb://{mongo_credentials['local']['user']}:{mongo_credentials['local']['password']}@{mongo_credentials['local']['host']}:{mongo_credentials['local']['port']}/?authSource=admin&readPreference=primary&ssl=false" #local
        else:
            CONNECTION_STRING = f"mongodb://{mongo_credentials['prod']['user']}:{mongo_credentials['prod']['password']}@{mongo_credentials['prod']['host']}:{mongo_credentials['prod']['port']}/?authSource=admin&readPreference=primary&ssl=false" #server
        mongo_client = MongoClient(CONNECTION_STRING)
        self._db = mongo_client['cryptos']
        #try:
        #    self.createCollection('trade',True)
        #except:
        #    return
        
    def createCollection(self, collection_name:str,timeseries:bool=True)->None:
        """_summary_

        Args:
            collection_name (str): _description_
            timeseries (bool, optional): _description_. Defaults to True.
        """
        if timeseries ==True:
            self._db.create_collection(collection_name,timeseries={ 'timeField': 'timestamp' })
        else:
            self._db.create_collection(collection_name)
            
    def insertInCollection(self,data:dict or list(dict),many:bool=False)->None:
        """_summary_

        Args:
            data (dict): _description_
            collection (str): _description_
            many (bool, optional): _description_. Defaults to False.
        """
        if many==False:
            self._db[u'trade'].insert_one(data)
        else:
            self._db[u'trade'].insert_many(data)
        