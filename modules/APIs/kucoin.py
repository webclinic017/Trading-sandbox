import json
import os
from kucoin.client import Trade, Market, User

class KucoinAPI:
    def __init__(self,sandbox:bool=True) -> None:
        with open('/var/www/crypto-data-alert/APIs/kucoin_keys.json') as json_file:
            self._credentials = json.load(json_file)
        self._trade_client = Trade(key=self._credentials['api_key'], secret=self._credentials['api_secret'], passphrase=self._credentials['api_passphrase'], is_sandbox=sandbox)
        self._market_client = Market(url='https://api.kucoin.com')
        self._user_client = User(key=self._credentials['api_key'], secret=self._credentials['api_secret'], passphrase=self._credentials['api_passphrase'])
        
    def buyLimitOrder(self, buy_trigger_price:float, symbol:str='BTC',size:int=0.002)->dict:
        clientOid=self.generateOrderId()
        return dict({'orderId':self._trade_client.create_limit_order(f'{symbol}-USDT', 'buy', size=size, price=str(buy_trigger_price),clientOid=clientOid),"clientOid":clientOid})
    
    def sellLimitOrder(self, sell_trigger_price:float, symbol:str='BTC',size:int=0.002)->dict:
        clientOid=self.generateOrderId()
        return dict({'orderId':self._trade_client.create_limit_order(f'{symbol}-USDT',side='sell', size=size, price=str(sell_trigger_price),clientOid=clientOid),"clientOid":clientOid})
    
    # Market order (direct way)
    def buyOrder(self, symbol:str='BTC',size:int=0.002)->dict:
        clientOid=self.generateOrderId()
        return dict({'orderId':self._trade_client.create_market_order(f'{symbol}-USDT',side='buy',clientOid=clientOid,size=size),"clientOid":clientOid})
    
    # Market order (direct way)
    def sellOrder(self, symbol:str='BTC',size:int=0.002)->dict:
        clientOid=self.generateOrderId()
        return dict({'orderId':self._trade_client.create_market_order(f'{symbol}-USDT',side='sell',clientOid=clientOid,size=size),"clientOid":clientOid})
    
    
    # Utils function for this class
    def generateOrderId()->str:
        """method that generate a unique order id with random char.

        Returns:
            str: The clientOId.
        """
        #max_char = 40
        return os.urandom(14).hex()  
          
    def getOrderInfo(self,orderId:str)->dict:
        return dict(self._trade_client.get_order_details(orderId))