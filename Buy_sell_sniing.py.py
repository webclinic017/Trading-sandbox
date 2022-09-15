import time
import ccxt

class KucoinAPI:
    def __init__(self) -> None:
        self._credentials = {
            "apiKey":"62a05a9ccf9fd60001e2ff9e",
            "secret":"578687f6-5420-46d7-9b7a-26a737713cfb",
            "password":"3FiF$4sR6p5Y9fPnSNijL?HaGcd@A7"
        }
        self._session = ccxt.kucoin(self._credentials)
        self.market = self._session.load_markets()
        #self._session.set_sandbox_mode(True)

    
    def authentication_required(fn):
        """Annotation for methods that require auth."""
        def wrapped(self, *args, **kwargs):
            if not self._credentials or "apiKey" not in self._credentials or "secret" not in self._credentials or "password" not in self._credentials: 
                raise Exception(f"You must be authenticated to use this method {fn}") 
                #sys.exit()
            else:
                return fn(self, *args, **kwargs)
        return wrapped
        
    @authentication_required    
    def placeMarketOrder(self,symbol:str,side:str,amount:str)->dict:
        assert side.lower() in ["buy","sell"]
        return self._session.create_market_order(f'{symbol}/USDT', side.lower(), amount)

    
    @authentication_required     
    def placeLimitOrder(self):
        pass  
    
    @authentication_required   
    def cancelOrder(self, orderId:str)->dict:
        return self._session.cancel_order(orderId)
    
    @authentication_required
    def getUSDTBalance(self)->float:
        try:
            for coin in self._session.fetchBalance()['info']['data']:
                if coin['currency']=='USDT' :
                    return float(coin['balance'])
        except Exception as err:
            raise err
        
    def getCurrentPrice(self,symbol:str)->float:
        try:
            return self._session.fetch_ticker(f'{symbol}/USDT')["ask"]
        except Exception as err:
            raise err
        
    def reloadMarkets(self):
        self.market = self._session.load_markets()
        
    #def getAmountForSellingAll(self,symbol:str)->str:
    #    return self._session.amount_to_precision(f'{symbol}/USDT', self.qty_in_position)
    
    def getAmountForTrade(self, symbol:str,pct_wallet:int=1)->str:
        self.reloadMarkets()
        if self.market[f'{symbol}/USDT']['limits']['amount']['min']<self.getUSDTBalance()*pct_wallet/self.getCurrentPrice(f'{symbol}'):
            return self._session.amount_to_precision(f'{symbol}/USDT', self.getUSDTBalance()*pct_wallet/self.getCurrentPrice(f'{symbol}'))
        else:
            raise Exception(f"Not enough USDT to trade {symbol}") 
        
        
kucoin_client = KucoinAPI()
USD_balance = kucoin_client.getUSDTBalance()
print(f"Total USD balance: {USD_balance}, USD in the trade: {USD_balance*0.01}")
amount = kucoin_client.getAmountForTrade('CMP',0.01)
print(amount)
print('Beginning')
try:
    print(kucoin_client.placeMarketOrder('CMP',
                                         'buy',
                                         amount))
    print('Buy trade placed')
    time.sleep(10)
    print(kucoin_client.placeMarketOrder('CMP',
                                        'sell',
                                         amount))
    print('Sell trade placed')
except:
    print('In the catch')
    print('Everything sold !')