from datetime import datetime

import pytz


class Trade:
    def toDict(self,side:str,price:float,strategy:str,comment:str='',symbol:str='BTC') -> None:
        """_summary_

        Args:
            side (str): The 'BUY' or 'SELL' for that trade.
            price (float): The price of the trade.
            timestamp (int): The trade's timestamp.
            strategy (str): Strategy used for the trade : 'RSI_VWAP', 'Moving_MLR', 'Forrain', 'Trix'...
            symbol (str, optional): The symbol traded : BTC, ETH, SOL, UNI... Defaults to 'BTC'.
            comment (str, optional): Commment on the trade 'Take profit reached', 'Stop loss reached'... Defaults to ''.

        Returns:
            dict: The trade in dict format
        """
        
        self._side=side
        self._price=price
        self._timestamp=datetime.today().replace(microsecond=0,tzinfo=pytz.utc)
        self._strategy=strategy
        self._comment=comment
        self._symbol=symbol
        return dict({
                'side':self._side,
                'price':self._price,
                'timestamp':self._timestamp,
                'strategy':self._strategy,
                'comment':self._comment,
                'symbol':self._symbol})
     