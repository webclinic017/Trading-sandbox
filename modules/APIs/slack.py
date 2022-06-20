
import requests
import json

class SlackAPI:
    def __init__(self) -> None:
        self._url = "https://hooks.slack.com/services/T03E7D4K6H5/"
        with open('./APIs/slack_channels.json') as json_file:
            self._channel_dict = json.load(json_file)
        
    def sendMessage(self, msg:str,crypto_channel:str='BTC' )->bool:
        """This function uses the slack API to send message

        Args:
            msg (str): The message in a string format
            crypto_channel (str): The crypto channel where the message will be sent in a string format. Default to BTC.

        Returns:
            bool: Returns True if the message was successfully sent False if not
        """
        return True if requests.post(f'{self._url}{self._channel_dict[crypto_channel]}',headers={'Content-type': 'application/json' }, json={'text':msg}).text =="ok" else False
    