import dask.dataframe as dd
from langdetect import detect

import re

import warnings 
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='app.log', filemode='w', format='%(levelname)s - %(message)s',level=logging.DEBUG)

def cleanTweets(txt):
    txt = str(txt)
    txt = re.sub(r'#','',txt)
    txt = re.sub(r'\n','',txt)
    txt = re.sub(r'https?:\/\/\S+','',txt)
    txt = re.sub(r'@','',txt)
    txt = re.sub(r'[^a-zA-Z0-9\\s!"#$€£%&\'()*+,-.\\/:;<=>?@[\\]^_`{|}~]','',txt)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', txt).lower()


from langdetect import detect

def detectLang(txt):
    try :
        return detect(txt)
    except :
        #print(txt)
        return "none"
    
    
def SentimentToCode(txt):
    if txt=="Positive":
      return 2
    elif txt=="Negative":
      return 0
    else:
      return 1 
  
  
if __name__ == "__main__":
    ddf = dd.read_parquet('./mbsa/mbsa_SHARD_1.parquet')
    logging.debug('File loaded')
    ddf['Sentiment'] = ddf['Sentiment'].apply(SentimentToCode,meta=('Sentiment','str'))
    logging.debug('Sentiment filtered')
    ddf['text'] = ddf['text'].apply(cleanTweets,meta=('text','str'))
    logging.debug('Text cleaned')
    ddf = ddf[ddf['text']!='']
    logging.debug('Empty text removed')
    ddf['lang'] = ddf['text'].apply(detectLang,meta=('text','str'))
    logging.debug(f'Lang added')
    ddf = ddf[ddf['lang']=='en']
    logging.debug(f'Lang removed')
    ddf.compute().to_csv('./mbsa_SHARD1_cleaned.csv',index=False)
    logging.debug('csv written !')
    ddf.compute().to_parquet('./mbsa_SHARD1_cleaned.parquet')
    logging.debug('Finished !')