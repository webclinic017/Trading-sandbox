import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=4,progress_bar=True)

df = pd.read_csv('./depot/AUDUSD.txt',dtype={'<TICKER>':str,'<OPEN>':float,'<HIGH>':float,'<LOW>':float,'<CLOSE>':float,'<VOL>':int,'<TIME>':str})
df = df[df['<TIME>']!='0']
df['<DTYYYYMMDD>'] = pd.to_datetime(df['<DTYYYYMMDD>'],format='%Y%m%d')
df['<TIME>'] = pd.to_datetime(df['<TIME>'],format='%H%M%S' )
print(df.shape)

def date_str_wrangling(row:pd.Series)->str:
    date_str = f"{row['<DTYYYYMMDD>']} {row['<TIME>']}"
    return date_str[:11]+date_str[-8:]

df['Date'] = pd.to_datetime(df.parallel_apply(date_str_wrangling,axis=1),format= '%Y-%m-%d %H:%M:%S')
df.drop(columns=['<TICKER>','<DTYYYYMMDD>','<TIME>'],inplace=True)
df.rename({'<OPEN>': 'Open', '<CLOSE>': 'Close','<HIGH>':'High','<LOW>':'Low','<VOL>':'Volume'}, axis=1, inplace=True)
df = df.set_index('Date')
df['Timestamp'] = [int(str(date)[:10]) for date in df.index.astype('int')]

df.to_csv("./data/AUDUSD.csv",index=False,sep=',')