import pandas as pd
import sys
import numpy as np
import datetime

reload(sys)
sys.setdefaultencoding('utf8')

def date_range(start_date,end_date):
    for n in range(int((end_date-start_date).days)):
        yield start_date+datetime.timedelta(n)


def get_sentiment(d, td):
    for idx, da in enumerate(td['date']):
        if d == da:
            return td['compound'][idx]
    return 0


def get_pct_change(d,sd):
    for idx, da in enumerate(sd['date']):
        if d == da:
            return sd['pct_change'][idx - 2], sd['pct_change'][idx - 1], sd['pct_change'][idx]
    return -1, -1, -1

stock_codes = ['AAPL','CSCO','INCT','MSFT']
for stock_code in stock_codes:
    tweet_path = '../data/tweets_compound/' + stock_code + '_compound.csv'
    stock_path = '../data/stock/' + stock_code + '.csv'
    out_path = '../data/train_data/' + stock_code + '.csv'
    out_txt_path = '../data/train_data/' + stock_code + '.txt'

    tweets_data = pd.read_csv(tweet_path)
    stock_data = pd.read_csv(stock_path)





    date_list = []
    last_sentiment_list = []
    last_pct_change_list = []
    last_two_sentiment_list = []
    last_two_pct_change_list = []
    label_list = []

    start = datetime.datetime(2016,4,1,0,0,0)
    end = datetime.datetime(2016,6,16,0,0,0)
    for i in date_range(start, end):
        date = i.strftime('%Y-%m-%d')
        last_date = (i + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        last_two_date = (i + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        last_two_pct_change, last_pct_change, pct_change = get_pct_change(date, stock_data)
        if last_pct_change != -1:
            last_sentiment = get_sentiment(last_date, tweets_data)
            last_two_sentiment = get_sentiment(last_two_date, tweets_data)
            date_list.append(date)
            last_sentiment_list.append(float(last_sentiment))
            last_pct_change_list.append(last_pct_change)
            last_two_sentiment_list.append(float(last_two_sentiment))
            last_two_pct_change_list.append(last_two_pct_change)
            label_list.append(1.0 if pct_change > 0 else -1.0)

    df2 = pd.DataFrame({
            'date': pd.Series(date_list),
            'sentiment': pd.Series(last_sentiment_list),
            'pct_change': pd.Series(last_pct_change_list),
            'last_sentiment': pd.Series(last_two_sentiment_list),
            'last_pct_change': pd.Series(last_two_pct_change_list),
            'label': pd.Series(label_list)
        })
    df2.to_csv(out_path,
               index=False,
               columns=['date', 'last_sentiment', 'last_pct_change', 'sentiment', 'pct_change', 'label'],
               encoding='utf-8')

    with open(out_txt_path,'w') as fout:
        for i in range(len(date_list)):
            fout.write('%f,%f,%f,%f,%f\n'%(last_sentiment_list[i], last_pct_change_list[i], \
                                       last_two_sentiment_list[i],last_two_pct_change_list[i],label_list[i]))