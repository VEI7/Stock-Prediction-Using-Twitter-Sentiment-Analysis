import pandas as pd
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')

stock_codes = ['AAPL','CSCO','INCT','MSFT']
for stock_code in stock_codes:
    tweets_path = '../data/tweets/'+stock_code+'_tweets.csv'
    prediction_path = '../data/tweets_for_Sentiment/'+stock_code+'_predictions.txt'
    out_path = '../data/tweets_compound/'+stock_code+'_compound.csv'

    tweets_data = pd.read_csv(tweets_path)
    sentiment_list = []
    compound_dict = dict()
    with open(prediction_path, 'r') as prediction_file:
        for line in prediction_file.readlines():
            info = line.strip().split('\t')
            if len(info) == 3:
                sentiment_list.append(info[1])

    for idx,date in enumerate(tweets_data['Date']):
        if date not in compound_dict:
            compound_dict[date] = {'positive': 0, 'neutral': 0, 'negative': 0}
        if not np.isnan(tweets_data['Following'][idx]):
            compound_dict[date][sentiment_list[idx]] += int(tweets_data['Following'][idx])
        else:
            compound_dict[date][sentiment_list[idx]] += 1

    date_list = []
    positive_list = []
    neutral_list = []
    negative_list = []
    compound_list = []
    compound_dict= sorted(compound_dict.iteritems(), key=lambda d:d[0])

    for k,v in compound_dict:
        date_list.append(k)
        positive_list.append(v['positive'])
        neutral_list.append(v['neutral'])
        negative_list.append(v['negative'])
        compound_list.append(v['positive'] - v['negative'])

    df2 = pd.DataFrame({
            'date': pd.Series(date_list),
            'compound': pd.Series(compound_list)
        })
    df2.to_csv(out_path,
               index=False,
               columns=['date', 'compound'],
               encoding='utf-8')