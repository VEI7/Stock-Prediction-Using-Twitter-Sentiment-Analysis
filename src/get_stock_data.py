from yahoofinancials import YahooFinancials
import pandas as pd
import os
from optparse import OptionParser

parser = OptionParser("Help for get stock data",
        description="algorithm implemented in python.",
        version="1.0"
    )
parser.add_option("-o", "--output", action="store", dest="output",type="string", help="Output directory")
options, args = parser.parse_args()

class YahooData:
    def __init__(self, startDate, endDate):
        self.startDate = startDate
        self.endDate = endDate

    def getYahooData(self, keyword):
        yahoo = YahooFinancials(keyword)
        historical_data = yahoo.get_historical_price_data(self.startDate, self.endDate, "daily")
        print historical_data
        return historical_data


basedir = os.path.abspath(os.path.dirname(__file__))
stock_path = os.path.join(basedir,options.output)
start_date = "2016-03-01"
end_date = "2016-06-16"
stock_code = ['AAPL', 'CSCO', 'INCT', 'MSFT']


yahoo_data = YahooData(start_date, end_date)
for sc in stock_code:
    historical_data = yahoo_data.getYahooData(sc)
    date_list = []
    open_list = []
    close_list = []
    high_list = []
    low_list = []
    adjclose_list = []
    volume_list = []
    last_close = 100.0
    for idx,data in enumerate(historical_data[sc]['prices']):
        if idx == 0:
            last_close = data['close']
        date_list.append(data['formatted_date'])
        open_list.append(data['open'])
        close_list.append(data['close'])
        high_list.append(data['high'])
        low_list.append(data['low'])
        adjclose_list.append((data['close'] - last_close)/last_close)
        last_close = data['close']
        volume_list.append(data['volume'])
    df2 = pd.DataFrame({
        'date': pd.Series(date_list),
        'open': pd.Series(open_list),
        'close': pd.Series(close_list),
        'high': pd.Series(high_list),
        'low': pd.Series(low_list),
        'pct_change': pd.Series(adjclose_list),
        'volume': pd.Series(volume_list)
    })
    df2.to_csv(os.path.join(stock_path, sc+'.csv'),
               index=False,
               columns=['date', 'open', 'close', 'high', 'low', 'pct_change', 'volume'],
               encoding='utf-8')
