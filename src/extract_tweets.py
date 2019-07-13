#coding=utf-8
import pandas
import re
import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')
from optparse import OptionParser

parser = OptionParser("Help for extract tweets",
        description="algorithm implemented in python.",
        version="1.0"
    )
parser.add_option("-i", "--input", action="store", dest="input",type="string", help="Input directory")
parser.add_option("-o", "--output", action="store", dest="output",type="string", help="Output directory")
options, args = parser.parse_args()

stock_codes = ['AAPL', 'CSCO', 'INCT', 'MSFT']
for stock_code in stock_codes:
	in_file = os.path.join(options.input,stock_code+'_tweets.xlsx')
	out_file = os.path.join(options.output,stock_code+'_tweets.csv')
	out_txt = os.path.join(options.output,stock_code+'_tweets.txt')
	file = pandas.read_excel(in_file)
	FORMAT = ['Date', 'Tweet content', 'Following']
	tweet_data = file[FORMAT]

	tweet_list = []
	fout = open(out_txt,'w')
	for tweet in tweet_data['Tweet content']:
		tweet1 = tweet.replace(',', ' ')
		tweet1 = tweet.replace('\n', ' ')
		tweet_list.append(tweet1)
		out_str = '0000\tneutral\t'+tweet1+'\n'
		fout.write(out_str)
	fout.close()
	tweet_data_new = tweet_data.drop('Tweet content', axis=1)
	tweet_data_new = tweet_data_new.assign(tweet_content=pandas.Series(tweet_list).values)
	tweet_data_new.to_csv(out_file, index=False, sep=',', encoding='utf-8')
