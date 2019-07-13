## Twitter mood predicts the stock market
This code is for Stock Prediction Using Twitter Sentiment Analysis. 

## Env Setting
```
pip install requirement.txt
```

**Including**

```
pandas
sklearn
matplotlib
yahoofinancials
pickle
cachetools==2.0.1
scipy==1.0.0
numpy==1.13.1
nltk==3.2.4
tqdm==4.19.4
ekphrasis==0.4.9
Keras==2.2.0
keras_utilities==0.5.0
frozendict==1.2
```


## How To Run The Codes
#### ATTENTION: Must under python 2.7 to run this code!

To start our system, you need:

#### 1.  Get stock data

```shell
python get_stock_data.py -o ../data/stock/
```

#### 2. Extract Tweets data

```shell
python extract_tweets.py -i ../data/original_tweets/ -o ../data/tweets/
```

#### 3. Train Sentiment Analysis Model

##### (1) Download pre-trained Word Embeddings

The models were trained on top of word embeddings pre-trained on a big collection of Twitter messages. We collected a big dataset of 330M English Twitter messages posted from 12/2012 to 07/2016. For training the word embeddings we used [GloVe](https://github.com/stanfordnlp/GloVe). For preprocessing the tweets we used [ekphrasis](https://github.com/cbaziotis/ekphrasis), which is also one of the requirements of this project.

You can download one of the following word embeddings:

- [datastories.twitter.50d.txt](https://mega.nz/#!zsQXmZYI!M_y65hkHdY88iC3I8Yeo7N9IRBI4D9mrpz016fqiXwQ): 50 dimensional embeddings
- [datastories.twitter.100d.txt](https://mega.nz/#!OsYTjIrQ!gLp6YLa0A3ncXjaUffbgL2RtUI74bvSkUKpflAS0OyQ): 100 dimensional embeddings
- [datastories.twitter.200d.txt](https://mega.nz/#!W5BXBISB!Vu19nme_shT3RjVL4Pplu8PuyaRH5M5WaNwTYK4Rxes): 200 dimensional embeddings
- [datastories.twitter.300d.txt](https://mega.nz/#!u4hFAJpK!UeZ5ERYod-SwrekW-qsPSsl-GYwLFQkh06lPTR7K93I): 300 dimensional embeddings

##### (2) Modify config.py

##### (3) Train

```shell
python train_model.py
```

##### (4) Deploy

```shell
python test_model.py
```

#### 4. Get compound tweets data and generate train data

```shell
python get_compound_tweets.py
python gen_train_data.py
```

####  5. Train classifer for stock prediction

```shell
python stock_prediction.py
```

------------------------------------------------------
Contributors: Jiaqi Wei<br/>