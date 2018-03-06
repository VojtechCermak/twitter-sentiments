#Standard
import time
import numpy as np
import pandas as pd

# file manipulation
import os

# word embedings
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim import corpora, models, similarities

#read lists
from ast import literal_eval

# scikit
#load classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# laod vectorizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# evaluation
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

# Custom Functions
# convert date, hours and minutes multiindex to single datetime index
def index_to_datetime(inputDF, freq):
    dataFrame = inputDF.copy()
    dataFrame = dataFrame.reset_index()
    
    if freq == 'min':
        dataFrame['DateTime'] = dataFrame['date'] + ' ' + dataFrame['hour'].astype(str) + ':' + dataFrame['minute'].astype(str)
        dataFrame = dataFrame.drop(['date', 'hour', '5min', 'minute'], axis=1)
    elif freq == '5min':
        dataFrame['DateTime'] = dataFrame['date'] + ' ' + dataFrame['hour'].astype(str) + ':' + dataFrame['5min'].astype(str)
        dataFrame = dataFrame.drop(['date', 'hour', '5min'], axis=1)
    elif freq == 'hour':
        dataFrame['DateTime'] = dataFrame['date'] + ' ' + dataFrame['hour'].astype(str)
        dataFrame = dataFrame.drop(['date', 'hour'], axis=1)        
    else:
        print('Unsupported frequency')
        return
    
    dataFrame['DateTime'] = pd.to_datetime(dataFrame['DateTime'])    
    dataFrame = dataFrame.set_index('DateTime')
    return dataFrame

# loads market data and converts to suitable format
def load_marketdata(path):
    stockDF = pd.read_csv(path)
    stockDF['DateTime'] = stockDF['Date'] + ' ' + stockDF['Time']
    stockDF['DateTime'] = pd.to_datetime(stockDF['DateTime'])
    stockDF = stockDF.set_index('DateTime')
    return stockDF

# creates time grid with suitable format
def load_grid(start, end, freq='min'):
    grid = pd.date_range(start=start, end=end, freq=freq)
    grid = pd.Series(grid).rename('DateTime')
    grid = pd.DataFrame(grid).set_index('DateTime')
    return grid

def load_tweets(path):
    tweets = pd.read_csv(path)
    # convert column values to lists of words
    tweets['lemmas'] = tweets['lemmas'].apply(literal_eval)
    tweets['tokens'] = tweets['tokens'].apply(literal_eval)
    
    # create time variables
    tweets['created_at'] = pd.to_datetime(tweets['created_at'], format='%Y-%m-%d %H:%M:%S')
    tweets['date'] = tweets['created_at'].astype(str).str[:10]
    tweets['hour'] = tweets['created_at'].astype(str).str[11:13]
    tweets['minute'] = tweets['created_at'].astype(str).str[14:16]
    tweets['5min'] = (tweets['minute'].astype(int)//5)*5

    #Spam filtering - Remove duplicate tweets in date
    tweets = tweets.drop_duplicates(['date', 'text'])

    # Indexing
    tweets.set_index(['date', 'hour', '5min' ,'minute', 'id'], inplace = True)
    return tweets

def aggregate_tweets(inputDF, freq, forms):
    tweets = inputDF.copy()
    special = ['F_exclamation', 'F_question', 'F_ellipsis', 'F_hashtags', 'F_cashtags', 'F_usermention', 'F_urls']
    
    if freq == 'min':
        level = ['date', 'hour', '5min', 'minute']
    elif freq == '5min':
        level = ['date', 'hour', '5min']
    elif freq == 'hour':
        level = ['date', 'hour']
    elif freq == 'none':
        level = ['date', 'hour', '5min', 'minute', 'id']
        freq = 'min'
    else:
        print('Unsupported frequency') 
        return
    
    sum_text = tweets[forms].groupby(level=level).apply(sum)
    sum_special = tweets[special].groupby(level=level).sum().add_prefix('sum')
    avg_special = tweets[special].groupby(level=level).mean().add_prefix('avg')
    count_tweets = tweets.groupby(level=level).size().rename('tweet_count')

    finalDF = pd.concat([sum_special, avg_special, count_tweets, sum_text], axis = 1)
    finalDF = finalDF.rename(columns={forms: "text"}) #rename lemmas/tokens to text
    finalDF = index_to_datetime(finalDF, freq)
    return finalDF

def get_label(tweetDF, shift, biclass = True):
    """
    shift = n  - label is n minutes lagged
    shift = -n  - label is n minute in future
    """
    df = grid.join(prices['Close'])
    df = df.fillna(method = 'ffill')
    
    if shift > 0 :
        df['minLag'] = df['Close'].shift(shift)
        conditions = [df['minLag'] == df['Close'], df['minLag'] < df['Close'], df['minLag'] > df['Close']]
        df['Label'] = np.select(conditions, ['NoChange', 'Growth', 'Decline'], default='Missing')
    else:
        df['minShift'] = df['Close'].shift(shift)
        conditions = [df['minShift'] == df['Close'], df['minShift'] > df['Close'], df['minShift'] < df['Close']]
        df['Label'] = np.select(conditions, ['NoChange', 'Growth', 'Decline'], default='Missing')
        
    finalDF = df.join(tweetDF)
    finalDF = finalDF.dropna()
    
    # delete nochange labels if biclass TRUE
    if biclass:
        finalDF = finalDF[finalDF['Label'] != 'NoChange']
    
    return finalDF
	
	
def get_model_prediction(inputDF, labeling,  method, validations=5):
    if method == 'logit':
        model = LogisticRegression(C=1e30,penalty='l2')
        pred = cross_val_predict(model, inputDF, labeling, cv=validations, n_jobs=1, verbose=0)
        
    elif method == 'L2_logit':
        model = LogisticRegression(C=1, penalty='l2')
        pred = cross_val_predict(model, inputDF, labeling, cv=validations, n_jobs=1, verbose=0)    
        
    elif method == 'L1_logit':
        model = LogisticRegression(C=1, penalty='l1')
        pred = cross_val_predict(model, inputDF, labeling, cv=validations, n_jobs=1, verbose=0)    
        
    elif method == 'nb':
        model = MultinomialNB()
        pred = cross_val_predict(model, inputDF, labeling, cv=validations, n_jobs=1, verbose=0)  
    else:
        raise ValueError('Method is not supported')
        
    return pred


def get_metric(pred, label, method):
    if method == 'kappa':
        value = cohen_kappa_score(label, pred)
    elif method == 'acc':
        value = accuracy_score(label, pred)
    else:
        raise ValueError('Method is not supported')
        
    return value

# List of vectorization methods
def BOW_vectorize(inputText, method):
    # COUNT VECTORIZER
    # binary terms vectorizer
    if method == 'binary':
        vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, 
                              binary=True)
        train = vec.fit_transform(inputText)

    # Simple count vectorizer
    elif method == 'count':
        vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, 
                              binary=False)
        train = vec.fit_transform(inputText)

    # Simple count vectorizer with stopwords filter
    elif method == 'count_sw':
        vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, 
                              stop_words='english', binary=False)
        train = vec.fit_transform(inputText)

    # TFIDF VECTORIZER
    # Term frequencies vectorizer
    elif method =='frequency':
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x,  
                              sublinear_tf = False, use_idf=False)
        train = vec.fit_transform(inputText)

    #simple TFIDF vectorizer
    elif method =='tfidf':
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, 
                              sublinear_tf = False, use_idf=True)
        train = vec.fit_transform(inputText)

    elif method =='tfidf_sw':
        #simple TFIDF vectorizer with english stop words
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, 
                              stop_words='english',sublinear_tf = False, use_idf=True)
        train = vec.fit_transform(inputText)

    elif method =='log_tfidf':
        #LOG tf TFIDF vectorizer
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x,  
                              sublinear_tf = True, use_idf=True)
        train = vec.fit_transform(inputText)

    elif method =='log_tfidf_sw':
        #LOG tf TFIDF vectorizer with english stop words
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, 
                              stop_words='english', sublinear_tf = True, use_idf=True)
        train = vec.fit_transform(inputText)
    else:
        raise ValueError('Method is not supported')
    return train

def BOW_gridsearch(inputDict):
    d = {}
    # Create dataset
    for form in inputDict['forms']:
        d[form] = {}
        for agg in inputDict['aggregates']:
            d[form][agg] = {} 

            #create dataset based on values forms and aggregation methods
            dataset = aggregate_tweets(tweets, agg, form)

            # Add labels
            for direction in inputDict['directions']:
                d[form][agg][direction] = {}
                for window in windows:
                    d[form][agg][direction][window] = {}

                    # get direction of window
                    if direction == 'past':
                        window_dir = window
                    elif direction == 'future':
                        window_dir = -1*window

                    # Add label based on window to dataset
                    labeled_dataset = get_label(dataset, window_dir)
                    labeled_dataset = labeled_dataset.sample(frac=1) # shuffle

                    text = labeled_dataset['text']
                    label = labeled_dataset['Label']

                    # create features using vectorizer
                    for vec in inputDict['vectorizers']:
                        d[form][agg][direction][window][vec] = {}
                        features = BOW_vectorize(text, vec)
                        print(form +' '+ agg +' '+ direction +' '+ str(window) +' '+ vec)

                        # validate dataset using models and metrics
                        for model in inputDict['models']:
                            d[form][agg][direction][window][vec][model] = {}
                            pred = get_model_prediction(features, label, model)
                            for metric in inputDict['metrics']:
                                value = get_metric(pred, label, metric)
                                d[form][agg][direction][window][vec][model][metric] = value
    return d

def reform_BOW_gridsearch(inputDict):
    reform = {(level1_key, level2_key, level3_key, level4_key, level5_key, level6_key, level7_key): values
        for level1_key, level2_dict in inputDict.items()
        for level2_key, level3_dict in level2_dict.items()
        for level3_key, level4_dict in level3_dict.items()
        for level4_key, level5_dict in level4_dict.items()
        for level5_key, level6_dict in level5_dict.items()
        for level6_key, level7_dict in level6_dict.items()
        for level7_key, values      in level7_dict.items()}
    dataFrame = pd.DataFrame(reform, index=[0]).T
    return dataFrame