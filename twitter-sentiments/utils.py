# Vectorization and evaluation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as txt

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

def aggregate_tweets(inputDF, freq, forms):
    '''
    Agregates text over selected frequency.

    Selectable frequencies are 'hour', '5min' ,'minute' and 'none' for no aggragating (whole tweets are returned)
    Tweets with identical text occuring more than once per day are assumed to be spamm and are filtered.

    '''
    tweets = inputDF.copy()
    special = ['F_exclamation', 'F_question', 'F_ellipsis', 'F_hashtags', 'F_cashtags', 'F_usermention', 'F_urls']

    if freq == 'none':
        level = ['date', 'hour', '5min', 'minute', 'id']
    elif freq == 'min':
        level = ['date', 'hour', '5min', 'minute']
    elif freq == '5min':
        level = ['date', 'hour', '5min']
    elif freq == 'hour':
        level = ['date', 'hour']
    else:
        raise ValueError('Frequency is not supported')

    # Aggregate tweets and special features
    sum_text = tweets[forms].groupby(level=level).apply(sum).rename("text")
    sum_special = tweets[special].groupby(level=level).sum().add_prefix('sum')
    avg_special = tweets[special].groupby(level=level).mean().add_prefix('avg')
    count_tweets = tweets.groupby(level=level).size().rename('tweet_count')
    df = pd.concat([sum_special, avg_special, count_tweets, sum_text], axis = 1)

    # Reconstruct index to single lablel
    df = df.reset_index()
    if freq == 'none':
        df['DateTime'] = df['date'] + ' ' + df['hour'].astype(str) + ':' + df['minute'].astype(str)
        df = df.drop(['date', 'hour', '5min', 'minute', 'id'], axis=1)
    elif freq == 'min':
        df['DateTime'] = df['date'] + ' ' + df['hour'].astype(str) + ':' + df['minute'].astype(str)
        df = df.drop(['date', 'hour', '5min', 'minute'], axis=1)
    elif freq == '5min':
        df['DateTime'] = df['date'] + ' ' + df['hour'].astype(str) + ':' + df['5min'].astype(str)
        df = df.drop(['date', 'hour', '5min'], axis=1)
    elif freq == 'hour':
        df['DateTime'] = df['date'] + ' ' + df['hour'].astype(str)
        df = df.drop(['date', 'hour'], axis=1)
    else: 
        raise ValueError('Frequency is not supported')
        
    df['DateTime'] = pd.to_datetime(df['DateTime'])    
    df = df.set_index('DateTime')

    return df

def get_label(textDF, pricesDF, shift):
    """
    shift = n  - label is n minutes lagged
    shift = -n  - label is n minute in future
    """
    
    df = pd.DataFrame(pricesDF['Close'])
    
    if shift > 0 :
        df['minLag'] = df['Close'].shift(shift)
        conditions = [df['minLag'] == df['Close'], df['minLag'] < df['Close'], df['minLag'] > df['Close']]
        df['Label'] = np.select(conditions, ['NoChange', 'Growth', 'Decline'], default='Missing')
    else:
        df['minShift'] = df['Close'].shift(shift)
        conditions = [df['minShift'] == df['Close'], df['minShift'] > df['Close'], df['minShift'] < df['Close']]
        df['Label'] = np.select(conditions, ['NoChange', 'Growth', 'Decline'], default='Missing')
    
    # delete missing label, and also nochange labels if biclass TRUE
    df.loc[df['Label'] == 'Missing', 'Label'] = np.nan
    df.loc[df['Label'] == 'NoChange', 'Label'] = np.nan
        
    text_index = pd.DataFrame(index = textDF.index)
    labelDF = text_index.join(df)
    labelDF = labelDF.reset_index()
    
    return labelDF

def get_model_prediction(inputDF, labeling,  method, validations=5):
    if method == 'logit':
        model = LogisticRegression(C=1e30,penalty='l2')
    elif method == 'L2_logit':
        model = LogisticRegression(C=1, penalty='l2')
    elif method == 'L1_logit':
        model = LogisticRegression(C=1, penalty='l1')
    elif method == 'nb':
        model = MultinomialNB()
    else:
        raise ValueError('Method is not supported')
    pred = cross_val_predict(model, inputDF, labeling, cv=validations, n_jobs=1, verbose=0)    
    return pred     

# Vectorization methods
def tweet2vec_mean(tokens, embedding):
    tweetVec = []
    for word in tokens:
        try:
            wordVec = embedding[word]
            tweetVec.append(wordVec)
        except: continue   
            
    if len(tweetVec) < 1:
        tweetVec = np.zeros(embedding.vector_size)
        return tweetVec
    
    return np.mean(tweetVec, axis=0)

def tweet2vec_minmax(tokens, embedding):
    tweetVec = []
    for word in tokens:
        try:        
            wordVec = embedding[word]
            tweetVec.append(wordVec)
        except: continue
            
    if len(tweetVec) < 1:
        tweetVec= np.zeros((embedding.vector_size)*2)
        return tweetVec
        
    minVec = np.min(tweetVec, axis=0)
    maxVec = np.max(tweetVec, axis=0)
    return np.append(maxVec, minVec)



def tweet2vec_mean_sw(tokens, embedding):
    tweetVec = []
    for word in tokens:
        try:
            if word not in txt.ENGLISH_STOP_WORDS:
                wordVec = embedding[word]
                tweetVec.append(wordVec)
        except: continue
            
    if len(tweetVec) < 1:
        tweetVec = np.zeros(embedding.vector_size)
        return tweetVec
    
    return np.mean(tweetVec, axis=0)

def tweet2vec_tfidf(tokens, embedding, tfidf):
    tweetVec = []
    weights = []
    
    vocabulary = tfidf.vocabulary_
    idf = tfidf.idf_
    
    for word in tokens:
        try:        
            wordVec = np.array(embedding[word])
            weight = idf[vocabulary[word]]
            
            tweetVec.append(wordVec)
            weights.append(weight)
        except: continue
            
    if len(tweetVec) < 1:
        tweetVec= np.zeros(embedding.vector_size)
        return tweetVec
        
    weights = weights / np.sum(weights)
    tweetVec = np.array(tweetVec)
    weighted_vec = tweetVec * weights[:,None]
    return weighted_vec.sum(axis = 0)
    
    


def BOW_vectorize(inputText, method):
    '''
    Calls scikit text vectorizers based on parameters. Returns sparse matrix. 

    '''
    
    if method == 'binary':          # binary terms vectorizer
        vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, binary=True)
    elif method == 'count':         # Simple count vectorizer
        vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, binary=False)
    elif method == 'count_sw':      # Simple count vectorizer with stopwords filter
        vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, stop_words='english', binary=False)
    elif method =='frequency':      # Term frequencies vectorizer
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, sublinear_tf = False, use_idf=False)
    elif method =='tfidf':          #simple TFIDF vectorizer
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, sublinear_tf = False, use_idf=True)
    elif method =='tfidf_sw':       #simple TFIDF vectorizer with english stop words
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, stop_words='english',sublinear_tf = False, use_idf=True)
    elif method =='log_tfidf':      #LOG tf TFIDF vectorizer
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, sublinear_tf = True, use_idf=True)
    elif method =='log_tfidf_sw':   #LOG tf TFIDF vectorizer with english stop words
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, stop_words='english', sublinear_tf = True, use_idf=True)
    else:
        raise ValueError('Method is not supported')
    train = vec.fit_transform(inputText)
    return train


def VW_vectorize(inputText, embedding, method):
    # Embedding simple average
    if method == 'mean':
        df = inputText.apply(tweet2vec_mean, args=[embedding])

    # Embedding simple average without stopwords
    elif method == 'mean_sw':
        df = inputText.apply(tweet2vec_mean_sw, args=[embedding])

    # Embedding minimum + maxiumum values concacenated
    elif method == 'minmax':
        df = inputText.apply(tweet2vec_minmax, args=[embedding])
    
    # Embedding IDF weighted average
    elif method == 'idf':
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
        train = vec.fit_transform(inputText)
        df = inputText.apply(tweet2vec_tfidf, args=[embedding, vec])
    
    # Embedding IDF weighted average without stopwords
    elif method == 'idf_sw':
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, stop_words='english')
        train = vec.fit_transform(inputText)
        df = inputText.apply(tweet2vec_tfidf, args=[embedding, vec])
    else:
        raise ValueError('Method is not supported')
        
    train = df.apply(pd.Series).fillna(0)
    return train