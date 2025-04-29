# -*- coding: utf-8 -*-
import pdb
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd
nltk.download('stopwords')
nltk.download('twitter_samples')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
train_pos = all_positive_tweets[:4000]
train_neg = all_negative_tweets[:4000]
test_pos = all_positive_tweets[4000:]
test_neg = all_negative_tweets[4000:]
train_x = train_pos + train_neg
test_x = test_pos + test_neg
train_y = np.append(np.ones((len(train_pos),1)),np.zeros((len(train_neg),1)))
test_y = np.append(np.ones((len(test_pos),1)), np.zeros((len(test_neg),1)))  
def process_tweet(tweet):
    nltk.download('stopwords')
    tweet2 = re.sub(r'^RT[\s]+','',tweet)
    tweet2 = re.sub(r'http?:\/\/.*[\r\n]*','',tweet2)
    tweet2 = re.sub(r'#','',tweet2)
    tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True, reduce_len = True)
    tweet_tokens = tokenizer.tokenize(tweet2)
    stopwords_english = stopwords.words('english')
    tweets_clean = []
    # removing stopwords and punctuations
    for word in tweet_tokens :
        if (word not in stopwords_english and word not in string.punctuation):
            tweets_clean.append(word)
            # Stemming
    stemmer = PorterStemmer()
    tweets_stem = []
    for word in tweets_clean:
        stem_word = stemmer.stem(word)
        tweets_stem.append(stem_word)
    return tweets_stem     

def count_tweets(result, tweets, ys):
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            pair = (word ,y)
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result
freqs = count_tweets({}, train_x, train_y)    
def lookup(freqs, word, label):
    return freqs.get((word,label),0)
def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1]>0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]
    D = len(train_y)
    D_pos = np.sum((train_y == 1).astype(int))
    D_neg = np.sum((train_y == 0).astype(int))
    logprior = np.log(D_pos/D_neg)
    for word in vocab:
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)
        p_w_pos = (freq_pos +1)/(N_pos + V)
        p_w_neg = (freq_neg +1)/(N_neg + V)
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    return logprior, loglikelihood
def naive_bayes_predict(tweet, logprior, loglikelihood):
    word_1 = process_tweet(tweet)
    p = 0;
    p+= logprior
    for word in word_1:
        if word in loglikelihood:
            p += loglikelihood[word]
    if p>0:
        print("POSITIVE")
    elif p<0:
        print("NEGATIVE")
    else:
        print("NEUTRAL")
    return p
def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0
    y_hats = []
    for tweet in test_x:
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)
    error = (1/len(test_x))*np.sum(np.abs(y_hats - test_y))
    accuracy = 1 - error
    return accuracy
           
    
    
            
            