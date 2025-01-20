# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np

tf = pickle.load(open('tf.sav' , 'rb'))

def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return len(w1 & w2)

def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return (len(w1) + len(w2))

def test_token_func(q1 , q2) :
    
   
    
    SAFE_DIVISION = 0.0001
    token_features = [0.0]*8
    
    STOP_WORDS = pickle.load(open('stopwords.pkl','rb'))
    
    token1 = q1.split()
    token2 = q2.split()
    
    
    ## In case there are no tokens (for safety)
    if len(token1) == 0 or len(token2) == 0 :
        return token_features 
    
    
    ## Now for common words (which are non stopwords) for both q1 and q2 
    word1 = [ word for word in token1 if word not in STOP_WORDS]
    word2 = [ word for word in token2 if word not in STOP_WORDS]
    ## Now we have to see which are stopwords in both q1 and q2 
    
    stopword1 = [ word for word in token1 if word not in STOP_WORDS]
    stopword2 = [ word for word in token2 if word not in STOP_WORDS]
    
    ## Now we to find common words(non-stopwords) and stopwords
    
    common_words = set(word1).intersection(set(word2))
    common_stopwords = set(stopword1).intersection(set(stopword2))
    
    ## common tokens 
    
    common_tokens = set(token1).intersection(set(token2))
    
    token_features[0] = len(common_words)/ (min(len(word1) , len(word2)) +   SAFE_DIVISION )
    token_features[1] = len(common_words)/ (max(len(word1) , len(word2)) +   SAFE_DIVISION )
    token_features[2] = len(common_stopwords)/ (min(len(stopword1) , len(stopword2)) +   SAFE_DIVISION )
    token_features[3] = len(common_stopwords)/ (max(len(stopword1) , len(stopword2)) +   SAFE_DIVISION )
    token_features[4] = len(common_tokens)/ (min(len(token1) , len(token2)) +   SAFE_DIVISION )
    token_features[5] = len(common_tokens)/ (max(len(token1) , len(token2)) +   SAFE_DIVISION )
    
    
    # Last word of both question is same or not
    token_features[6] = int(token1[-1] == token2[-1])
    
    # First word of both question is same or not
    token_features[7] = int(token1[0] == token2[0])
    
    return token_features


def test_length(q1 , q2) :
  
    
    SAFE_DIVISION = 0.0001
    length_features = [0.0]*3
    token1 = q1.split()
    token2 = q2.split()
    if len(token1) == 0 or len(token2) == 0 :
        return length_features 
    
    length_features[0] =abs(len(token1) - len(token2))
    length_features[1] =(len(token1) + len(token2)) / 2
    
    strs = list(distance.lcsubstrings(q1,q2))
    length_features[2] = len(strs[0]) / (min(len(q1) , len(q2)) + 1 )
    
    return length_features
    
def test_fuzzy(q1 , q2) :
    
    
    
    fuzzy_features=[0.0]*4
    
    fuzzy_features[0] = fuzz.ratio(q1,q2)
    
    fuzzy_features[1] = fuzz.partial_ratio(q1,q2)
    
    fuzzy_features[2] = fuzz.token_sort_ratio(q1,q2)
    
    fuzzy_features[3] = fuzz.token_set_ratio(q1,q2)
    
    return fuzzy_features
    
def preprocess(q) :
    
    
    q= str(q).lower().strip()
    

    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    
    ## The pattern [math] was there several time in Dataset
    q= q.replace('[math]' ,'')
    
    
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q= re.sub(r'([0-9]+)000000000' , r'\1b' , q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    ## Contracting words 
    #https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/19794953#19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }
    ques_contracted = []
    
    for word in q.split():
        if word in contractions:
            word = contractions[word]

        ques_contracted.append(word)
        
    q= ' '.join(ques_contracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags 
    q= BeautifulSoup(q)
    q= q.get_text()
    
    ## Removing punctuations 
    pattern = re.compile('\W')
    q= re.sub(pattern, ' ' , q).strip()
        
    return q

def query_function(q1, q2):
    input_query = []
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    # String lengths
    input_query.append(len(q1))
    input_query.append(len(q2))
    
    # Number of words
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))  # Fixed duplicate

    # Common words and total words
    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))

    # Word share
    total_words = test_total_words(q1, q2)
    input_query.append(round(test_common_words(q1, q2) / total_words, 2) if total_words > 0 else 0)

    # Token features
    token_features = test_token_func(q1, q2)
    input_query.extend(token_features)

    # Length features
    length_features = test_length(q1, q2)
    input_query.extend(length_features)

    # Fuzzy features
    fuzzy_features = test_fuzzy(q1, q2)
    input_query.extend(fuzzy_features)

    # TF-IDF features
    bowq1 = tf.transform([q1]).toarray()
    bowq2 = tf.transform([q2]).toarray()

    # Ensure dimensions match
    input_query = np.array(input_query).reshape(1, -1)  # Dynamic reshape
    final_input = np.hstack((input_query, bowq1, bowq2))  # Horizontal stacking

    return final_input


