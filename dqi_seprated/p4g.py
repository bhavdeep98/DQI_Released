import pandas as pd
import numpy as np 
import json
import csv

import matplotlib.pyplot as plt
# import seaborn as sns

from tqdm import tqdm


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 
import nltk
#nltk.download('averaged_perceptron_tagger')

import spacy
import math

import string
import sys
import random
import pickle


from collections import Counter
from itertools import chain

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from sklearn.metrics.pairwise import cosine_similarity

from numba import jit, cuda , vectorize, njit

#from numpy import genfromtxt
#gwords = genfromtxt('/home/user1/MNLI/Anjana/dev_good_words.csv', delimiter=',')

gwords=pd.read_csv("/home/user1/MNLI/Anjana/dev_good_words.csv")
#print(gwords)
#print(gwords.columns)
#@jit
#@vectorize(target='cuda')

nlp = spacy.load("en_trf_bertbaseuncased_lg")

#@vectorize(target='cuda')
#@jit(nopython=True, parallel=True)
#@njit
@jit
def p1(sm,token_l,token_m):
    #nlp = spacy.load("en_trf_bertbaseuncased_lg")
    sm=sm+token_l.similarity(token_m)
    return sm
#@vectorize(target='cuda')
#@jit(nopython=True, parallel=True)
#@njit
#@jit
@vectorize(target='cuda')
def p2(sm,length,arr,WSIML,sl):
    sm=sm/length
    arr.append(sm)
    sm=abs(sm-WSIML)
    sl=sl+sm
    return sm,arr,sl

def param4(s="",size=0,arr=[],WSIML=0.5):
    word = s.split()
    length=len(word)
    sl=0
    for i in (range(length)):
        sm=0
        token_l=nlp(word[i])
        for j in (range(length)):
            if(i!=j):
                token_m=nlp(word[j])
                sm = p1(sm,token_l,token_m)
        sm,arr,sl = p2(sm,length,arr,WSIML,sl)
        #sm=sm/length
        #arr.append(sm)
        #sm=abs(sm-WSIML)
        #sl=sl+sm
    return arr, size/sl

#print(sentence)
#@jit(target='cuda')#(forceobj=True)#(nopython=False)#(target ="cuda")  
#@vectorize(target='cuda')
#@jit
def parameter4(sentence,size,WSIML):
    lists=[]
    vals=[]
    for x in tqdm(range(len(sentence))):
        arr,value = param4(s = sentence[x],size = size,arr = [],WSIML=WSIML)
        lists.append(arr)
        vals.append(value)

    df=pd.DataFrame(lists)
    df.to_csv("/home/user1/MNLI/Anjana/wordsimgabs.csv")
    df=pd.DataFrame(vals)
    df.to_csv("/home/user1/MNLI/Anjana/dqic4gabs.csv")  

size=len(gwords)
WSIML=0.5
#sentence=gwords[-2]
sentence=gwords['fullnopunc']
#sentence = sentence.to_numpy()
sentence_list = []
for sent in sentence:
    sentence_list.append(sent)
#print(sentence)
parameter4(sentence_list,size,WSIML)
