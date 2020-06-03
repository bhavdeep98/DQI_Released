import pandas as pd
import numpy as np 
import json
import csv


import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

from tqdm import tqdm as tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 
import nltk
nltk.download('averaged_perceptron_tagger')

import spacy
import math
import argparse

import string
import sys
import random

from collections import Counter
from itertools import chain

def main():                                                                     
    parser = argparse.ArgumentParser()                                          
                                                                                
    ## Required parameters                                                      
    parser.add_argument("--dataset_path",type=str,required=True,help="Which dataset to attack.")
    parser.add_argument("--output_path", type=str,required=True,     help="Where to save")
    args = parser.parse_args()   

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    gwords=pd.read_csv(args.dataset_path)

    nlp = spacy.load("en_trf_bertbaseuncased_lg")

    x=int(len(gwords)/2)
    arr1=[]
    for i in tqdm(range(x)):
        sxlist=(gwords.iloc[i]['fullnopunc'].lower()).split()
        ssxlist=(gwords.iloc[i+x]['fullnopunc'].lower()).split()
        for j in tqdm(range(len(sxlist))):
            arr=[]
            for k in range(len(ssxlist)):
                arr.append(nlp(sxlist[j]).similarity(nlp(ssxlist[k])))
            arr1.append(pd.Series(arr).max())

    wordmax=pd.DataFrame(arr1)
    wordmax.to_csv(args.output_path)
if __name__ == "__main__":                                                      
    main()  

