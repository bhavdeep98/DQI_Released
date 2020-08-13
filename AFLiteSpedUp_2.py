#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pandas as pd
import cudf as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MinMaxScaler
#from skmultilearn.problem_transform import LabelPowerset

import nltk

import json
import csv

import string
import sys
import random

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

import numpy as np
import os

import argparse

from torch.utils.data import TensorDataset
from tqdm import tqdm, trange
import pickle
import itertools

from numba import jit

from multiprocessing import Pool, TimeoutError
import concurrent.futures

import logging
logging.basicConfig(level=logging.ERROR)


EMBEDDING_BOOL = False
FINE_TUNE = False


# In[ ]:


## Load module for test train split
from sklearn.model_selection import train_test_split

## Load set of Linear models
from sklearn.linear_model import LogisticRegression


## Load non-Linear Models
from sklearn import svm
#from thundersvm import SVC

from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel
import torch

# transformer libraries for fine tune embeddings
# from pytorch_transformers import BertPreTrainedModel, RobertaConfig, \
#     ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel
# from pytorch_transformers.modeling_roberta import RobertaClassificationHead
from torch.nn import CrossEntropyLoss



# In[2]:


if os.getcwd().split('/')[-1] != 'snli_1.0':
    os.chdir('/home/srmishr1/AFLite_2/glue_data/SNLI')


# In[3]:


D = pd.read_csv('/home/srmishr1/AFLite_2/glue_data/SNLI/xaa',sep="\t")#, error_bad_lines=False)
S=D

print(len(D))

# In[5]:


sentences_list = []
label_list = []
for index in range(0,len(S['sentence1'])):
    sentences_list.append(str(S['sentence1'][index])+"</s>"+str(S['sentence2'][index]))
    if S['gold_label'][index] == "entailment":
        label_list.append(0)
    elif S['gold_label'][index] == "neutral":
        label_list.append(1)
    else:
        label_list.append(2)


# In[7]:


## For distributed training: local_rank
local_rank=-1
no_cuda = False
seed=42
fp16=False
fp16_opt_level='01'

# In[8]:


# Setup CUDA, GPU & distributed training
if local_rank == -1 or no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = 1

print(device)


# In[9]:


tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')

model.to(device)

"""
     Help you save GPU memory and thus allow bigger models
     and bigger batches under the same hardware limitation.
     https://medium.com/the-artificial-impostor/use-nvidia-apex-for-easy-mixed-precision-training-in-pytorch-46841c6eed8c
"""
if fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

if local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError(
            "Please install apex from \
            https://www.github.com/nvidia/apex to use\
            distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1 and not no_cuda:
    model = torch.nn.DataParallel(model)

model.eval()



# In[10]:


## This is for Fine Tuning
if FINE_TUNE:
    get_ipython().system('export GLUE_DIR="~/AFLite/MNLI/train"')
    get_ipython().system('export TASK_NAME="MNLI"')

    get_ipython().system('python run_glue.py --model_name_or_path bert-base-uncased --task_name $TASK_NAME --do_train --do_eval --data_dir $GLUE_DIR --max_seq_length 128 --per_gpu_eval_batch_size=8 --per_gpu_train_batch_size=8 --learning_rate 1e-5   --num_train_epochs 3.0 --output_dir /scratch/srmishr1/$TASK_NAME/')

"""
 python run_classifier.py --data_dir /scratch/srmishr1/snli_data_adv/  --bert_model roberta-base --task_name snli --output_dir /scratch/srmishr1/results_classifier/SNLI_1e5_5_32_Roberta --cache_dir pytorch_cache  --do_eval --do_lower_case --do_resume
"""



########### Load the fine tuned model ##############
output_model_file = "/scratch/srmishr1/Results/RFLITE/SNLI/Roberta_1e_5_32_final/pytorch_model.bin"
model_state_dict = torch.load(output_model_file)
model = RobertaModel.from_pretrained('roberta-large',state_dict=model_state_dict)

model.to(device)

"""
     Help you save GPU memory and thus allow bigger models
     and bigger batches under the same hardware limitation.
     https://medium.com/the-artificial-impostor/use-nvidia-apex-for-easy-mixed-precision-training-in-pytorch-46841c6eed8c
"""
if fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)            
if local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:                                                                              raise ImportError(                                                                               "Please install apex from \
            https://www.github.com/nvidia/apex to use\
            distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1 and not no_cuda:
    model = torch.nn.DataParallel(model)

model.eval()


# In[12]:



# ### We are using pickle to save the progress in a file ###
if EMBEDDING_BOOL:
    file_name = '/scratch/srmishr1/Results/RFLITE/SNLI/Roberta_1e_5_32_final/listfile_pooled_1.data'
    X_prime = []
    ## Check if there was already some progress then continue from there otherwise just start from begining
    if not os.path.isfile("/scratch/srmishr1/Results/RFLITE/SNLI/Roberta_1e_5_32_final/start_pooled_1.data"):
        start = 0
    else:
        with open("/scratch/srmishr1/Results/RFLITE/SNLI/Roberta_1e_5_32_final/start_pooled_1.data","r") as fileHandle:
            start = fileHandle.readline()
            # Scenario if file is there but nothing is in the file
            if start == '':
                start = 0

    # for index in range(int(start),len(S['sentence'])):
    for index in tqdm(range(int(start),len(sentences_list))):
        if index%50==0:
            #print("Saving the progress for embeddings")
            with open(file_name, 'a+b') as filehandle:
                pickle.dump(X_prime, filehandle, pickle.HIGHEST_PROTOCOL)

            ## Reset the embeddings list X_prime and just keep appending the data in the file
            X_prime = []
            ## overwrite the index so far in seprate file
            with open("/scratch/srmishr1/Results/RFLITE/SNLI/Roberta_1e_5_32_final/start_pooled_1.data","w") as fileHandle:
                fileHandle.write(str(index))
        #print(index)
        try:
    #     input_ids = torch.tensor(tokenizer.encode(S['sentence'][index], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            input_ids = torch.tensor(tokenizer.encode(sentences_list[index], add_special_tokens=True, max_length=512,truncation=True)).unsqueeze(0)
            input_ids = input_ids.to(device)
    #     X_prime.append(TensorDataset(model(input_ids)[1][0]))#, masked_lm_labels=input_ids)
            #print(model(input_ids)[1][0])
            X_prime.append(model(input_ids)[1][0])#, masked_lm_labels=input_ids)
        except:
            with open("/scratch/srmishr1/Results/RFLITE/SNLI/Roberta_1e_5_32_final/exception_index.data","a") as fileHandle:
                fileHandle.write(str(index))
            continue
    ### We have to save one more time after the loop is over otherwise the last 50 value won't be saved just calculated ###
    #print("Saving the progress for embeddings")
    with open(file_name, 'a+b') as filehandle:
        pickle.dump(X_prime, filehandle, pickle.HIGHEST_PROTOCOL)





# In[13]:


X = []


# In[14]:

with open('/scratch/srmishr1/Results/RFLITE/SNLI/Roberta_1e_5_32_final/listfile_pooled_1.data', mode='b+r') as f:
    while 1:
        try:
            X.append(pickle.load(f))
        except EOFError as error:
            # Output expected EOFErrors.
            break
        except Exception as exception:
            # Output unexpected Exceptions.
            print("Unexpected error:", sys.exc_info())
            continue

### The final features look something like follows:
### < [ Sentence with _ replaced by <Option 1> ] embeddings>
### < [ Sentence with _ replaced by <Option 2> ] embeddings>


# In[15]:
print(len(X))

X_prime = []
flat_list = []
for sublist in X:
    for item in sublist:
        flat_list.append(item.detach().cpu().numpy())
X_copy = flat_list[::]

print(len(flat_list))
print(X_copy)
print(len(X_copy))
# In[16]:                                                                                                                                                                                 
X_temp2 = np.asarray(X_copy)
y_temp2 = np.asarray(label_list)


# In[17]:
                                                                                             
scaler2 = MinMaxScaler()
scaler2.fit(X_temp2)



# In[18]:


X_temp_scaled2 = scaler2.transform(X_temp2)


# In[23]:


df =  pd.DataFrame(X_temp_scaled2)
df[len(df.columns)] = [i for i in range(0,len(df))]


# In[24]:                                                                                    

df.rename(columns={len(df.columns)-1:'index'}, inplace=True)

print(df)
df_copy = df.copy()
# In[25]:

# When applying AFLITE
# to WINOGRANDE, we set m = 10, 000, n = 64, k = 500,
# and τ = 0.75.

#@jit(nopython=True)
def getPredictions(df,y):
    print("Iteration : "+str(i))
    ## 90 percent of the data is in the test set

    print("Training Samples: ")

    print("Test train Split")
    X_train, X_test, y_train, y_test = train_test_split(df,y, test_size=0.90)

    ## Train models
    ### Logistic Regression
    lm = LogisticRegression(random_state=0,C=0.01,penalty = 'l2',max_iter=400)
    lm.fit(X_train.loc[:,X_train.columns!=len(X_train.columns)], y_train)
    print("Logistic regression training complete")

    ## SVM
    rbf_svc = svm.SVC(kernel='rbf')
    svc = svm.SVC()
    rbf_svc.fit(X_train.loc[:,X_train.columns!=len(X_train.columns)],y_train)
#     svc = SVC()
#     svc.fit(X_train,y_train)
    print("SVM training complete")

    # Predictions
    predictions = lm.predict(X_test.loc[:,X_test.columns!=len(X_test.columns)])
    print("Done with the predictions for LR")
    predictions_svm = rbf_svc.predict(X_test.loc[:,X_test.columns!=len(X_test.columns)])
#     predictions_svm = svc.predict(X_test)
    print("Done with the predictions for SVM")

    print("Peek predictions")
    print(predictions[:10])
    print(predictions_svm[:10])

    indices = X_test.iloc[:,len(X_test.columns)-1]
    return predictions,predictions_svm, indices


## Target Dataset size ##
n = 50000
K = 10000
print(len(df))
y_original = y_temp2[:]


while len(df) > n:
    #Dictionary to save the mdoel predictions for every individual elements.
    #E = dict.fromkeys(S['index'])
    #EPrime = dict.fromkeys(S['sentence1'])

    D = {}
    m=64
    #m=3
#     for i in range(0,m):
    #pool = Pool(processes=4)

    #X = df.loc[:,df.columns!=len(df.columns)].to_numpy()
    y = list(y_temp2.values())

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results  = [executor.submit(getPredictions,df,y) for _ in tqdm(range(m))]

    for f in concurrent.futures.as_completed(results):
        prediction,prediction_svm,index = f.result()
        #print(f.result())
        #print(index)
    #predictions,predictions_svm,indices  = pool.map(partial(getPredictions,y=y),X)
    #print(predictions)
    #print(predictions_svm)
    #print(indices)
        #for prediction, prediction_svm, index in zip(predictions,predictions_svm,indices):
        for k,i in zip(index.keys(),range(len(index))):
            original_index = index[k]
            if original_index not in D.keys():
                D[original_index] = [prediction[i],prediction_svm[i]]
            else:
                D[original_index].append(prediction[i])
                D[original_index].append(prediction_svm[i])

#     print(D)
    #predictibility score
    print("Starting with Predictibility Scores")
    P = {}
    for key,value in D.items():
        P[key] = value.count(y_original[key])/len(value)

    X_new_index = {}
    print("The length of the Dictioanry is : "+str(len(P)))
    for key,value in P.items():
        ## 0.75 si the tao here
        ## First we select the instances where the predicitibility score was more than tao
        if value>=0.75:
        #if value>=0.50:
            X_new_index[key] = value

    # Sort them based on value
    X_new_index = {k: v for k, v in sorted(X_new_index.items(), key=lambda item: item[1])}


    #if len(X_new_index) < K:
    #    print("EXITING DUE TO K")
    #    break
    #else:
    #    # select top K values
    #    X_new_index = dict(itertools.islice(X_new_index.items(), K))
    X_new_index = dict(itertools.islice(X_new_index.items(), K))

#     for k in X_new_index.keys():
#         P.pop(k,None)
    print(len(X_new_index))
    df_prime = df[~df.index.isin(X_new_index.keys())]
    print(len(df_prime))
    y_temp2={}
    for i in df_prime['index']:
        y_temp2[i]=df_copy.at[i,'label']
    print(y_temp2)
#     y_temp3 = {}
#     for i in y_temp2.keys():
#         if i not in X_new_index.keys():
# #             y_temp3.append(y_temp2[i])
#             y_temp3[i] = y_temp2[i]
#     y_temp2 = y_temp3[:len(df_prime)]

    #print(len(y_temp2))
    #print(len(X_new_index.keys()))
    #print(len(y_temp2))
    #print(df_prime)
    print("New size after pruning")
    print(len(df_prime))
#     x = input()
#     No more pruning possible
    if len(df) == len(df_prime):
        break
    df = df_prime.copy()
    if len(df) < K:
        break
    # update labels as well

print(df)
print(X_new_index)


with open('/scratch/srmishr1/Results/RFLITE/SNLI/Roberta_1e_5_32_final/Ids_10_SVMandLR.pickle', 'wb') as handle:
    pickle.dump(X_new_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

df.to_csv("/scratch/srmishr1/Results/RFLITE/SNLI/Roberta_1e_5_32_final/train_good_10_SVMandLR.tsv")
