from semantic_text_similarity.models import WebBertSimilarity
from semantic_text_similarity.models import ClinicalBertSimilarity
import pandas as pd 
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize, Normalizer, MinMaxScaler           
import argparse

def main():
    parser = argparse.ArgumentParser()                                          
    
    ## Required parameters                                                      
    parser.add_argument("--dataset_path",type=str,required=True,help="Which dataset to attack.")                       
    parser.add_argument("--output_path", type=str,required=True,     help="Where to save") 
    args = parser.parse_args()                                                                                                                                   
                                                                                                                                                                                                   
    # random 100 samples from testgood
    train_df = pd.read_csv('~/MNLI/Anjana/dev_good_words_final.csv')   
    train_df_part = pd.read_csv(args.dataset_path)   

    ## append Sentences 

    sentence_list = []                                                              
    #sentence_list.extend(train_df['sentence1'])                                     
    #sentence_list.extend(train_df['sentence2'])  
    for sent1, sent2 in zip(train_df['sentence1'],train_df['sentence2']):
        sentence_list.append(str(sent1)+str(sent2))
    #print(train_df[:5])
    #print(sentence_list[:5])
    
    sentence_list_1 = []                                                              
    #sentence_list_1.extend(train_df_part['sentence1'])                                     
    #sentence_list_1.extend(train_df_part['sentence2'])  
    for sent1, sent2 in zip(train_df_part['sentence1'],train_df_part['sentence2']):
        sentence_list_1.append(str(sent1)+str(sent2))

    web_model = WebBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction

    def get_similarity(stringMatrix1,stringMatrix2):
        ret = np.ndarray((len(stringMatrix1), len(stringMatrix2))) 
        for index1 in tqdm(range(len(stringMatrix1))):
            #for index2 in tqdm(range(len(stringMatrix2))):
            for index2 in range(len(stringMatrix2)):
                ret[index1][index2] = web_model.predict([(stringMatrix1[index1],stringMatrix2[index2])])
        return ret

    #print(web_model.predict([("She won an olympic gold medal","The women is an olympic champion")]))
    #print(sentence_list[:12])
    ret = get_similarity(sentence_list_1,sentence_list)
    scaler = MinMaxScaler()                                                         
    scaler.fit(ret)                                                
    ret_norm = scaler.transform(ret)
    np.set_printoptions(suppress=True)
    #print(ret_norm) 
    np.save(("mnligood.npy"), ret_norm)
                                                                                
if __name__ == "__main__":                                                      
    main()   
