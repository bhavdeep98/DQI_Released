import pandas as pd                                                             
import numpy as np         
import sys                                                     
                                                                                
df_train_bad = pd.read_csv('squad_good_eff.csv')                                      
#df_train_good = pd.read_csv('testgood.csv')                                    
                                                                                
#print(len(df_train_good))                                                      
#print(len(df_train_bad))                                                       
                                                                                
#df_train_bad_reduced = df_train_bad.sample(n=len(df_train_good))               
df_train_bad_reduced = df_train_bad.sample(n=10)                               
#df_train_bad_reduced = df_train_bad.sample(n=1)                                
print(len(df_train_bad_reduced))                                                
df_train_bad_reduced = df_train_bad_reduced.reset_index()                       
df_train_bad_reduced.to_csv(sys.argv[1]) 
