import numpy as np                                                              
#import cupy as np                                                              
import sys                                                                      
                                                                               

from numba import jit, cuda                                                     
from tqdm import tqdm                                                           
 
from sklearn.metrics.pairwise import cosine_similarity
                                                                                
embedding_path = sys.argv[1] # '/data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt'
                                                                                
embeddings = []                                                                 
with open(embedding_path, 'r') as ifile:                                        
    for line in ifile:                                                          
        embedding = [float(num) for num in line.strip().split()[1:]]            
        embeddings.append(embedding)                                            
embeddings = np.array(embeddings)                                               
print(embeddings.T.shape)                                                       
                                                                                
@jit                                                                            
def cosine_similarity_n_space(m1, m2, batch_size=100):                          
    assert m1.shape[1] == m2.shape[1]                                           
    ret = np.ndarray((m1.shape[0], m2.shape[0]))                                
    for row_i in tqdm(range(0, int(m1.shape[0] / batch_size) + 1)):             
        start = row_i * batch_size                                              
        end = min([(row_i + 1) * batch_size, m1.shape[0]])                      
        if end <= start:                                                        
            break                                                               
        rows = m1[start: end]                                                   
        sim = cosine_similarity(rows, m2) # rows is O(1) size                   
        ret[start: end] = sim                                                   
    return ret                                                                  
                                                   
product = cosine_similarity_n_space(embeddings,embeddings,batch_size=4)
#norm = np.linalg.norm(embeddings, axis=1, keepdims=True)                       
#embeddings = np.asarray(embeddings / norm, "float32")                           
#product = np.dot(embeddings, embeddings.T)                                      
np.save(('/scratch/user1/cos_sim_counter_fitting_with_context_2.npy'), product)  
