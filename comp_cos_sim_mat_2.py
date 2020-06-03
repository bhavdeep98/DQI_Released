import cupy as np
from tqdm import tqdm

print('Start computing the cosine similarity matrix!')

embeddings = []
with open("counter-fitted-vectors_wordsAndEmbedding_3_context_refined.txt", 'r') as ifile:
    for line in tqdm(ifile):
        embedding = [float(num) for num in line.strip().split()[1:]]
        embeddings.append(embedding)
embeddings = np.array(embeddings)
product = np.dot(embeddings, embeddings.T)
norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
cos_sim = product / np.dot(norm, norm.T)
np.save(('/scratch/srmishr1/cos_sim_counter_fitting_with_context_2.npy'), cos_sim)
