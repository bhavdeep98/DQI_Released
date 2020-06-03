import torch
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm, tnrange

#tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
tokenizer = RobertaTokenizer.from_pretrained('/scratch/user1/tfool_1/')

## Read the file and encode the words
with open('counter-fitted-vectors_wordsOnly.txt','r') as f:
    word_ids = []
    words = []
    for word in f.readlines():
        words.append(word)
        word_enc = tokenizer.encode(word,add_special_tokens=True)
        word_ids.append(word_enc)             
###

#print(word_ids)

word_id_tensors = []
for word_id in word_ids:
    word_id_tensors.append(torch.LongTensor(word_id))

#print(word_id_tensors)

#model = RobertaModel.from_pretrained('roberta-large', output_hidden_states=True)
model = RobertaModel.from_pretrained('/scratch/user1/tfool_1/', output_hidden_states=True)

# Set the device to GPU (cuda) if available, otherwise stick with CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)

for i in range(0,len(word_id_tensors)):
    word_id_tensors[i] = word_id_tensors[i].to(device)
    #word_id_tensors[i] = word_id_tensors[i].unsqeeze(0)

model.eval()

for i in range(0,len(word_id_tensors)):
    word_id_tensors[i] = word_id_tensors[i].unsqueeze(0)

with open('counter-fitted-vectors_wordsAndEmbedding_2.txt',"w") as f:
    for word_id_tensor,word in tqdm(zip(word_id_tensors,words)):
        with torch.no_grad():
            out = model(input_ids=word_id_tensor)
            hidden_states = out[2]
            sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
            string = word.strip()+" "#+str(sentence_embedding.data)+"\n"
            for element in sentence_embedding.data.tolist():
                string += str(element) + " "
            string+='\n'
            #print(string)
            f.write(string)
            #print(len(sentence_embedding))
    
"""
# the output is a tuple
print(type(out))
# the tuple contains three elements as explained above)
print(len(out))
# we only want the hidden_states
hidden_states = out[2]
print(len(hidden_states))

print(model)

sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
print(sentence_embedding)
print(sentence_embedding.size())

# get last four layers
last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
# cast layers to a tuple and concatenate over the last dimension
cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
print(cat_hidden_states.size())

# take the mean of the concatenated vector over the token dimension
cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
print(cat_sentence_embedding)
print(cat_sentence_embedding.size())


# save our created sentence representation
torch.save(cat_sentence_embedding.cpu(), 'my_sent_embed.pt')

# load it again
loaded_tensor = torch.load('my_sent_embed.pt')
print(loaded_tensor)
print(loaded_tensor.size())

# convert it to numpy to use in e.g. sklearn
np_loaded_tensor = loaded_tensor.numpy()
print(np_loaded_tensor)
print(type(np_loaded_tensor))"""
