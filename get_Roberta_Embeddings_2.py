import torch
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm, tnrange
import pandas as pd

#tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
tokenizer = RobertaTokenizer.from_pretrained('/scratch/srmishr1/tfool_1/')

## Read the file and encode the words
df = pd.read_csv('words_sentences.csv')
words = df['word']
examples = df['examples']
indeces = []
for word,example in zip(words,examples):
    try:
        #marked_text = "[CLS] " + example + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = tokenizer.tokenize(example,add_special_tokens=True)
        indeces.append(tokenized_text.index(word.lower()))
    except:
        #print(word)
        #print(example)
        i = len(indeces)
        examples[i] = word
        indeces.append(0)
print(len(indeces))
print(len(words))
print(len(examples))
sent_ids = []
for word,example in tqdm(zip(words,examples)):
    try:
        sent_enc = tokenizer.encode(example,add_special_tokens=True)
        sent_ids.append(sent_enc)
    except:
        try:
            print(example)
            print(word)
            sent_enc = tokenizer.encode(word,add_special_tokens=True)
            sent_ids.append(sent_enc)
        except:
            continue

sent_id_tensors = []
for sent_id in sent_ids:
    sent_id_tensors.append(torch.LongTensor(sent_id))

model = RobertaModel.from_pretrained('/scratch/srmishr1/tfool_1/', output_hidden_states=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)

for i in range(0,len(sent_id_tensors)):
    sent_id_tensors[i] = sent_id_tensors[i].to(device)
    #sent_id_tensors[i] = sent_id_tensors[i].unsqeeze(0)

model.eval()

for i in range(0,len(sent_id_tensors)):
    sent_id_tensors[i] = sent_id_tensors[i].unsqueeze(0)

with open('counter-fitted-vectors_wordsAndEmbedding_3_context.txt',"w") as f:
    for sent_id_tensor,word,index,example in tqdm(zip(sent_id_tensors,words,indeces,examples)):
        with torch.no_grad():
            out = model(input_ids=sent_id_tensor)
            #print(len(out))
            hidden_states = out[2]
            #print("Number of hidden Layers : "+ str(len(hidden_states)))
            #print("Number of batches : "+ str(len(hidden_states[-1])))
            #print("Number of tokens : "+ str(len(hidden_states[-1][0])))
            #print("WORD : " + str(word))
            #print("Example : "+ str(example))
            #print("Index : "+str(index))
            #print("recalculated index : "+str(example.lower().split().index(word.lower())))
            #print("word embed length : "+ str(len(hidden_states[-1][0][index])))
            #print("word embedding : "+ str(hidden_states[-1][0][index]))
            
            #sentence_embedding = torch.mean(hidden_states[-1][0][index], dim=1).squeeze()
            #print(len(sentence_embedding))
            #print(sentence_embedding)
            #string = word.strip()+" "+str(sentence_embedding.data)+"\n"
            string = str(word).strip()+" "#+str(str(hidden_states[-1][0][index]))+"\n"
            try:
                #for element in sentence_embedding.data.tolist():
                for element in hidden_states[-1][0][index].tolist():
                    string += str(element) + " "
            except:
                print("WORD : " + str(word))
                print("Example : "+ str(example))
            string+='\n'
            #print(string)
            f.write(string)
            #print(len(sentence_embedding))
