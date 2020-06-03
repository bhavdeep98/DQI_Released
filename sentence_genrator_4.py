from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm
from numba import jit
import wikipedia

@jit
def get_word(word,i,df):
    page = requests.get('https://www.yourdictionary.com/'+word)
    #page = requests.get('https://sentence.yourdictionary.com/'+word)

    soup = BeautifulSoup(page.text, 'html.parser')

    sentence_list = soup.find(class_='greybullets')
    sentence_list_items = sentence_list.find_all('span')

    for sentence in sentence_list_items:
        example = ""
        for part_sent in sentence.contents:
            #print(part_sent)
            if word.lower() in str(part_sent).lower():
                example = example + word
            else:
                example = example + str(part_sent)
        #print(sentence.contents)
        df.loc[i] = [word,example]
        i = i + 1
    return i


words = []


with open('counter-fitted-vectors_wordsOnly.txt') as wordsFile:
    for word in wordsFile.readlines():
        words.append(word.strip('\n'))

df = pd.DataFrame(columns=['word','examples'])
i = 0
#for index in tqdm(range(10)):
for index in tqdm(range(len(words[15000:]))):
    try:
        i = get_word(words[index],i,df)
    except:
        try:
            word_page = wikipedia.page(words[index])
            #add_context(word_page.content,df,words[i])
            context = ""
            for sentence in sent_tokenize(word_page.content):
                #print(sentence.lower())
                if words[index].lower() in sentence.lower():
                    print(sentence.lower())
                    context = sentence
                    break
            df.loc[i] = [words[index],context]
        except wikipedia.DisambiguationError as e:
            try:
                for topic in e.options:
                    word_page = wikipedia.page(topic)
                    #add_context(word_page.content,df,words[i])
                    context = ""
                    for sentence in sent_tokenize(word_page.content):
                        print(sentence.lower())
                        if words[index].lower() in sentence.lower():
                            context = sentence
                            break
                        df.loc[i] = [words[index],context]
            except:
                continue
        except:
            continue
print(df)
df.to_csv('words_sentences_2.csv') 
