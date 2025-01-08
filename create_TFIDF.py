import math
import json
import math
import os
import pickle, bs4
import utlities as ut
import pandas as pd
from nltk.corpus import stopwords


from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import regex as re
stops = set(stopwords.words('english'))
import collections
import numpy as np

docIDs = {}
vocabIDCounter = 0
docIDsCounter = 0
vocab = {}
postings = {}
htmltexts = {}

test_config = {"stopwords": True, "stemming": True}


def simpleTokenizor(text):
    soup = BeautifulSoup(text, 'html.parser')
    paragraphs = soup.find_all('p')
    #print (headings)
    cleanedTokens = []
    for p in paragraphs:
        cleanedParagraphs = p.get_text()
        #print (cleanedParagraphs)
        reS1 = re.sub("<.+?>|\n", "", str(cleanedParagraphs))
        tokens = word_tokenize(reS1)
        #print(tokens)
        for token in tokens:
            if token not in stops and token != "," and token != " ":
                cleanedTokens.append(token.lower())
        freq = collections.Counter(cleanedTokens)
    paragraphs = soup.find_all('div')
    for p in paragraphs:
        cleanedParagraphs = p.get_text()
        #print (cleanedParagraphs)
        reS1 = re.sub("<.+?>|\n", "", str(cleanedParagraphs))
        tokens = word_tokenize(reS1)
        #print(tokens)
        for token in tokens:
            if token not in stops and token != "," and token != " ":
                cleanedTokens.append(token.lower())
    return cleanedTokens




# loads the text files into a dictionary.

directory='videogames'
print(os.getcwd())
filenames = os.listdir(directory)
n=2
df_metadata_labels = pd.read_csv("videogame-labels.csv")
for fname in filenames:
    fname_dir = os.path.join(directory, fname)
    f = open(fname_dir, "r", encoding='utf-8')
    text = f.read()
    result = df_metadata_labels[df_metadata_labels['url'].str[26:] == fname]
    if not result.empty:
        row_as_string = result.iloc[0].str.cat(sep=', ')[26:] #gets rid of "videogame/ps2.gamespy.com/" path before title
        for i in range(n):
            text+= " "
            text+=row_as_string
    f.close()
    
    docIDs[docIDsCounter] = fname
    htmltexts[docIDsCounter] = text
    tokens = set(simpleTokenizor(text))
    if test_config["stopwords"]:
        tokens = ut.remove_stopwords(tokens)
    
    if test_config["stemming"]:
        tokens = ut.stemmer(tokens)

    for t in tokens:
        if t in vocab.keys(): 
            vocabID = vocab[t]
        else: 
            vocab[t] = int(vocabIDCounter)
            vocabID = vocabIDCounter
            vocabIDCounter += 1
        
        if vocabID in postings.keys():
            postings[vocabID].append(docIDsCounter)
        else:
            postings[vocabID] = [docIDsCounter]
            
        
    docIDsCounter += 1






count_matrix = [0]*len(docIDs.keys())
tf_idf = [0]*len(docIDs.keys())


for key in docIDs.keys():
    document_vector = [0]*len(vocab.keys())
    fname = docIDs[key]
    fname_directory = os.path.join(directory, fname)
    f = open(fname_directory, "r", encoding='utf-8')
    
    
    text = f.read()
    f.close()
    
    tokens = simpleTokenizor(text)
    
    if test_config["stopwords"]:
        tokens = ut.remove_stopwords(tokens)
    
    if test_config["stemming"]:
        tokens = ut.stemmer(tokens)

    for token in tokens:
        vocab_index = vocab[token]
        document_vector[vocab_index] +=1
    #count_matrix.append(document_vector)
    count_matrix[int(key)] = document_vector

count = 0 
N = len(docIDs)
tfidf_matrix = np.array(count_matrix).T.astype("float64") #.t transposes - so it flips the rows and columns of our matrix
for doc_tfs in tfidf_matrix:
    weighted_tf_array = []
    document_freq = len(postings[count])
    for doc_term_freq in doc_tfs:
        weighted_tf = round(math.log((1+doc_term_freq))*math.log(N/document_freq), 2)
        weighted_tf_array.append(weighted_tf)
    tfidf_matrix[count] = np.array(weighted_tf_array)

    count+=1
print(tfidf_matrix)
# 'wb' means write binary mode # Unpickling the object from the file with open("data.pkl", "rb") as file: # 'rb' means read binary mode loaded_data = pickle.load(file) print("Object has been unpickled.") print(loaded_data)

with open("tfidf_matrix.pkl", "wb") as file:
    pickle.dump(tfidf_matrix, file) 
    print("tfidf matrix has been pickled and saved.")

with open("test_config.pkl", "wb") as file:
    pickle.dump(test_config, file) 
    print("Test Config has been pickled and saved.")

with open('vocab.pkl', 'wb') as f: 
    pickle.dump(vocab, f)
with open('postings.pkl', 'wb') as f: 
    pickle.dump(postings, f)
with open('docids.pkl', 'wb') as f: 
    pickle.dump(docIDs, f)    