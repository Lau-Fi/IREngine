#Here we import all of the modules used in creating the tfidf

import math
#imports math module to use more complex mathamatical equations in python
import os
#imports os module to navigate in the current directory - reading the videogames folder
import pickle
#imports pickle module to collect the tfidf-cosine matrix data into a binary pickle file 
#so it can be used on the other python files
#this means we do not have to run this piece of code each time we want to search for a query
import utlities as ut
#for shared functions between the creation and query python files, we store them in utilities. We can access them
#as .ut
import pandas as pd
#Pandas is used to read the title and metadata and put it on the text to enhance the search.
from nltk.corpus import stopwords
#imports from the nltk corpus library the stopwords module
#this allows to search for a single variance of a word and return documents containing others.

from bs4 import BeautifulSoup
#this imports from bs4 libarary - beautiful soup. This is used to get useful text from the html page
from nltk.tokenize import word_tokenize
#from the nltk libarary we import word_tokenize which returns a tokenzised copy of the text
import regex as re
#we import regular expressions to match certain patterns to prevent unwanted tokens such as punctuation
stops = set(stopwords.words('english'))
#we can generate a python set of stopwords here from the stopwords library to be used in the tokensing process.
import numpy as np
#the numpy module is used for creating and changing the array of tfidf scores.

#here we create some global initial variables, including the vocab, postings and docID dictionaries.
docIDs = {}
vocabIDCounter = 0
#this vocabID counter is initalised here. We use it to record the vocab ID.
docIDsCounter = 0
#similar concept as the vocabID counter but instead it is for the document IDs.
vocab = {}
postings = {}
htmltexts = {}

#we can use a dictionary to turn on or off the stopwords and stemming for testing the query with or without them
test_config = {"stopwords": False, "stemming": False}

#the tokenisnor takes in the html page text for each document in the videogames corpus and returns the 
# paragraph and/or div words, in a tokenised state without punctuation, in a lower case and without stopwords.
def Tokenizer(text):
    soup = BeautifulSoup(text, 'html.parser')
    #beautiful soups takes in the html raw text and parameter feature for beautiful soup - in our case an html parser
    paragraphs = soup.find_all('div')
    #beautiful soup takes text from all div tags
    cleanedTokens = []
    #initalise cleaned tokens list to return
    for p in paragraphs:
        cleanedParagraphs = p.get_text()
        #we can use the get text function from beautiful soup to get the text from the div paragraphs.
        reS1 = re.sub("<.+?>|\n", "", str(cleanedParagraphs))
        #using rege we can remove punctuation from our paragraphs of text. This is because the user will be 
        #querying video game titles, genres and other pieces of data that typically don't use punctuation. 
        tokens = word_tokenize(reS1)
        #we can use the nltk word tokenise to tokenise each word in the paragraph breaking them down into a list
        #This makes it much easier for analysis and standadises each word in the sentance.
        for token in tokens:
            if token not in stops and token != "," and token != " ":
                cleanedTokens.append(token.lower())
                #appends the cleaned tokens and turns all text to lower case to avoid letter case issues when 
                #the user is querying
    return cleanedTokens




# loads the text files into a dictionary.

directory='videogames'
print(os.getcwd())
filenames = os.listdir(directory)
#lists the directory for filenames to be directory (which = "videogames") - the folder we are working in.
n=2
df_metadata_labels = pd.read_csv("videogame-labels.csv")
#we use pandas to get the different meta data labels such as the title, genre and publisher
for fname in filenames:
    fname_dir = os.path.join(directory, fname)
    f = open(fname_dir, "r", encoding='utf-8')
    text = f.read()
    cut_metadata = df_metadata_labels[df_metadata_labels['url'].str[26:] == fname]
    if not cut_metadata.empty:
        row_as_string = cut_metadata.iloc[0].str.cat(sep=', ')[26:] 
        #using iloc gets rid of "videogame/ps2.gamespy.com/" path before title - seperating with comma.
        for i in range(n):
            text+= " "
            text+=row_as_string
    f.close()
    
    docIDs[docIDsCounter] = fname
    #sets the dictionary value as the filename and the key is the docID 
    htmltexts[docIDsCounter] = text
    #html text for testing purposes only

    tokens = set(Tokenizer(text))
    #the tokens set that uses the tokenizor function taking in the text parameter.
    if test_config["stopwords"]:
        tokens = ut.remove_stopwords(tokens)
    
    if test_config["stemming"]:
        tokens = ut.stemmer(tokens)

    for t in tokens:
        if t in vocab.keys(): 
            #if the token is in the vocab dictionary keys
            vocabID = vocab[t]
            #the vocab id is set by vocab dictionary made up from the specific vocab token taken from the tokenizor.
        else: 
            vocab[t] = int(vocabIDCounter)
            #the vocab token will have its value being the ID count.
            vocabID = vocabIDCounter
            vocabIDCounter += 1
            #will increase by +1 for the next vocab

        if vocabID in postings.keys():
            postings[vocabID].append(docIDsCounter)
            #if the vocab ID is in the posting dictionary keys, then the docIDs will be appended
            #to the posting vocabIDs
        else:
            postings[vocabID] = [docIDsCounter]
            #else the postings vocab IDs will have the doc IDs count set to it.
            
        
    docIDsCounter += 1
    #increases by 1 for doc IDs to count the documents.






count_matrix = [0]*len(docIDs.keys())
#creates a pre-populated matrix that will be the length of the video game html document amount
tf_idf = [0]*len(docIDs.keys())
#simiarly it does so for the tf_IDF scores

#this for loop will look through each of the document titles (keys)
for key in docIDs.keys():
    document_vector = [0]*len(vocab.keys())
    #defines a document vector as at the length of the entire vocabulary dictionary 
    fname = docIDs[key]
    fname_directory = os.path.join(directory, fname)
    #joins the file name directory with the videogame folder 
    f = open(fname_directory, "r", encoding='utf-8')
    #opens the file name videogames folder to read it.    
    
    text = f.read()
    #reads the file, turning the entire file into a pure unprocessed text.
    f.close()
    #best practice, closes file.
    
    tokens = Tokenizer(text)
    #processes the text with the tokenizor function
    
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
#n is the total amount of all documents
tfidf_matrix = np.array(count_matrix).T.astype("float64") 
#.t transposes - so it flips the rows and columns of our matrix
for doc_tfs in tfidf_matrix:
    weighted_tf_array = []
    document_freq = len(postings[count])
    for doc_term_freq in doc_tfs:
        weighted_tf = round(math.log((1+doc_term_freq))*math.log(N/document_freq), 2)
        #here we calculate our weighted term frequency score.
        weighted_tf_array.append(weighted_tf)
    tfidf_matrix[count] = np.array(weighted_tf_array)

    count+=1
print(tfidf_matrix)
# 'wb' means write binary mode # Unpickling the object from the file with open("data.pkl", "rb") as file: 
# 'rb' means read binary mode 


#at the end we will pickle our matrix and store it in a pickle file. This means we can run this process once,
#which will increase the speed of our program when querying signifigantly. 
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