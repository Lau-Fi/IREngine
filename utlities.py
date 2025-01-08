# we need to import the from nltk stopwords and the stemmer which we will use the Porter Stemmer method for.
from nltk.corpus import stopwords
from nltk import PorterStemmer


#remove stopwords function will remove the stopwords from the tokens, stopwords are words such as "and", "or" etc
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    for word in tokens:
        if word not in stop_words:
            filtered_tokens.append(word)
    return filtered_tokens

#this will stem the words bringing words back to their root etomology such as running - run.
def stemmer(tokens):
    stemmer = PorterStemmer()
    stemmed = []
    for token in tokens:
        stemmed_word = stemmer.stem(token)
        stemmed.append(stemmed_word)
    return stemmed