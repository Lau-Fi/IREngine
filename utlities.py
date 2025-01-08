from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    for word in tokens:
        if word not in stop_words:
            filtered_tokens.append(word)
    return filtered_tokens

def stemmer(tokens):
    stemmer = PorterStemmer()
    stemmed = []
    for token in tokens:
        stemmed_word = stemmer.stem(token)
        stemmed.append(stemmed_word)
    return stemmed