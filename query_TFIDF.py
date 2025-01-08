import json, math, pickle
import utlities as ut
import numpy as np
with open('docids.pkl', 'rb') as f: 
    docIDs = pickle.load(f)
with open('vocab.pkl', 'rb') as f: 
    vocab = pickle.load(f)
with open('postings.pkl', 'rb') as f: 
    postings = pickle.load(f)
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)
with open('test_config.pkl', 'rb') as f:
    test_config = pickle.load(f)
N = len(docIDs)

"""If we have a seperate method to create the query vector:
    if we are doing it as a part of the cosine similarity function
    we would be calculating the query vector each time we run it,
    making it a redundant calculation and slowing down the code.
    """
def get_query_vector(query_tokens, N):
    query_vector = [0]*len(vocab.keys())
    for term in query_tokens:
        count = 0
        try:
            term_ID = vocab[term]
        except:
            print(f"Not found {term}")
            continue
        print(f"{term_ID}, {term}")
        postings_term = postings[term_ID]
        for query_term in query_tokens:
            if query_term == term:
                count+=1
        wtq = (1+math.log(count))*math.log(N/len(postings_term))
        query_vector[term_ID] = wtq
    return query_vector


def cosine_score_get(query_vector, document_vector):
    results_vector = [0]*len(query_vector)
    for i in range(len(query_vector)):
        results_vector[i] = query_vector[i]*document_vector[i]
    return sum(results_vector)/get_length(document_vector)*get_length(query_vector)
        

def get_length(vector):
    sum_squares = 0
    for number in vector:
        sum_squares += number**2
    return math.sqrt(sum_squares)

while True:  #loop for user query
    results = {} #the results dictionary, each query word and the pages it appears on.
    query = input("What would you like to query?") 
    choice = input("What style of calculating multiple query words: (b for query_vector, a for addition, c for user relevance): ")
    if query == "":
        break
    query = query.lower()
    querylist = query.split(" ") #split the user query by each word into a list
    
    if test_config["stopwords"]:
        querylist = ut.remove_stopwords(querylist)
    
    if test_config["stemming"]:
        querylist = ut.stemmer(querylist)
    
    ranked_docs = []

    if choice == "a":
        stored_wordtfids = []
        for word in querylist:
            if word in vocab.keys():
                #step 1: find wordID from word
                wordID = vocab[word]
                #step 2: get wordID array from tfIDF scores
                word_tfidfs = tfidf_matrix[wordID]
                if len(stored_wordtfids) != 0:
                    added_final_tfidfs = stored_wordtfids+word_tfidfs
                stored_wordtfids = word_tfidfs

        #step 3: argsort tfidf docIDS
        ranked_docIDs_by_tfidfs = np.argsort(added_final_tfidfs)
        #step 4: use docid keys to get value
        for docID in ranked_docIDs_by_tfidfs:
            ranked_docs.append(docIDs[docID])
        print(ranked_docs[:10])

    elif choice == "b":
        scores = []
        query_tfidfs = get_query_vector(querylist, N)
        tfidf_matrix_flipped = tfidf_matrix.T
        for doc_tfidfs in tfidf_matrix_flipped:
            cosine_calculate = cosine_score_get(query_tfidfs, doc_tfidfs)
            scores.append(cosine_calculate)
                                                                                       
        # loop over document_vectors in count_matrix_transpose.
        # we can use the DocIDS for that
        # for each document_vector:
        # calculate cosine simularity between document vector and query vector
        # store the results in scores at position: documentID
        # use scores to get a ranked list of documents
        scores = np.array(scores)
        ranked_docIDS = np.argsort(-scores)
        print(ranked_docIDS)
        for doc_ID in ranked_docIDS:
            ranked_docs.append(f'{doc_ID}: {docIDs[doc_ID]}')
        print(ranked_docs[:10])
        
    elif choice == "c":
        scores = []
        augmented_scores = []
        query_tfidfs = get_query_vector(querylist, N)
        tfidf_matrix_flipped = tfidf_matrix.T



        
        for doc_tfidfs in tfidf_matrix_flipped:
            cosine_calculate = cosine_score_get(query_tfidfs, doc_tfidfs)
            scores.append(cosine_calculate)
                                                                                       
        # loop over document_vectors in count_matrix_transpose.
        # we can use the DocIDS for that
        # for each document_vector:
        # calculate cosine simularity between document vector and query vector
        # store the results in scores at position: documentID
        # use scores to get a ranked list of documents
        scores = np.array(scores)
        ranked_docIDS = np.argsort(-scores)
        print(ranked_docIDS)
        for doc_ID in ranked_docIDS:
            ranked_docs.append(f'{doc_ID}: {docIDs[doc_ID]}')
        
        print(ranked_docs[:10])
        
        final_relevent_vector = [0]*len(query_tfidfs)

        user_relevance_feedback = int(input("What document is relevant: (pick a number or enter to skip): "))
        if user_relevance_feedback == "":
            break
        relevant_ranked_docs = []

        relevent_vector = tfidf_matrix_flipped[user_relevance_feedback]

        for index in range(len(relevent_vector)):
            final_relevent_vector[index] = relevent_vector[index]+query_tfidfs[index]

        for doc_tfidfs in tfidf_matrix_flipped:
            cosine_calculate = cosine_score_get(final_relevent_vector, doc_tfidfs)
            augmented_scores.append(cosine_calculate)
        
        augmented_scores = np.array(augmented_scores)
        ranked_docIDS = np.argsort(-augmented_scores)
        print(ranked_docIDS)
        for doc_ID in ranked_docIDS:
            relevant_ranked_docs.append(f'{doc_ID}: {docIDs[doc_ID]}')
        
        #Could be faster if we calculate only the top 10 rather than cutting the list.
        print(relevant_ranked_docs[:10])





#list 3 different techniques in order to improve the work.

#THREE DIFFERENT METHODS OF IMPLEMENTING BELOW - and compare in video presentation.
#We need the program to enable ranking of the documents with multiple query words, taking them all into account. 
# Furthermore, we have to return only the top 10 most relevent documents.
#We can use cosine ranking to get documents more close to the query.
#We can create a short dynamic summary below each HTML file title, highlighting the query word and its sentence - 
# maybe with 10 words or one sentence between the query word? (Keyword in Context)
#We need to take metadata into account such as the genre, this can be useful for ranking, 
# such as displaying documents with the same genre.
###
#Remove stop words? - partially complete

#and then... We are done! 