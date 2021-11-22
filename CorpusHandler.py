import json
import nltk
from nltk.corpus import stopwords
import numpy as np
from collections import Counter

# download stopwords from nltk
def init_nltk():
    nltk.download('stopwords')

def preprocess():
    # Opening JSON file
    f = open('train.jsonl', 'r', errors='ignore')

    # we use nltks stopwords
    stop_words = set(stopwords.words('english'))
    json_list = list(f)
    docs = []

    # Go through each wikipedia document
    for json_str in json_list:
        doc = []
        result = json.loads(json_str)
        # Merge all sentences of a document
        for sentence in result['content']:
            doc += sentence['tokens']

        # We only include docs that are at least 400 characters long.
        if len(doc) > 400:
            docs.append(doc)

    # Counter for all words
    text_tokens_counter = Counter()

    # We loop over all documents and update the counter continuously
    for i, text_tokens in enumerate(docs):
        # Make to lowercase
        text_tokens = np.char.lower(text_tokens).tolist()

        # Update counter
        text_tokens_counter.update(text_tokens)

        # Reassign text_tokens to docs
        docs[i] = text_tokens

    # Here we get rid of commas, periods etc as well as stop words. We also remove words that occur less than 10 times
    for i, text_tokens in enumerate(docs):
        docs[i] = [w for w in text_tokens if not w.lower() in stop_words
                                and w.isalnum() and text_tokens_counter[w] >= 10]


    # We have to make a new counter now since a lot has been striped away.
    word_counter = Counter()

    # Update said counter
    for text_tokens in docs:
        word_counter.update(text_tokens)


    return docs, set(word_counter.keys()), word_counter

#yo = preprocess()
#hi = 2

