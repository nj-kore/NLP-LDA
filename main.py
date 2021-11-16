import nltk
from nltk.corpus import stopwords
import numpy as np
from collections import Counter

def init_nltk():
    nltk.download('gutenberg')
    nltk.download('stopwords')

def preprocess(book_name, split_word):
    text_tokens = nltk.corpus.gutenberg.words(book_name)
    #text_tokens = np.array(text_tokens)
    text_tokens = np.char.lower(text_tokens).tolist()
    text_tokens_counter = Counter(text_tokens)

    stop_words = set(stopwords.words('english'))
    text_tokens = np.array([w for w in text_tokens if not w.lower() in stop_words
                            and w.isalnum() and text_tokens_counter[w] >= 10])

    indexes = np.where(text_tokens == split_word)[0]
    text_tokens = text_tokens.tolist()



    np.append(indexes, len(text_tokens))
    docs = []

    prev_index = indexes[0]
    for index in indexes[1:]:
        docs.append(text_tokens[prev_index:index-1])
        prev_index = index

    return docs


if __name__ == "__main__":
    #init_nltk()
    docs = preprocess('austen-emma.txt', 'chapter')