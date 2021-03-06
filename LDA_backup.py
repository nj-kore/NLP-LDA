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

    word_counter = Counter(text_tokens)
    indexes = np.where(text_tokens == split_word)[0]
    text_tokens = text_tokens.tolist()


    np.append(indexes, len(text_tokens))
    docs = []

    prev_index = indexes[0]
    for index in indexes[1:]:
        docs.append(text_tokens[prev_index:index-1])
        prev_index = index

    return docs, set(text_tokens), word_counter


if __name__ == "__main__":
    #init_nltk()
    docs, word_set, word_counter = preprocess('austen-emma.txt', 'chapter')
    D = len(docs)
    V = len(word_set)
    K = 10
    alpha = 0.1
    beta = 0.1

    word_dict = dict()
    word_count = dict()
    topic_dict = dict()
    topic_count = [[0 for _ in range(K)] for d in range(D)]
    topic_word_count = dict()
    z_dict = [[-1 for _ in range(len(docs[d]))] for d in range(D)]

    for w in word_set:
        word_dict[w] = 0

    for k in range(K):
        word_count[k] = word_dict.copy()
        topic_dict[k] = 0
        topic_word_count[k] = 0

    sample_iterations = 10
    for sample_itr in range(sample_iterations):
        for d in range(D):
            for j in range(len(docs[d])):
                #d = np.random.randint(0, D)
                #j = np.random.randint(0, len(docs[d]))
                word = docs[d][j]
                old_k = z_dict[d][j]
                if old_k != -1:
                    word_count[old_k][word] -= 1
                    topic_count[d][old_k] -= 1
                    topic_word_count[old_k] -= 1

                q = []
                cumsum = 0
                for k in range(K):
                    p = (alpha + topic_count[d][k]) * (beta + word_count[k][word]) / (V*beta + topic_word_count[k])
                    q.append(p)
                    cumsum += p


                q = [float(i) / cumsum for i in q]
                new_k = np.random.choice(K, 1, p=q)[0]
                word_count[new_k][word] += 1
                topic_count[d][new_k] += 1
                topic_word_count[new_k] += 1
                z_dict[d][j] = new_k

    #thetas = [np.random.dirichlet(np.array(topic_count[d]) + alpha) for d in range(D)]
    #phis = [np.random.dirichlet(np.array(list(word_count[k].values())) + beta) for k in range(K)]

    for k in range(K):
        rel_dict = dict()
        for key in word_counter.keys():
            rel_dict[key] = (word_count[k][key] * word_count[k][key]) / word_counter[key]

        sorted_words = dict(sorted(rel_dict.items(), key=lambda item: item[1], reverse=True))
        for i, item in enumerate(sorted_words.items()):
            if i > 10:
                break
            print(item)
        print()







