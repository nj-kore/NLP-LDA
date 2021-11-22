import nltk
from nltk.corpus import stopwords
import numpy as np
from collections import Counter
import sys
import CorpusHandler

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
    #docs_emma, word_set_emma, word_counter_emma = preprocess('austen-emma.txt', 'chapter')
    #docs_parents, word_set_parents, word_counter_parents = preprocess('austen-persuasion.txt', 'chapter')
    #docs_sense, word_set_sense, word_counter_sense = preprocess('austen-sense.txt', 'chapter')

    #docs = docs_emma + docs_parents + docs_sense
    #word_set = set.union(word_set_emma, word_set_parents, word_set_sense)
    #word_counter = word_counter_emma + word_counter_parents + word_counter_sense

    docs, word_set, word_counter = CorpusHandler.preprocess()
    D = len(docs)
    V = len(word_set)
    K = 5
    alpha = 0.1
    beta = 0.1
    delta = 5
    gamma = 1

    vocab = dict()
    vocab_rev = []

    for i, w in enumerate(word_set):
        vocab[w] = i
        vocab_rev.append(w)

    w_t = np.zeros((K, V), dtype=int)

    t_t_d = np.zeros((D, K, K), dtype=int)

    t_d = np.zeros((D, K), dtype=int)

    t_ = np.zeros(K, dtype=int)

    z_dict = [[-1 for _ in range(len(docs[d]))] for d in range(D)]

    t_start = np.zeros(K, dtype=int)

    sample_iterations = 100
    for sample_itr in range(sample_iterations):
        print(sample_itr)
        for d in range(D):
            prev_k = -1
            for j in range(len(docs[d])):
                word = vocab[docs[d][j]]
                old_k = z_dict[d][j]

                if old_k != -1:
                    w_t[old_k, word] -= 1

                    t_[old_k] -= 1

                    if prev_k == -1:
                        t_start[old_k] -= 1
                    elif t_t_d[d, prev_k, old_k] != 0:
                        t_t_d[d, prev_k, old_k] -= 1

                if(prev_k != -1):
                    q = (alpha + t_t_d[d, prev_k, :]) * (beta + w_t[:, word]) / (V * beta + t_[:] * t_d[:])
                else:
                    q = (gamma + t_start[:]) * (beta + w_t[:, word]) / (V * beta + t_[:])

                q = q / sum(q)

                x = np.random.rand()
                cum = 0
                new_k = 0
                for new_k, p in enumerate(q):
                    cum += p
                    if x < cum:
                        break


                if prev_k == -1:
                    t_start[new_k] += 1
                else:
                    t_t_d[d, prev_k, new_k] += 1


                prev_k = new_k

                w_t[new_k, word] += 1
                t_[new_k] += 1
                z_dict[d][j] = new_k

    #thetas = [np.random.dirichlet(np.array(topic_count[d]) + alpha) for d in range(D)]
    #phis = [np.random.dirichlet(filename = "LDA_" + str(alpha) + "_" + str(K) + "_" + str(sample_iterations) + ".txt"
    filename = "output/HMTM_" + str(alpha) + "_" + str(K) + "_" + str(sample_iterations) + ".txt"
    f = open(filename, 'w')
    original_stdout = sys.stdout
    sys.stdout = f  # Change the standard output to the file we created.

    for k in range(K):
        rel_dict = dict()
        for key in range(V):
            word = vocab_rev[key]
            rel_dict[word] = w_t[k, key] / word_counter[word]

        sorted_words = dict(sorted(rel_dict.items(), key=lambda item: item[1], reverse=True))
        for i, item in enumerate(sorted_words.items()):
            if i > 20:
                break
            print(item)
        print()

    sys.stdout = original_stdout  # Reset the standard output to its original valuenp.array(list(word_count[k].values())) + beta) for k in range(K)]

