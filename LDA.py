import numpy as np
import sys
import CorpusHandler

if __name__ == "__main__":
    docs, word_set, word_counter = CorpusHandler.preprocess()

    # A check if we want to know how many tokens we have
    #token_sum = 0
    #for key, value in word_counter.items():
    #    token_sum += value

    D = len(docs)
    V = len(word_set)

    # Variables
    K = 50
    alpha = 0.01
    beta = 0.01

    # We create a vocab so we only have to work with numbers when performing matrix multiplications in the algorithm
    vocab = dict()
    vocab_rev = []

    # Init dict and a reverse dict
    for i, w in enumerate(word_set):
        vocab[w] = i
        vocab_rev.append(w)

    # Here we create the different counters

    # Topic, Word
    t_w = np.zeros((K, V), dtype=int)

    # Document, Topic
    d_t = np.zeros((D, K), dtype=int)

    # Topic
    t_ = np.zeros(K)

    # Topics for position in the texts
    z_dict = [[-1 for _ in range(len(docs[d]))] for d in range(D)]


    sample_iterations = 150

    # Go through all docs and words for many iterations
    for sample_itr in range(sample_iterations):
        print(sample_itr)
        for d in range(D):
            for j in range(len(docs[d])):
                # Translate the word to a number
                word = vocab[docs[d][j]]

                # Extract old Z_dj
                old_k = z_dict[d][j]

                # If Z_dj had not been set yet, we don't subtract here
                if old_k != -1:
                    t_w[old_k, word] -= 1
                    d_t[d, old_k] -= 1
                    t_[old_k] -= 1

                # Calculate probilities
                q = (alpha + d_t[d, :]) * (beta + t_w[:, word]) / (V * beta + t_[:])

                # Normalize
                q = q / sum(q)

                # Sample k. We use this inlined sample method since it is fast
                x = np.random.rand()
                cum = 0
                new_k = 0
                for new_k, p in enumerate(q):
                    cum += p
                    if x < cum:
                        break

                # Update according to new Z_dj
                t_w[new_k, word] += 1
                d_t[d, new_k] += 1
                t_[new_k] += 1
                z_dict[d][j] = new_k


    # Print it to file
    filename = "output/LDA_" + str(alpha) + "_" + str(K) + "_" + str(sample_iterations) + ".txt"
    f = open(filename, 'w')
    original_stdout = sys.stdout
    sys.stdout = f  # Change the standard output to the file we created.

    for k in range(K):
        rel_dict = dict()
        for key in range(V):
            word = vocab_rev[key]

            # Calcuate scores through relative frequency
            rel_dict[word] = (t_w[k, key] / word_counter[word], word_counter[word])

        # Sort first on relative frequency and then on overall word count
        sorted_words = dict(sorted(rel_dict.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True))
        for i, item in enumerate(sorted_words.items()):

            # Print top 20 words
            if i > 20:
                break
            print(item[0], item[1][0], item[1][1])
        print()

    sys.stdout = original_stdout  # Reset the standard output to its original value






