import numpy as np
import sys
import CorpusHandler
import random


if __name__ == "__main__":
    # Fetch corpus
    docs, word_set, word_counter = CorpusHandler.preprocess()

    D = len(docs)
    V = len(word_set)

    # Set variables
    K = 20
    alpha = 0.1
    beta = 0.1
    delta = 10
    gamma = 1

    # We create a vocab so we only have to work with numbers when performing matrix multiplications in the algorithm
    vocab = dict()
    vocab_rev = []

    # Init dict and a reverse dict
    for i, w in enumerate(word_set):
        vocab[w] = i
        vocab_rev.append(w)

    # Here we create the different counters. Notice that they are filled with parameters such as alpha, beta, delta.
    # This is done to avoid special cases later on in the algorithm.

    # Topic, Word
    t_w = np.full((K, V), beta, dtype=int)

    # Document, Topic, Topic (transition matrix).
    d_t_t = np.full((D, K, K), alpha, dtype=int)

    # We override diagonals with alpha + delta
    for d in range(D):
        for k in range(K):
            d_t_t[d, k, k] = alpha + delta

    # Document, Topic
    d_t = np.full((D, K), K * alpha + delta, dtype=int)

    # Topic
    t_ = np.full(K, V*beta, dtype=int)

    # Init Z to random topics
    z_dict = [random.choices(range(K), k=len(docs[d])) for d in range(D)]

    # Topics starting a document
    t_start = np.full(K, gamma, dtype=int)

    # Extra 2d correspond to the +1 which occurs in the case where (Z_d, j-1) = (Z_d, j) != (Z_d, j+1)
    extra2d = np.eye(K, dtype=int)

    # Extra 3d correspond to the +1 which occurs in the case where (Z_d, j-1) = (Z_d, j) = (Z_d, j+1)
    extra3d = [[np.zeros(K) for _ in range(K)] for _ in range(K)]
    for k in range(K):
        extra3d[k][k][k] = 1

    # A vector with equal probabilites. (Used later on in some cases)
    equal_q = [1 / K for _ in range(K)]

    # Go through all docs and positions to init values of our counters
    for d in range(D):
        prev_k = -1
        for j in range(len(docs[d])):
            word = vocab[docs[d][j]]
            curr_k = z_dict[d][j]

            if prev_k != -1:
                d_t_t[d, prev_k, curr_k] += 1
            else:
                t_start[curr_k] += 1

            t_w[curr_k, word] += 1
            d_t[d, curr_k] += 1
            t_[curr_k] += 1

            prev_k = curr_k


    sample_iterations = 150

    # Go through all docs and words for many iterations
    for sample_itr in range(sample_iterations):
        print(sample_itr)
        for d in range(D):
            doc_size = len(docs[d])
            for j in range(doc_size):

                # Translate the word to a number
                word = vocab[docs[d][j]]

                # Extract old Z_dj
                old_k = z_dict[d][j]

                prev_k = -1
                next_k = -1

                # Check if we are at the start or end of document
                if j != 0:
                    prev_k = z_dict[d][j-1]
                if j != (doc_size - 1):
                    next_k = z_dict[d][j+1]

                # Remove presence of Z_dj from counters
                t_w[old_k, word] -= 1
                t_[old_k] -= 1
                d_t[d, old_k] -= 1

                if prev_k != -1:
                    d_t_t[d, prev_k, old_k] -= 1
                else:
                    t_start[old_k] -= 1

                if next_k != -1:
                    d_t_t[d, old_k, next_k] -= 1

                # We need to code 3 different cases. One for the beginning of doc, one for the end, and one for
                # everything else. The other special cases are already 'built in' to the vectors from the original
                # initialization

                if prev_k == -1:
                    q = t_w[:, word] * t_start[:] * d_t_t[d, :, next_k] / (t_[:] * d_t[d, :])
                elif next_k == -1:
                    q = t_w[:, word] * d_t_t[d, prev_k, :] / t_[:]
                else:
                    q = t_w[:, word] * (d_t_t[d, prev_k, :] + extra3d[prev_k][next_k]) * (d_t_t[d, :, next_k]) \
                        / (t_[:] * d_t[d, :] + extra2d[prev_k, :])

                # We came across that all elements in q sometimes equaled 0.
                # We could not figure out if this was a bug or not but in the event that this occurs, we give all topics
                # an equal probability
                q_sum = sum(q)
                if q_sum == 0:
                    q = equal_q
                else:
                    # Normalize
                    q = q / q_sum

                # Sample k. We use this inlined sample method since it is fast
                x = np.random.rand()
                cum = 0
                new_k = 0
                for new_k, p in enumerate(q):
                    cum += p
                    if x < cum:
                        break

                # Update the counts
                if prev_k != -1:
                    d_t_t[d, prev_k, new_k] += 1
                else:
                    t_start[new_k] += 1

                if next_k != -1:
                    d_t_t[d, new_k, next_k] += 1

                t_w[new_k, word] += 1
                d_t[d, new_k] += 1
                t_[new_k] += 1
                z_dict[d][j] = new_k


    filename = "output/HMTM_" + str(alpha) + "_" + str(delta) + "_" + str(K) + "_" + str(sample_iterations) + ".txt"
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

