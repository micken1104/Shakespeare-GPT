import numpy as np
from util import *


def main():
    # load the word vectors
    A = np.load('lA.npy')

    # 2. prepare the dictionary(same as train.py))
    with open('data/corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    words = clean_text(text).split() # simple tokenization
    vocab = list(set(words))
    word2idx = {w: i for i, w in enumerate(vocab)} # {'I': 0, 'have': 1, 'to': 2, 'eat': 3}
    idx2word = {i: w for i, w in enumerate(vocab)}

    # test1
    query_word = 'love'.lower()
    print(f"--- Top similarities for '{query_word}' ---")
    query_v = A[word2idx[query_word]]

    similarities = []
    for i in range(len(vocab)):
        sim = cosine_similarity(query_v, A[i])
        similarities.append((idx2word[i], sim))
    
    # show by sorting similarities
    similarities.sort(key=lambda x: x[1], reverse=True)
    for word, score in similarities[:10]: # top 10
        print(f"{word}: {score:.4f}")
    print("\n")

    # test2
    combined_v = A[word2idx["i"]] + A[word2idx["have"]]
    print("--- Vector Arithmetic Test: 'i' + 'have' ---")

    arithmetic_sims = []
    for i in range(len(vocab)):
        sim = cosine_similarity(combined_v, A[i])
        arithmetic_sims.append((idx2word[i], sim))
    
    arithmetic_sims.sort(key=lambda x: x[1], reverse=True)
    for word, score in arithmetic_sims[:3]: # top 3
        print(f"Result cadidate: {word} (score: {score:.4f})")

def cosine_similarity(v1, v2):
    # returning closeness between two vectors from -1 to 1
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7) # i don't know why

if __name__ == '__main__':
    main()
