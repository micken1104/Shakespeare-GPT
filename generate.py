import numpy as np
import pickle
from util import *

def main():
    # 1. load the word2idx dictionary
    A = np.load('lA.npy')
    with open('word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)
    
    # 2. prepare idx2word(dictionary for reverse lookup)
    idx2word = {i: w for w, i in word2idx.items()}

    # 3. settings
    start_word = "romeo" # as you like
    length = 20 # length of sentence to generate
    if start_word not in word2idx:
        print(f"'{start_word}' not in dictionary.")
        return
    
    current_id = word2idx[start_word]
    sentence = [start_word]

    # 4. generate a sentence
    print(f"--- Generating text starting with '{start_word}' ---")

    # to avoid same output every time
    used_ids = []

    for _ in range(length):
        # get the word vector of the current word
        z = A[current_id]

        # predict the next word
        B = np.load('lB.npy')
        # add temperature 
        temperature = 0.7
        u = (B @ z) / temperature

        # last 3 words restriction
        for last_id in used_ids:
            u[last_id] = -1e10  # very small value to avoid re-selection

        # transform to probabilities using softmax
        probs = softmax_batch(u.reshape(1, -1))[0]

        # sample the next word id from the probability distribution
        # this generates fluctuations in the output text
        next_id = np.random.choice(len(probs), p=probs)

        word = idx2word[next_id]
        if word in [".", "!", "?"]:
            sentence.append(word + "\n") # add newline after sentence-ending punctuation
        else:
            sentence.append(word)
        current_id = next_id
        used_ids.append(next_id)

    print(' '.join(sentence))

if __name__ == '__main__':
    main()