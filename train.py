import numpy as np
import math #sqrt用
from util import *



def main():
    np.random.seed(46) #学籍番号下2桁

    # 1. make a corpus and dictionary
    with open('data/corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    words = clean_text(text).split() # simple tokenization
    vocab = list(set(words))
    word2idx = {w: i for i, w in enumerate(vocab)} # {'I': 0, 'have': 1, 'to': 2, 'eat': 3}
    n = len(vocab) # vocab size
    k = 128 # dimension of word vector(initially small value)

    # 2. make training data
    window_size = 5
    contexts_list = []
    targets_list = []
    for i in range(window_size, len(words) - window_size):
        target = word2idx[words[i]]
        # make context from back and front words
        context = []
        for j in range(-window_size, window_size + 1):
            if j!=0: # skip the center word
                context.append(word2idx[words[i + j]])
        contexts_list.append(context)
        targets_list.append(target)

    # 3. initialize weight matrices
    A = (np.random.rand(n, k) * 2 - 1) / math.sqrt(k)
    B = (np.random.rand(n, k) * 2 - 1) / math.sqrt(k)


    # 4. train
    A, B = train_linear(k, n, contexts_list, targets_list, A, B)


    # 5. show word vectors
    #for i, word in enumerate(vocab):
    #    print(f'{word}: {A[i]}')

    # 6. save matrices
    np.save('lA.npy', A)
    np.save('lB.npy', B)

    # 7. save dictionary
    import pickle
    with open('word2idx.pkl', 'wb') as f:
        pickle.dump(word2idx, f)
    
    print("\n Training finished. Matrices and dictionary are saved.")


# 行列計算による圧縮・復元
def train_linear(
        k: int, # dim of vector
        n: int, # vocab size
        contexts_list: list, #list of neighboring words ID
        targets_list: list, # list of the center word
        A,
        B
    ):

    epoch = 500
    batch_size = 512 # Batch size
    gamma = 0.1 # training rate
    num_data = len(targets_list)

    # make lists into numpy arrays
    contexts_array = np.array(contexts_list) # (num_data, 2*window_size)
    targets_array = np.array(targets_list)   # (num_data,)

    # 繰り返し処理
    for m in range(epoch):
        # shuffle data
        indices = np.random.permutation(num_data)
        total_loss = 0.0
        

        for i in range(0, num_data, batch_size):
            batch_indices = indices[i:i+batch_size] # take a slice
            S = len(batch_indices)

            # !!!avoid for loop by matrix calculation!!!
            # variables for gradient accumulation
            #delta_A = np.zeros_like(A)
            #delta_B = np.zeros_like(B)
            #batch_loss = 0.0

            # mini-batch processing
            # for idx in batch_indices:
            #     # input layer: create the average vector of sorrounding words
            #     ctx_ids = contexts_list[idx]
            #     Z = np.mean(A[ctx_ids], axis=0)

            #     # output layer: predict the center word
            #     u = B @ Z
            #     Y = softmax(u) # activation function

            #     # calculate the difference between prediction and true word
            #     Target_OneHot = np.zeros(n)
            #     Target_OneHot[targets_list[idx]] = 1
            #     E = Y - Target_OneHot

            #     # calculate gradients
            #     # updating output layer of B
            #     delta_B += np.outer(E, Z) # cross product
            #     # updating input layer of A
            #     E_A = B.T @ E
            #     # accumulate gradients for each context word
            #     for cidx in ctx_ids:
            #         delta_A[cidx] += E_A / len(ctx_ids)
                
            #     # record loss(cross-entropy)
            #     batch_loss += -np.log(Y[targets_list[idx]] + 1e-7)

            # get contexts and apply average
            ctx_ids = contexts_array[batch_indices] # (S, 2*window_size)
            Z_batch = np.mean(A[ctx_ids], axis=1) # (S, k)

            # output layer: predict the center word
            u = Z_batch @ B.T # (S, n)
            Y = softmax_batch(u) # (S, n)
            
            # calculate the difference between prediction and true word
            # not make one-hot vectors(huge matrix), subtract directly from array indexing
            E = Y.copy()
            E[np.arange(S), targets_array[batch_indices]] -=1

            # calculate gradients
            dB = (E.T @ Z_batch) / S #delta_B (n, k)
            dA_shared = (E @ B) / (S * ctx_ids.shape[1]) # I don't understand

            # update B
            B -= gamma * dB
            # update A
            for j in range(ctx_ids.shape[1]):
                np.add.at(A, ctx_ids[:, j], -gamma * dA_shared)

            # record loss(cross-entropy)
            batch_loss = -np.sum(np.log(Y[np.arange(S), targets_array[batch_indices]] + 1e-7)) # I don't understand
            total_loss += batch_loss

        

        # update total loss
        #total_loss += batch_loss

        # update A and B using avarage of mini-batch gradients
        #B -= gamma * (delta_B / S)
        #A -= gamma * (delta_A / S)

        if m % 50 == 0:
            print(f'{m}: Loss = {total_loss / num_data}')

    return A, B



if __name__ == '__main__':
    main()
