import numpy as np
import re

# def softmax(x):
#     e_x = np.exp(x - np.max(x)) # notice subtracting max
#     return e_x / e_x.sum()

def softmax_batch(x):
    # x: (batch_size, n)
    c = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def clean_text(text):
    # 1. remove newlines and carriage returns
    text = text.replace('\n', ' ').replace('\r', ' ') 
    # 2. insert spaces around punctuation
    text = re.sub(r'([.,!?;])', r' \1 ', text)
    # 3. lowercase
    text = text.lower()
    return text