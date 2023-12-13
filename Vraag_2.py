from RNN import load_imdb
from Vraag_1 import get_batches

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
x_train, y_train = get_batches(x_train, y_train, batch_size=32)

def seq2seq(vocab_size, emb_dim, hidden_dim, num_classes, input_batches):

    outputs = []
    for input_tensor in input_batches:
        embedding = nn.Embedding(vocab_size, emb_dim)
        embedded = embedding(input_tensor)
        linear1 = nn.Linear(emb_dim, hidden_dim)
        linear_output = linear1(embedded)
        relu = nn.ReLU()
        relu_output = relu(linear_output)
        global_max_pool = nn.AdaptiveMaxPool1d(1)
        pooled_output = global_max_pool(relu_output.permute(0, 2, 1))
        linear2 = nn.Linear(hidden_dim, num_classes)
        linear_output_final = linear2(pooled_output.squeeze(dim=2))

        outputs.append(linear_output_final)

    outputs_tensor = torch.stack(outputs)
    return outputs_tensor

num_epochs = 10
token_size = len(w2i)
embedding_size = 300
hidden_size = 300
num_classes = 2  #because binary classification

all_outputs = seq2seq(token_size, embedding_size, hidden_size, num_classes, x_train)

print(all_outputs.shape)
