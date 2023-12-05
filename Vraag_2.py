from RNN import load_imdb
from Vraag_1 import get_batches

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device('cpu')

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

x_train, y_train = get_batches(x_train, y_train, batch_size=32)

def simple_seq2seq_model(vocab_size, emb_dim, hidden_dim, num_classes, input_batches):
    # 1) Embedding layer
    embedding = nn.Embedding(vocab_size, emb_dim)

    # List to store outputs for each batch
    outputs = []

    for input_tensor in input_batches:
        embedded = embedding(input_tensor)

        # 2) Linear layer applied to each token
        linear1 = nn.Linear(emb_dim, hidden_dim)
        linear_output = linear1(embedded)

        # 3) ReLU activation
        relu = nn.ReLU()
        relu_output = relu(linear_output)

        # 4) Global max pool along the time dimension
        global_max_pool = nn.AdaptiveMaxPool1d(1)
        pooled_output = global_max_pool(relu_output.permute(0, 2, 1))

        # 5) Linear layer
        linear2 = nn.Linear(hidden_dim, num_classes)
        linear_output_final = linear2(pooled_output.squeeze(dim=2))

        outputs.append(linear_output_final)

    # Stack outputs into a single tensor
    outputs_tensor = torch.stack(outputs)

    return outputs_tensor

# Example usage
vocab_size = len(w2i)
emb_dim = 300       
hidden_dim = 300   
num_classes = 2  # because binary classification

# Forward pass for all batches
all_outputs = simple_seq2seq_model(vocab_size, emb_dim, hidden_dim, num_classes, x_train)

# Print the overall output shape
print(all_outputs.shape)
