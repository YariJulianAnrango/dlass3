from rnn_data import load_ndfa, load_brackets
from padbatch_auto import get_batches
from autoregress import LSTM

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


# Load and preprocess data
# x_train, (i2w, w2i) = load_ndfa(n=150_000)
x_train, (i2w, w2i) = load_brackets(n=150_000)

x_train, y_train = get_batches(x_train, w2i)

# Init model
learning_rate = 0.001
num_epochs = 10
vocab_size = len(w2i)
emb_dim = 32
hidden_dim = 16

device = torch.device("cpu")

model = LSTM(vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim, vocab=vocab_size, device=device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Get sample
seed_seq = [w2i['.start'], w2i['('], w2i['('], w2i[')']]
seq_torch = torch.tensor(seed_seq).unsqueeze(0)

hidden = None
cell = None

seq_torch = seq_torch.to(device)
optimizer.zero_grad()

for i in range(10):
    output, (hidden, cell) = model(seq_torch, hidden, cell)
    next_probs = output[0, 1, :].detach().numpy()
    choice = np.random.choice(range(len(next_probs)), p=next_probs)
    seed_seq.append(choice)
    seq_torch = torch.tensor(seed_seq).unsqueeze(0)
    if choice == w2i[".end"]:
        print(seq_torch)
        break
