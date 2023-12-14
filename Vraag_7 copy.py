import torch.distributions as dist
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from autoregress import LSTM
from padbatch_auto import get_batches
from rnn_data import load_brackets

def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome logits
    :param temperature: Sampling temperature. 1.0 follows the given
    distribution, 0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """
    if temperature == 0.0:
        return lnprobs.argmax()
    p = F.softmax(lnprobs / temperature, dim=0)
    #print(p)
    #print()
    cd = dist.Categorical(p)
    return cd.sample()

# Prepare data
x_train, (i2w, w2i) = load_brackets(n=150_000)

x_train, y_train = get_batches(x_train, w2i)
#print(x_train[-1])
train_dataset = [(x, y) for x, y in zip(x_train, y_train)]

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
criterium = nn.CrossEntropyLoss()


# Train loop
hidden = None
cell = None
epoch_loss = []
for epoch in range(num_epochs):
    # Get seed sample
    seed_seq = [w2i['.start'], w2i['('], w2i['('], w2i[')']]
    seq_torch = torch.tensor(seed_seq).unsqueeze(0).to(device)

    tot_loss = 0
    for i, data in enumerate(tqdm(train_dataset)):
        input, target = data[0], data[1].long()

        optimizer.zero_grad()
        output, (hidden, cell) = model(input, hidden, cell)
        loss = criterium(output.permute(0,2,1), target)

        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
    average_loss = tot_loss/len(train_dataset)
    epoch_loss.append(average_loss)
    print(f"Epoch: {epoch}, with loss: {average_loss}")
    print()
    model.eval()


    h = None
    c = None
    for i in range(10):
        output, (h, c) = model(seq_torch, h, c)
        sam = sample(output[0,1,:], temperature=0.3)
        seed_seq.append(sam)
        seq_torch = torch.tensor(seed_seq).unsqueeze(0).to(device)
        print(seq_torch)

    model.train()


