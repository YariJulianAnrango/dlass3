import torch.distributions as dist
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

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

#import data
x_train, (i2w, w2i) = load_brackets(n=150_000)

x_train, y_train = get_batches(x_train, w2i)
print(x_train)
train_dataset = [(x, y) for x, y in zip(x_train, y_train)]

#define parameters
learning_rate = 0.001
num_epochs = 20
vocab_size = len(w2i)
emb_dim = 32
hidden_dim = 16

device = torch.device("cpu")

model = LSTM(vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim, vocab=vocab_size, device=device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterium = nn.CrossEntropyLoss(ignore_index=0)


hidden = None
cell = None
epoch_loss = []

#Tensorboard
writer = SummaryWriter()

#Updates weights and biases and calculate loss per token and calculate average loss.
for epoch in range(num_epochs):
    tot_loss = 0
    num_tokens = 0
    for i, data in enumerate(tqdm(train_dataset)):
        input, target = data[0], data[1].long()
        optimizer.zero_grad()
        output, (hidden, cell) = model(input, hidden, cell)
        loss = criterium(output.permute(0,2,1), target)
        num_tokens += target.sum().item()
        loss.backward()
        #To scale the gradient before updating the model
        clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        tot_loss += loss.item()
    average_loss = tot_loss / num_tokens
    epoch_loss.append(average_loss)

    print(f"Epoch: {epoch}, with loss: {average_loss}")
    writer.add_scalar('Loss/Train', average_loss, num_epochs)


#make function to generate sequence, using xtrain data as input data
#in line: generated_sequence = torch.cat((input_sequence, next_element.unsqueeze(0).unsqueeze(0)))
#RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 4 but got size 1 for tensor number 1 in the list.
"""
def make_sequence(model, input_sequence):
    hidden = None
    cell = None
    model.eval()
    output, (hidden, cell) = model(input_sequence, hidden, cell)
    next_element = sample(output[0,1,:])
    #print('hello', input_sequence.shape)
    #print('hoi', next_element.unsqueeze(0).unsqueeze(0).shape)
    generated_sequence = torch.cat((input_sequence, next_element.unsqueeze(0).unsqueeze(0))) 
    return generated_sequence

#function to calculate accuracy 
def compute_accuracy(predictions, targets):
    if predictions.shape != targets.shape:
        raise ValueError("Predicted and target sequences must have the same shape.")

    correct_predictions = torch.sum(predictions == targets).item()
    total_predictions = predictions.numel()
    accuracy = correct_predictions / total_predictions
    return accuracy

#Print accuracy between y_train and predicted sentences. 
#
accuracies = []
for i in range(10):
    x_train_sequence = x_train[i]
    y_train_sequence = y_train[i]
    generated_sequence = make_sequence(model, x_train_sequence)
    print(generated_sequence.shape)
    print(y_train_sequence.shape)
    accuracy = compute_accuracy(generated_sequence, y_train_sequence)
    accuracies.append(accuracy)

overall_accuracy = sum(accuracies) / len(accuracies)
print("Overall Accuracy:", overall_accuracy)
"""

#plot loss
plt.plot(epoch_loss)
plt.title('Loss over 50 epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

writer.close()