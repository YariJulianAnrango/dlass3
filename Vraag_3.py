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

class SimpleSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super(SimpleSeq2SeqModel, self).__init__()

        # 1) Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # 2) Linear layer applied to each token
        self.linear1 = nn.Linear(emb_dim, hidden_dim)

        # 3) ReLU activation
        self.relu = nn.ReLU()

        # 4) Global max pool along the time dimension
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 5) Linear layer
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden):
        # Input x: (batch, time)

        # 1) Embedding layer
        embedded = self.embedding(x)

        # 2) Linear layer applied to each token
        linear_output = self.linear1(embedded)

        # 3) ReLU activation
        relu_output = self.relu(linear_output)

        # 4) Global max pool along the time dimension
        pooled_output = self.global_max_pool(relu_output.permute(0, 2, 1))

        # 5) Linear layer
        linear_output_final = self.linear2(pooled_output.squeeze(dim=2))

        return linear_output_final

learning_rate = 0.001
num_epochs = 5
vocab_size = len(w2i)
emb_dim = 300
hidden_dim = 300

model = SimpleSeq2SeqModel(vocab_size, emb_dim, hidden_dim, numcls)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = [(x, y) for x, y in zip(x_train, y_train)]

# Training loop
loss_list = []
hidden = None
for epoch in range(num_epochs):
    total_loss = 0.0
    for input_batch, target_batch in tqdm(train_dataset):
        input, target = input_batch.to(device), target_batch.to(device).long()

        optimizer.zero_grad()

        output = model(input, hidden)

        loss = F.cross_entropy(output, target)

        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()
        loss_list.append(loss.item())
        # Print average loss for the epoch
    average_loss = total_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

with open("./results/results_baseline_model.txt", "w") as file:
    file.write(loss_list[0])
    for i in loss_list:
        file.write(","+str(i))