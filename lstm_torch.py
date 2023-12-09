from rnn_data import load_imdb
from padconvertion import get_batches

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

torch.manual_seed(1)

device = torch.device("cpu")

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

x_train, y_train = get_batches(x_train, y_train, batch_size=32)

class Net(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        # 1) Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # 2) Linear layer applied to each token
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

        # 3) ReLU activation
        self.relu = nn.ReLU()

        # 4) Global max pool along the time dimension
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 5) Linear layer
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden, cell):

        # 1) Embedding layer
        embedded = self.embedding(x)

        # 2) RNN layer
        b, t, e = embedded.size()
        if hidden is None:
            hidden = torch.zeros(1, b, e, dtype=torch.float).to("cpu")
        if cell is None:
            cell = torch.zeros(1, b, e, dtype=torch.float).to("cpu")
        lstm_output, (h, c) = self.lstm(embedded, (hidden.detach(), cell.detach()))

        # 3) ReLU activation
        relu_output = self.relu(lstm_output)
        relu_perm = relu_output.permute(0, 2, 1)
        # 4) Global max pool along the time dimension
        pooled_output = self.global_max_pool(relu_perm)

        # 5) Linear layer
        pool_squeeze = pooled_output.squeeze()
        linear_output_final = self.linear2(pool_squeeze)

        return linear_output_final, (h,c)

learning_rate = 0.001
num_epochs = 10
vocab_size = len(w2i)
emb_dim = 300
hidden_dim = 300

model = Net(vocab_size, emb_dim, hidden_dim, numcls)
model.to(device)

loss_crit = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = [(x, y) for x, y in zip(x_train, y_train)]

# Training loop
loss_list = []
hidden = None
cell = None
batch_loss_it = 25
epoch_loss = []
for epoch in range(num_epochs):
    total_loss = 0.0
    batch_loss = 0
    for i, data in enumerate(tqdm(train_dataset)):
        input, target = data[0].to(device), data[1].to(device).long()

        optimizer.zero_grad()

        output, (hidden, cell) = model(input, hidden, cell)

        loss = loss_crit(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_loss += loss.item()
        if i % batch_loss_it == batch_loss_it-1:
            avg_loss_b = batch_loss/batch_loss_it
            batch_loss = 0
            loss_list.append(avg_loss_b)
        # Print average loss for the epoch
    average_loss = total_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
    epoch_loss.append(average_loss)

with open("./results/results_lstm_torch_epoch.txt", "w") as file:
    file.write(str(epoch_loss[0]))
    for i in epoch_loss:
        file.write(","+str(i))