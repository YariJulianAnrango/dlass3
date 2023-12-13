from rnn_data import load_imdb
from padconvertion import get_batches

import torch
import torch.nn as nn

import torch.optim as optim
from tqdm import tqdm

torch.manual_seed(1)
torch.mps.empty_cache()
device = torch.device("cpu")

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

x_train, y_train = get_batches(x_train, y_train, batch_size=32)

class Net(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        
        #rnn
        b, t, e = embedded.size()
        if hidden is None:
            hidden = torch.zeros(1, b, e, dtype=torch.float).to("cpu")
        if cell is None:
            cell = torch.zeros(1, b, e, dtype=torch.float).to("cpu")
        lstm_output, (h, c) = self.lstm(embedded, (hidden.detach(), cell.detach()))

        relu_output = self.relu(lstm_output)
        relu_perm = relu_output.permute(0, 2, 1)
        pooled_output = self.global_max_pool(relu_perm)
        pool_squeeze = pooled_output.squeeze()
        linear_output_final = self.linear2(pool_squeeze)

        return linear_output_final, (h,c)

num_epochs = 10
token_size = len(w2i)
embedding_size = 300
hidden_size = 300

train_dataset = [(x, y) for x, y in zip(x_train, y_train)]

# Training loop
batch_loss_it = 25
for l in [0.001, 0.0003, 0.0001]:
    model = Net(token_size, embedding_size, hidden_size, numcls)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=l)
    loss_crit = nn.CrossEntropyLoss()
    epoch_loss = []
    loss_list = []

    hidden = None
    cell = None
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

    with open(f"./results/learning_rate_exp/results_lstm_lr{l}.txt", "w") as file:
        file.write(str(epoch_loss[0]))
        for i in epoch_loss[1:]:
            file.write(","+str(i))