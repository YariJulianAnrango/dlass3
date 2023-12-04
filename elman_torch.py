from rnn_data import load_imdb
from padconvertion import get_batches

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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
        self.elman = nn.RNN(emb_dim, hidden_dim, batch_first=True)

        # 3) ReLU activation
        self.relu = nn.ReLU()

        # 4) Global max pool along the time dimension
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 5) Linear layer
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden):

        # 1) Embedding layer
        embedded = self.embedding(x)

        # 2) RNN layer
        b, t, e = embedded.size()
        if hidden is None:
            hidden = torch.zeros(1, b, e, dtype=torch.float).to("cpu")
        elman_output, hidden_elman = self.elman(embedded, hidden.detach())

        # 3) ReLU activation
        relu_output = self.relu(elman_output)
        relu_perm = relu_output.permute(0, 2, 1)
        # 4) Global max pool along the time dimension
        pooled_output = self.global_max_pool(relu_perm)

        # 5) Linear layer
        pool_squeeze = pooled_output.squeeze()
        linear_output_final = self.linear2(pool_squeeze)

        return linear_output_final, hidden_elman

learning_rate = 0.001
num_epochs = 5
vocab_size = len(w2i)
emb_dim = 300
hidden_dim = 300

model = Net(vocab_size, emb_dim, hidden_dim, numcls)
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

        output, hidden = model(input, hidden)

        loss = F.cross_entropy(output, target)

        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()
        loss_list.append(loss.item())
        # Print average loss for the epoch
    average_loss = total_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

with open("./results/results_rnn_torch.txt", "w") as file:
    file.write(str(loss_list[0]))
    for i in loss_list:
        file.write(","+str(i))