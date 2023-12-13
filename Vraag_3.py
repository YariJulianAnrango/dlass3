from rnn_data import load_imdb
from Vraag_1 import get_batches

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device('cpu')

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

x_train, y_train = get_batches(x_train, y_train, batch_size=32)

print(x_train)

torch.manual_seed(1)

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

    def forward(self, x):
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

num_epochs = 20
vocab_size = len(w2i)
emb_dim = 300
hidden_dim = 300

train_dataset = [(x, y) for x, y in zip(x_train, y_train)]

# Training loop
batch_loss_it = 25
for l in [0.003, 0.001, 0.0003, 0.0001]:
    model = SimpleSeq2SeqModel(vocab_size, emb_dim, hidden_dim, numcls)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=l)
    loss_crit = nn.CrossEntropyLoss()
    hidden = None
    loss_list = []
    epoch_loss = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_loss = 0
        for i, data in enumerate(tqdm(train_dataset)):
            input, target = data[0].to(device), data[1].to(device).long()

            optimizer.zero_grad()

            output = model(input)

            loss = loss_crit(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_loss += loss.item()
            if i % batch_loss_it == batch_loss_it - 1:
                avg_loss_b = batch_loss / batch_loss_it
                batch_loss = 0
                loss_list.append(avg_loss_b)
            # Print average loss for the epoch
        average_loss = total_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
        epoch_loss.append(average_loss)

    with open(f"./results/learning_rate_exp/results_baseline_model_lr{l}.txt", "w") as file:
        file.write(str(epoch_loss[0]))
        for i in epoch_loss[1:]:
            file.write(","+str(i))