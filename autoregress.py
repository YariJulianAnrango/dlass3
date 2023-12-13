import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, vocab, device):
        super(LSTM, self).__init__()
        self.device = device

        self.hidden_dim =hidden_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

        self.linear = nn.Linear(hidden_dim, vocab)

        self.softmax = nn.Softmax(dim=2)
    def forward(self, x, h, c):
        embedded = self.embedding(x)

        b, t, e = embedded.size()
        if h is None:
            h = torch.zeros(1, b, self.hidden_dim, dtype=torch.float).to(self.device)
            c = torch.zeros(1, b, self.hidden_dim, dtype=torch.float).to(self.device)
        lstm_output, (hn, cn) = self.lstm(embedded, (h.detach(), c.detach()))

        linoutput = self.linear(lstm_output)
        # output = self.softmax(linoutput)

        return linoutput , (hn, cn)

