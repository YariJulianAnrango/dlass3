import matplotlib.pyplot as plt
import numpy as np

with open("./results/results_rnn_torch_nn.txt", "r") as file:
    dat = file.read().strip().split(",")
    dat = np.array(dat).astype(float)

with open("./results/results_rnn.txt", "r") as file:
    dat1 = file.read().strip().split(",")
    dat1 = np.array(dat1).astype(float)

with open("./results/results_lstm_torch.txt") as file:
    dat2 = file.read().strip().split(",")
    dat2 = np.array(dat2).astype(float)

plt.plot(dat)
plt.plot(dat1)
plt.plot(dat2)
plt.title("Custom RNN, torch RNN, and LSTM results")
plt.xlabel("Batch iteration")
plt.ylabel("Loss")
plt.legend(["Custom RNN", "Torch RNN", "Torch LSTM"])