import matplotlib.pyplot as plt
import numpy as np

models = ["baseline_model", "lstm", "rnn", "rnn_torch"]
lrs = ["0.001", "0.003", "0.0003", "0.0001"]

data = {}
for model in models:
    for l in lrs:
        with open(f"./results/learning_rate_exp/results_{model}_lr{l}.txt", "r") as file:
            dat = file.read().strip().split(",")
            dat = np.array(dat).astype(float)
        key = f"{model}{l}"
        data[key] = dat

fig, axs = plt.subplots(2, 2)
# lr 0.003
axs[0,0].plot(data["lstm0.003"])
axs[0,0].plot(data["rnn0.003"])
axs[0,0].plot(data["baseline_model0.003"])
axs[0,0].plot(data["rnn_torch0.003"])
axs[0,0].set_title("learning rate 0.003", fontsize = 12)
axs[0,0].set_ylabel("loss")

# lr 0.001
axs[0,1].plot(data["lstm0.001"])
axs[0,1].plot(data["rnn0.001"])
axs[0,1].plot(data["baseline_model0.001"])
axs[0,1].plot(data["rnn_torch0.001"])
axs[0,1].set_title("learning rate 0.001", fontsize = 12)

# lr 0.0003
axs[1,0].plot(data["lstm0.0003"])
axs[1,0].plot(data["rnn0.0003"])
axs[1,0].plot(data["baseline_model0.0003"])
axs[1,0].plot(data["rnn_torch0.0003"])
axs[1,0].set_title("learning rate 0.0003", fontsize = 12)
axs[1,0].set_ylabel("loss")
axs[1,0].set_xlabel("epoch")

# lr 0.0001
axs[1,1].plot(data["lstm0.0001"])
axs[1,1].plot(data["rnn0.0001"])
axs[1,1].plot(data["baseline_model0.0001"])
axs[1,1].plot(data["rnn_torch0.0001"])
axs[0,1].legend(["lstm", "rnn", "baseline", "torch rnn"])
axs[1,1].set_title("learning rate 0.0001", fontsize = 12)
axs[1,1].set_xlabel("epoch")
fig.tight_layout()
plt.savefig("./figures/param_testing.png")