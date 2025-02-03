import matplotlib.pyplot as plt
import re

# Raw training data
data = """
Epoch: 1	Val Acc: 0.6748395721925132,	Val Acc EMA: 0.6897994652406416,	Avg Loss: 0.6831404522821485,	LR: 0.0001
Epoch: 2	Val Acc: 0.8894117647058819,	Val Acc EMA: 0.9061898395721922,	Avg Loss: 0.31403864573596013,	LR: 0.0001
Epoch: 3	Val Acc: 0.9224732620320854,	Val Acc EMA: 0.934786096256685,	Avg Loss: 0.2322151387609026,	LR: 0.0001
Epoch: 4	Val Acc: 0.9336631016042788,	Val Acc EMA: 0.9485695187165765,	Avg Loss: 0.18467471698638713,	LR: 0.0001
Epoch: 5	Val Acc: 0.9436631016042782,	Val Acc EMA: 0.9587834224598919,	Avg Loss: 0.15061747641390769,	LR: 0.0001
Epoch: 6	Val Acc: 0.9527139037433145,	Val Acc EMA: 0.963943850267379,	Avg Loss: 0.13454773331922706,	LR: 0.0001
Epoch: 7	Val Acc: 0.9583957219251336,	Val Acc EMA: 0.9676871657754019,	Avg Loss: 0.12882422143338915,	LR: 0.0001
Reduced learning rate to 5e-05
Epoch: 8	Val Acc: 0.9613770053475927,	Val Acc EMA: 0.9700267379679163,	Avg Loss: 0.11441784617838424,	LR: 5e-05
Epoch: 9	Val Acc: 0.9703208556149742,	Val Acc EMA: 0.97381016042781,	Avg Loss: 0.07229256348457756,	LR: 5e-05
Reduced learning rate to 2.5e-05
"""


# Parsing function
def parse_training_log(data):
    epochs = []
    val_acc = []
    val_acc_ema = []
    avg_loss = []
    lr = []

    lines = data.splitlines()
    for line in lines:
        match = re.match(
            r"Epoch: (\d+)\s+Val Acc: ([\d.]+),\s+Val Acc EMA: ([\d.]+),\s+Avg Loss: ([\d.]+),\s+LR: ([\de-]+)", line)
        if match:
            epoch = int(match.group(1))
            acc = float(match.group(2))
            acc_ema = float(match.group(3))
            loss = float(match.group(4))
            learning_rate = float(match.group(5))

            epochs.append(epoch)
            val_acc.append(acc)
            val_acc_ema.append(acc_ema)
            avg_loss.append(loss)
            lr.append(learning_rate)

    return epochs, val_acc, val_acc_ema, avg_loss, lr


# Parse data
epochs, val_acc, val_acc_ema, avg_loss, lr = parse_training_log(data)

# Plotting
plt.figure(figsize=(12, 5))

# Average Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, avg_loss, label="Training Loss", marker='o', color='orange')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.xticks(epochs[1::2])
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc, label="Val Acc", marker='o')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.xticks(epochs[1::2])
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)






# Adjust layout
plt.tight_layout()
plt.savefig('../results/plot(recurrent_memory).png')
plt.show()
