import matplotlib.pyplot as plt
import re

# Raw training data
data = """
Start Training
Epoch: 1	Val Acc: 0.6702941176470587,	Val Acc EMA: 0.6786764705882353,	Avg Loss: 0.7016673825879328,	LR: 0.0001
Epoch: 2	Val Acc: 0.8891978609625671,	Val Acc EMA: 0.9020454545454538,	Avg Loss: 0.32897906010573696,	LR: 0.0001
Epoch: 3	Val Acc: 0.917593582887701,	Val Acc EMA: 0.9337032085561499,	Avg Loss: 0.23920659751972556,	LR: 0.0001
Epoch: 4	Val Acc: 0.9326336898395728,	Val Acc EMA: 0.9480347593582882,	Avg Loss: 0.19810982001176075,	LR: 0.0001
Epoch: 5	Val Acc: 0.9460695187165773,	Val Acc EMA: 0.9572593582887685,	Avg Loss: 0.1733938748367454,	LR: 0.0001
Epoch: 6	Val Acc: 0.9535962566844904,	Val Acc EMA: 0.9614973262032083,	Avg Loss: 0.1449625637783827,	LR: 0.0001
Epoch: 7	Val Acc: 0.9488770053475932,	Val Acc EMA: 0.9641844919786089,	Avg Loss: 0.1405859130422352,	LR: 0.0001
Reduced learning rate to 5e-05
Epoch: 8	Val Acc: 0.9596925133689835,	Val Acc EMA: 0.9672192513368986,	Avg Loss: 0.11654311010891912,	LR: 5e-05
Epoch: 9	Val Acc: 0.9674197860962578,	Val Acc EMA: 0.970574866310162,	Avg Loss: 0.08378454605279605,	LR: 5e-05
Reduced learning rate to 2.5e-05
Epoch: 10	Val Acc: 0.9676871657754016,	Val Acc EMA: 0.9720588235294142,	Avg Loss: 0.08267719534498953,	LR: 2.5e-05
Epoch: 11	Val Acc: 0.973475935828879,	Val Acc EMA: 0.974465240641713,	Avg Loss: 0.057821785416438586,	LR: 2.5e-05
Reduced learning rate to 1.25e-05
Epoch: 12	Val Acc: 0.9735026737967942,	Val Acc EMA: 0.975641711229949,	Avg Loss: 0.05376170688601756,	LR: 1.25e-05
Epoch: 13	Val Acc: 0.9758422459893075,	Val Acc EMA: 0.9767245989304844,	Avg Loss: 0.04557933572405342,	LR: 1.25e-05
Reduced learning rate to 6.25e-06
Epoch: 14	Val Acc: 0.9759893048128366,	Val Acc EMA: 0.9768582887700561,	Avg Loss: 0.039626128471399596,	LR: 6.25e-06
Epoch: 15	Val Acc: 0.9771791443850294,	Val Acc EMA: 0.9774598930481311,	Avg Loss: 0.040196858026935456,	LR: 6.25e-06
Reduced learning rate to 3.125e-06
Epoch: 16	Val Acc: 0.977018716577543,	Val Acc EMA: 0.9772192513369017,	Avg Loss: 0.03751296077581218,	LR: 3.125e-06
Epoch: 17	Val Acc: 0.977259358288773,	Val Acc EMA: 0.9771791443850295,	Avg Loss: 0.033821900588340996,	LR: 3.125e-06
Epoch: 18	Val Acc: 0.9771925133689867,	Val Acc EMA: 0.977045454545457,	Avg Loss: 0.03293724224640767,	LR: 3.125e-06
Epoch: 19	Val Acc: 0.9777272727272756,	Val Acc EMA: 0.9775401069518741,	Avg Loss: 0.033954539554328665,	LR: 3.125e-06
Epoch: 20	Val Acc: 0.9773529411764738,	Val Acc EMA: 0.9772459893048155,	Avg Loss: 0.03422661324877883,	LR: 3.125e-06
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
plt.savefig('../results/plot_recall_step_specific.png')
plt.show()
