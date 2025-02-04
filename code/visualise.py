import matplotlib.pyplot as plt
import re

# Raw training data
data = """
Start Training
Epoch: 1	Val Acc: 0.644264705882353,	Val Acc EMA: 0.6546524064171124,	Avg Loss: 0.717972962747834,	LR: 0.0001
Epoch: 2	Val Acc: 0.6979946524064171,	Val Acc EMA: 0.7101470588235301,	Avg Loss: 0.6228008082884854,	LR: 0.0001
Epoch: 3	Val Acc: 0.7145989304812834,	Val Acc EMA: 0.7264037433155092,	Avg Loss: 0.571346854502789,	LR: 0.0001
Epoch: 4	Val Acc: 0.7282352941176479,	Val Acc EMA: 0.7387165775401074,	Avg Loss: 0.5373723364733135,	LR: 0.0001
Epoch: 5	Val Acc: 0.87701871657754,	Val Acc EMA: 0.8818449197860958,	Avg Loss: 0.3170715068286211,	LR: 0.0001
Epoch: 6	Val Acc: 0.9038235294117649,	Val Acc EMA: 0.9172727272727277,	Avg Loss: 0.2559520522594041,	LR: 0.0001
Epoch: 7	Val Acc: 0.9226069518716584,	Val Acc EMA: 0.9365374331550803,	Avg Loss: 0.2038097951682852,	LR: 0.0001
Epoch: 8	Val Acc: 0.9307085561497315,	Val Acc EMA: 0.9440374331550795,	Avg Loss: 0.1848833365183614,	LR: 0.0001
Epoch: 9	Val Acc: 0.9396657754010691,	Val Acc EMA: 0.9494117647058811,	Avg Loss: 0.1654522311888168,	LR: 0.0001
Epoch: 10	Val Acc: 0.9442647058823518,	Val Acc EMA: 0.955601604278074,	Avg Loss: 0.15030298552794133,	LR: 0.0001
Epoch: 11	Val Acc: 0.9526069518716563,	Val Acc EMA: 0.9614037433155078,	Avg Loss: 0.1264482512135862,	LR: 0.0001
Reduced learning rate to 5e-05
Epoch: 12	Val Acc: 0.9545320855614967,	Val Acc EMA: 0.9647326203208553,	Avg Loss: 0.12127001791538987,	LR: 5e-05
Epoch: 13	Val Acc: 0.9646524064171133,	Val Acc EMA: 0.9679679144385038,	Avg Loss: 0.08491035365807607,	LR: 5e-05
Reduced learning rate to 2.5e-05
Epoch: 14	Val Acc: 0.9652673796791442,	Val Acc EMA: 0.9681684491978627,	Avg Loss: 0.08107271519757916,	LR: 2.5e-05
Epoch: 15	Val Acc: 0.9676336898395732,	Val Acc EMA: 0.9696791443850288,	Avg Loss: 0.059177048963468294,	LR: 2.5e-05
Reduced learning rate to 1.25e-05
Epoch: 16	Val Acc: 0.9685561497326209,	Val Acc EMA: 0.9695454545454558,	Avg Loss: 0.059245826796184925,	LR: 1.25e-05
Epoch: 17	Val Acc: 0.9692112299465258,	Val Acc EMA: 0.9703877005347615,	Avg Loss: 0.051218451388762645,	LR: 1.25e-05
Reduced learning rate to 6.25e-06
Epoch: 18	Val Acc: 0.9697994652406438,	Val Acc EMA: 0.9705080213903764,	Avg Loss: 0.043728090594765325,	LR: 6.25e-06
Epoch: 19	Val Acc: 0.9705748663101622,	Val Acc EMA: 0.9707085561497346,	Avg Loss: 0.040341998833194774,	LR: 6.25e-06
Reduced learning rate to 3.125e-06
Epoch: 20	Val Acc: 0.9704010695187187,	Val Acc EMA: 0.9705481283422485,	Avg Loss: 0.042128468448993386,	LR: 3.125e-06
Finished Training
Highest validation accuracy: 0.9705748663101622 at epoch 19
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
