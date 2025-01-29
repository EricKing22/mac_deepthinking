import matplotlib.pyplot as plt
import re

# Raw training data
data = """
Epoch: 1	Val Acc: 0.630494652406417,	Val Acc EMA: 0.6418315508021395,	Avg Loss: 0.772084879410779,	LR: 0.0001
Epoch: 2	Val Acc: 0.8481951871657764,	Val Acc EMA: 0.844131016042782,	Avg Loss: 0.42099609792931003,	LR: 0.0001
Epoch: 3	Val Acc: 0.9069786096256681,	Val Acc EMA: 0.9223796791443852,	Avg Loss: 0.27816923506504126,	LR: 0.0001
Epoch: 4	Val Acc: 0.9304545454545462,	Val Acc EMA: 0.9392513368983959,	Avg Loss: 0.21059903638293706,	LR: 0.0001
Epoch: 5	Val Acc: 0.9376069518716568,	Val Acc EMA: 0.9477941176470582,	Avg Loss: 0.181797252657675,	LR: 0.0001
Epoch: 6	Val Acc: 0.9429411764705865,	Val Acc EMA: 0.9537967914438494,	Avg Loss: 0.1678105429612196,	LR: 0.0001
Epoch: 7	Val Acc: 0.9459759358288762,	Val Acc EMA: 0.9596390374331543,	Avg Loss: 0.14616353588887288,	LR: 0.0001
Epoch: 8	Val Acc: 0.9561898395721917,	Val Acc EMA: 0.9658155080213905,	Avg Loss: 0.1404393012782116,	LR: 0.0001
Reduced learning rate to 5e-05
Epoch: 9	Val Acc: 0.9592245989304805,	Val Acc EMA: 0.9689037433155084,	Avg Loss: 0.11993453916235228,	LR: 5e-05
Epoch: 10	Val Acc: 0.9691577540106973,	Val Acc EMA: 0.9724331550802152,	Avg Loss: 0.08540980200829049,	LR: 5e-05
Reduced learning rate to 2.5e-05
Epoch: 11	Val Acc: 0.9704010695187181,	Val Acc EMA: 0.9743048128342269,	Avg Loss: 0.07749136075929129,	LR: 2.5e-05
Epoch: 12	Val Acc: 0.973836898395724,	Val Acc EMA: 0.9752272727272752,	Avg Loss: 0.060926813864805474,	LR: 2.5e-05
Reduced learning rate to 1.25e-05
Epoch: 13	Val Acc: 0.974505347593585,	Val Acc EMA: 0.9760294117647084,	Avg Loss: 0.05899115668881989,	LR: 1.25e-05
Epoch: 14	Val Acc: 0.9756417112299488,	Val Acc EMA: 0.9765508021390399,	Avg Loss: 0.04328173728871014,	LR: 1.25e-05
Reduced learning rate to 6.25e-06
Epoch: 15	Val Acc: 0.9761096256684523,	Val Acc EMA: 0.9768315508021416,	Avg Loss: 0.04773234846936665,	LR: 6.25e-06
Epoch: 16	Val Acc: 0.977125668449201,	Val Acc EMA: 0.9770053475935857,	Avg Loss: 0.035732008429178584,	LR: 6.25e-06
Reduced learning rate to 3.125e-06
Epoch: 17	Val Acc: 0.9770053475935858,	Val Acc EMA: 0.9772192513369017,	Avg Loss: 0.037942524947945706,	LR: 3.125e-06
Epoch: 18	Val Acc: 0.977232620320859,	Val Acc EMA: 0.9775267379679174,	Avg Loss: 0.03537945682287476,	LR: 3.125e-06
Epoch: 19	Val Acc: 0.977352941176474,	Val Acc EMA: 0.9774598930481312,	Avg Loss: 0.034964473124706186,	LR: 3.125e-06
Epoch: 20	Val Acc: 0.9777272727272764,	Val Acc EMA: 0.9778074866310197,	Avg Loss: 0.037622775481668654,	LR: 3.125e-06
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
plt.savefig('../results/plot.png')
plt.show()
