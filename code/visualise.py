import matplotlib.pyplot as plt
import re
import argparse
from datasets import ClevrDataset, collate_fn
from config import cfg_from_file, cfg
import torch
import os
import mac as mac
import mac_org as mac_step_specific
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pickle
from parser import parse_answer

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



def get_acc_at_iteration_steps(model_path, dataset, device):
    dataset = ClevrDataset(cfg.DATASET.DATA_DIR, dataset, 'train_org_long')
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=cfg.WORKERS, collate_fn=collate_fn)

    vocab = mac.load_vocab(cfg)

    all_accuracies = []

    for steps in range(2, cfg.TRAIN.MAX_STEPS+1,2):
        print(f"Validating using inference iteration: {steps}")
        cfg.TRAIN.MAX_STEPS = steps
        model, model_ema = mac.load_MAC(cfg, vocab)
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        accuracies = []
        for data in tqdm(loader, desc='Validating', total=len(loader)):
            image, question, question_len, answer = data['image'], data['question'], data['question_length'], data['answer']
            answer = answer.long()
            image = image.to(device)
            question = question.to(device)
            answer = answer.squeeze().to(device)
            question_len = torch.tensor(question_len).to("cpu")

            with torch.no_grad():
                scores = model(image, question, question_len)

            correct = scores.detach().argmax(1) == answer
            accuracy = correct.sum().cpu().numpy() / answer.shape[0]
            accuracies.append(accuracy)

        avg_accuracy = sum(accuracies) / float(len(accuracies))
        print(f"{steps} steps model avg accuracy: {avg_accuracy} on {args.set}")
        all_accuracies.append(avg_accuracy)

    return all_accuracies



def log_results(accuracies, model_name, max_steps, file_name=None):
    if not os.path.exists(f'../results/{file_name}.txt'):
        with open(f'../results/{file_name}.txt', 'w') as f:
            f.write("")

    with open(f'../results/{file_name}.txt', 'a') as f:
        f.write(f"-----{model_name} accuracies on {args.set}-----\n")
        for (iteration, accuracy) in zip(range(2,max_steps,2),accuracies):
            f.write(f"Iteration {iteration}: {accuracy}\n")
        f.write("\n")


def plt_accuracies(acc_file):
    with open(f"../results/{acc_file}.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "-----" in line:
                model_name = line.split(" ")[0].replace("-","")
                accuracies = []
            elif "Iteration" in line:
                accuracies.append(float(line.split(" ")[-1]))
            elif line == "\n":
                if "specific" in model_name:
                    continue
                    plt.plot(range(4, cfg.TRAIN.MAX_STEPS+1, 4), accuracies, label=model_name, marker='o')
                else:
                    plt.plot(range(2, cfg.TRAIN.MAX_STEPS, 2), accuracies, label=model_name, marker='o')

    plt.axhline(y=0.8512736111594253, color='r', linestyle='--', label='4_steps_MAC')

    plt.xlabel("Inference-Time Iterations")
    plt.xticks(range(2, cfg.TRAIN.MAX_STEPS, 2))
    plt.ylabel("Accuracy")
    plt.ylim(0.3,1)
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.savefig(f'../results/long_models_results.png')
    plt.show()

def plt_confusion_matrix(answer_path):
    wrong_answers = []
    correct_answers = []
    with open("../results/" + answer_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == "":
            continue
        wrong_answer, correct_answer = line.strip().split(",")
        if not correct_answer.isdigit():
            continue
        correct_answers.append(correct_answer)
        wrong_answers.append(wrong_answer)


    classes = sorted(list(set(map(int, correct_answers + wrong_answers))))

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}


    matrix = np.zeros((len(classes),len(classes)),dtype=int)

    # Populate the matrix with mismatched results
    for true_label, pred_label in zip(correct_answers, wrong_answers):
        if true_label != pred_label:
            row = class_to_idx[int(true_label)]
            col = class_to_idx[int(pred_label)]
            matrix[row][col] += 1

    for i in range(len(classes)):
        matrix[i][i] = 900

    cmap = sns.light_palette("navy", as_cmap=True)

    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=False, cmap=cmap,  vmin=0, xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix on counting problems")
    plt.show()




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='..\\cfg\\clevr_train_mac.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu', type=str, default='0')
    parser.add_argument('--set', dest='set', type=str, choices=['org', 'human', 'hard'], default='val')
    parser.add_argument('--max_steps', dest='max_steps', type=int, default=48)
    parser.add_argument('--img_name', dest='img_name', type=str, default="human_models_results")
    parser.add_argument('--acc_file', dest='acc_file', type=str, default='accuracies_long')
    parser.add_argument('--model_path', dest='model_path', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu is not None:
        cfg.GPU_ID = args.gpu
    if args.max_steps is not None:
        cfg.TRAIN.MAX_STEPS = args.max_steps
    cfg.TRAIN.FLAG = False

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # with open("../data/val.pkl", "rb") as f:
    #     val_data = pickle.load(f)
    #
    #     count = 0
    #     identification = 0
    #     justification = 0
    #
    #     for i in range(len(val_data)):
    #         _, _, answer = val_data[i]
    #         answer = parse_answer(answer)
    #         if answer == 'yes' or answer == 'no':
    #             justification += 1
    #         elif answer.isdigit():
    #             count += 1
    #         else:
    #             identification += 1
    #     print(justification, count, identification)
    #
    #
    #
    # with open(args.wrong_answers_path, 'r') as f:
    #     lines = f.readlines()
    #
    #     count = 0
    #     identification = 0
    #     justification = 0
    #
    #     for line in lines:
    #         if line.split(",")[0] == 'yes' or line.split(",")[0] == 'no':
    #             justification += 1
    #         elif line.split(",")[0].isdigit():
    #             count += 1
    #         else:
    #             identification += 1
    #
    #     print(justification, count, identification)


    # plt_confusion_matrix("..\\results\\wrong_answers_4steps.txt")


    # train_short_model_path = '../log/50_epochs_4_steps_short/Model/model_checkpoint_000050.pth'
    # train_short_accuracies = get_acc_at_iteration_steps(train_short_model_path, args.set, device)
    # log_results(train_short_accuracies, "4_steps_short_train_DTL", cfg.TRAIN.MAX_STEPS, args.acc_file)

    # step_4_model_path = '../log/50_epochs_4_steps_final/Model/model_checkpoint_000050.pth'
    # steps_4_accuracies = get_acc_at_iteration_steps(step_4_model_path, args.set, device)
    #
    # log_results(steps_4_accuracies, "4_steps_DTL", cfg.TRAIN.MAX_STEPS)
    # plt.plot(range(2, cfg.TRAIN.MAX_STEPS+1, 2), steps_4_accuracies, label="4_steps_DTL", marker='o')
    #
    # step_8_model_path = '../log/50_epochs_8_steps_final/Model/model_checkpoint_000050.pth'
    # steps_8_accuracies = get_acc_at_iteration_steps(step_8_model_path, args.set, device)
    #
    # log_results(steps_8_accuracies, "8_steps_DTL", cfg.TRAIN.MAX_STEPS)
    # plt.plot(range(2, cfg.TRAIN.MAX_STEPS+1, 2), steps_8_accuracies, label="8_steps_DTL", marker='o')
    #
    #
    # recall_model_path = '../log/model_recall.pth'
    # recall_accuracies = get_acc_at_iteration_steps(recall_model_path, args.set, device)
    #
    # log_results(recall_accuracies, "4_steps_DT", cfg.TRAIN.MAX_STEPS)



    # recall_model_step_specific_path = '../log/50_epochs_4_steps_DTL_specific/Model/model_checkpoint_000015.pth'
    # recall_step_specific_accuracies = get_acc_at_iteration_steps(recall_model_step_specific_path, args.set, device)
    #
    # log_results(recall_step_specific_accuracies, "4_step_DTL_specific", 48, args.acc_file)

    # plt_accuracies("accuracies_long")


    with open("../results/MAC-DT_results.txt", "r") as f:
        model_name = "4_steps_recall"
        lines = f.readlines()
        accuracies = []
        for line in lines:
            accuracies.append(float(line.split(" ")[-1]))

    plt.plot(range(2, 48, 2), accuracies, label=model_name, marker='o')
    plt.axhline(y=0.9766855460381485, color='r', linestyle='--', label='4_steps_MAC')

    plt.xlabel("Inference-Time Iterations")
    plt.xticks(range(4, 50, 4))
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f'../results/MAC_vs_MACDT.png')
    plt.show()



