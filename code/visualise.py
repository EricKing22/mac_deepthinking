import matplotlib.pyplot as plt
import re
import argparse
from datasets import ClevrDataset, collate_fn
from config import cfg_from_file, cfg
import torch
import os
import mac_org as mac
from tqdm import tqdm

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
    dataset = ClevrDataset(cfg.DATASET.DATA_DIR, dataset, 'val')
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
        accuries = []
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
            accuries.append(accuracy)

        avg_accuracy = sum(accuries) / float(len(accuries))
        print(f"{steps} steps model avg accuracy: {avg_accuracy} on {args.set}")
        all_accuracies.append(avg_accuracy)

    return all_accuracies



def log_results(accuracies, model_name, max_steps):
    if not os.path.exists(f'../results/{args.acc_file}.txt'):
        with open(f'../results/{args.acc_file}.txt', 'w') as f:
            f.write("")

    with open(f'../results/{args.acc_file}.txt', 'a') as f:
        f.write(f"-----{model_name} accuracies on {args.set}-----\n")
        for (iteration, accuracy) in zip(range(2,max_steps,2),accuracies):
            f.write(f"Iteration {iteration}: {accuracy}\n")
        f.write("\n")


def plt_accuracies():
    with open(f"../results/{args.acc_file}.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "-----" in line:
                model_name = line.split(" ")[0].replace("-","")
                accuracies = []
            elif "Iteration" in line:
                accuracies.append(float(line.split(" ")[-1]))
            elif line == "\n":
                plt.plot(range(2, cfg.TRAIN.MAX_STEPS, 2), accuracies, label=model_name, marker='o')

    # plt.axhline(y=0.9766855460381485, color='r', linestyle='--', label='4_step_recall+specific')

    plt.xlabel("Inference-Time Iterations")
    plt.xticks(range(2, cfg.TRAIN.MAX_STEPS, 2))
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.savefig(f'../results/{args.img_name}.png')
    plt.show()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='..\\cfg\\clevr_train_mac.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu', type=str, default='0')
    parser.add_argument('--set', dest='set', type=str, choices=['org', 'human', 'hard'], default='human')
    parser.add_argument('--max_steps', dest='max_steps', type=int, default=48)
    parser.add_argument('--img_name', dest='img_name', type=str, default="human_models_results")
    parser.add_argument('--acc_file', dest='acc_file', type=str, default='accuracies_human')
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



    # step_4_model_path = '../log/50_epochs_4_steps_final/Model/model_checkpoint_000050.pth'
    # steps_4_accuracies = plot_iteration_steps(step_4_model_path, args.set, device)
    #
    # log_results(steps_4_accuracies, "4_steps_DTL", cfg.TRAIN.MAX_STEPS)
    #
    # plt.plot(range(2, cfg.TRAIN.MAX_STEPS+1, 2), steps_4_accuracies, label="4_steps_DTL", marker='o')
    #
    # step_8_model_path = '../log/50_epochs_8_steps_final/Model/model_checkpoint_000050.pth'
    # steps_8_accuracies = plot_iteration_steps(step_8_model_path, args.set, device)
    #
    # plt.plot(range(2, cfg.TRAIN.MAX_STEPS+1, 2), steps_8_accuracies, label="8_steps_model", marker='o')
    #
    # log_results(steps_8_accuracies, "8_steps_DTL", cfg.TRAIN.MAX_STEPS)
    #
    # recall_model_path = '../log/model_recall.pth'
    # recall_accuracies = get_acc_at_iteration_steps(recall_model_path, args.set, device)
    #
    # log_results(recall_accuracies, "4_steps_recall", cfg.TRAIN.MAX_STEPS)

    plt_accuracies()
    #
    # recall_model_step_specific_path = '../log/model_recall_step_specific.pth'
    # recall_step_specific_accuracies = get_acc_at_iteration_steps(recall_model_step_specific_path, args.set, device)
    #
    # log_results(recall_step_specific_accuracies, "recall_step_specific_model", 2)

