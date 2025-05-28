from trainer import Trainer
from datasets import ClevrDataset, collate_fn
import argparse
from config import cfg_from_file, cfg
import os
import torch
import mac
import mac_org as mac_org
from utils import load_vocab
from tqdm import tqdm
from parser import parse_answer, parse_question


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='..//cfg//clevr_train_mac.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu', type=str, default='0')
    parser.add_argument('--set', dest='set', type=str, choices=['org', 'human', 'hard'], default='org')
    parser.add_argument('--steps', dest='steps', type=int, default=4)

    parser.add_argument('--model_path', dest='model_path', type=str, default='..//log//50_epochs_4_steps_viva//Model/model_checkpoint_000025.pth')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--wrongs_answers_path', type=str)
    args = parser.parse_args()
    return args


def validate(model_path, device, set):
    print(f"Using inference iteration: {cfg.TRAIN.MAX_STEPS}")

    val_dataset = ClevrDataset(cfg.DATASET.DATA_DIR, set, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=cfg.WORKERS, collate_fn=collate_fn)


    vocab = load_vocab(cfg)

    model,model_ema = mac.load_MAC(cfg, vocab)


    checkpoint = torch.load(model_path,weights_only=True)
    #print(f"Model trained using max steps: {checkpoint['max_steps']}")

    model.load_state_dict(checkpoint["model"])
    model.eval()
    all_accuracies = []
    all_entropies = []
    all_wrong_answers = []
    all_correct_answers = []
    for data in tqdm(iter(val_loader), desc='Validating', total=len(val_loader)):
        image, question, question_len, answer = data['image'], data['question'], data['question_length'], data['answer']
        answer = answer.long()
        image = image.to(device)
        question = question.to(device)
        answer = answer.squeeze().to(device)

        question_len = torch.tensor(question_len).to("cpu")

        with torch.no_grad():
            scores = model(image, question, question_len)

        correct = scores.detach().argmax(1) == answer
        probs = torch.softmax(scores, dim=1)
        prediction_entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=1)
        all_entropies.append(prediction_entropy.mean().cpu().numpy())
        accuracy = correct.sum().cpu().numpy() / answer.shape[0]
        all_accuracies.append(accuracy)

        if (args.wrongs_answers_path != None):
            incorrect = scores.detach().argmax(1) != answer
            wrong_answers = scores[incorrect].detach().argmax(1).to("cpu")
            wrong_questions = question[incorrect].detach().to("cpu")
            correct_answers = answer[incorrect].detach().to("cpu")

            for i in range(len(wrong_answers)):
                question_str = parse_question(wrong_questions[i].tolist())
                correct_answer_str = parse_answer(correct_answers[i].item())
                wrong_answer_str = parse_answer(wrong_answers[i].item())

                all_wrong_answers.append(wrong_answer_str)
                all_correct_answers.append(correct_answer_str)

    if (args.wrongs_answers_path != None):
        with open(f"../results/{args.wrongs_answers_path}", "w") as f:
            for wrong,correct in zip(all_wrong_answers, all_correct_answers):
                f.write(f"{wrong},{correct}\n")

    accuracy = sum(all_accuracies) / float(len(all_accuracies))
    print(f"Validation accuracy: {accuracy}")

    avg_entropy = sum(all_entropies) / float(len(all_entropies))
    print(f"Average entropy: {avg_entropy}")

    return accuracy

def model_graident_norm(model_path):
    vocab = load_vocab(cfg)
    model,model_ema = mac_org.load_MAC(cfg, vocab)

    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    model.eval()  # Make sure you're in eval mode (no dropout, etc.)

    for name, param in model.named_parameters():
        if "weight" in name and len(param.shape) >= 2:  # Only apply to 2D weights (e.g., Linear, Conv)
            try:
                spec_norm = torch.linalg.svdvals(param)[0].item()  # Largest singular value = spectral norm
                print(f"{name}: Spectral norm = {spec_norm:.4f}")
            except Exception as e:
                print(f"Could not compute spectral norm for {name}: {e}")


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu is not None :
        cfg.GPU_ID = args.gpu
    if args.steps is not None:
        cfg.TRAIN.MAX_STEPS = args.steps
    cfg.TRAIN.FLAG = False

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validate(args.model_path, device, args.set)
    # model_graident_norm(args.model_path)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # # List of spectral norms you collected
    # spectral_norms = [
    #     2.3705, 9.8882, 7.5268, 4.8669,
    #     0.4463, 4.8046, 3.8833, 3.7493
    # ]
    #
    # # Plot KDE
    # plt.figure(figsize=(8, 5))
    # sns.kdeplot(spectral_norms, fill=False, color="blue", bw_adjust=0.5)
    # plt.xlim(left=0)
    # plt.xlabel("Spectral Norm", fontsize=14)
    # plt.ylabel("Density", fontsize=14)
    # plt.grid(True)
    # plt.savefig("../results/spectral_norm_analysis.png",)
    # plt.tight_layout()
    # plt.show()




