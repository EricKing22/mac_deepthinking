from trainer import Trainer
from datasets import ClevrDataset, collate_fn
import argparse
from config import cfg_from_file, cfg
import os
import torch
#import mac
import mac_org as mac
from utils import load_vocab
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='..\\cfg\\clevr_train_mac.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu', type=str, default='0')
    parser.add_argument('--set', dest='set', type=str, choices=['org', 'human', 'hard'], default='org')
    parser.add_argument('--steps', dest='steps', type=int, default=4)

    parser.add_argument('--model_path', dest='model_path', type=str, default='..\\log\\model_recall.pth')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def validate(model_path, device, set):
    val_dataset = ClevrDataset(cfg.DATASET.DATA_DIR, set, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=cfg.WORKERS, collate_fn=collate_fn)


    vocab = load_vocab(cfg)
    model,model_ema = mac.load_MAC(cfg, vocab)
    checkpoint = torch.load(model_path,weights_only=True)
    #print(f"Model trained using max steps: {checkpoint['max_steps']}")
    print(f"Using inference iteration: {cfg.TRAIN.MAX_STEPS}")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    all_accuracies = []

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
        accuracy = correct.sum().cpu().numpy() / answer.shape[0]
        all_accuracies.append(accuracy)

    accuracy = sum(all_accuracies) / float(len(all_accuracies))
    print(f"Validation accuracy: {accuracy}")

    return accuracy




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




