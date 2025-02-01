import argparse
import pickle

import os
import nltk
import numpy as np
from PIL import Image
import mac
import torch.nn as nn
from utils import load_vocab
from config import cfg, cfg_from_file
from torchvision import transforms, models
import torch
from parser import parse_answer, get_answer_dict

class Resnet(nn.Module):
    def __init__(self):
        if not hasattr(models, "resnet101"):
            raise ValueError('Invalid model"')

        super(Resnet, self).__init__()
        weights = models.ResNet101_Weights.DEFAULT
        self.cnn = getattr(models, "resnet101")(weights=weights)
        self.layers = [
            self.cnn.conv1,
            self.cnn.bn1,
            self.cnn.relu,
            self.cnn.maxpool,
        ]
        for i in range(3):
            name = 'layer%d' % (i + 1)
            self.layers.append(getattr(self.cnn, name))
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='..\\cfg\\clevr_train_mac.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='D:\\University\\Project\\CLEVR_v1.0')
    parser.add_argument('--image_file', type=str, help='image file path', default='D:\\University\\Project\\CLEVR_v1.0\\images\\val\\CLEVR_val_000012.png')
    parser.add_argument('--question', type=str, help='question string', default='What is the color of the cube?')
    args = parser.parse_args()
    return args

def preprocess_question(question):
    with open("../data/dic.pkl", "rb") as f:
        dictionaries = pickle.load(f)
        word_dict = dictionaries["word_dic"]

    question_token = []
    words = nltk.word_tokenize(question)
    for word in words:
        try:
            question_token.append(word_dict[word])
        except:
            print(f"Word \"{word}\" not found in question word dictionary")
            exit()

    return torch.tensor(question_token)

def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to('cuda')

    resnet = Resnet().to('cuda').eval()
    return resnet(image)


def inference(image_path, question, device):
    image = preprocess_image(image_path).to(device)
    question = preprocess_question(question).unsqueeze(0).to(device)
    question_length = torch.tensor(len(question[0])).unsqueeze(0).to(device)

    cfg_from_file(args.cfg_file)
    cfg.DATA_DIR = args.data_dir

    vocab = load_vocab(cfg)
    model, model_ema = mac.load_MAC(cfg, vocab)
    checkpoint = torch.load(os.path.join(os.pardir, "log", "model(2.1).pth"), weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        prediction = model(image, question, question_length)
        result = prediction.argmax(1).item()

    answer_dict = get_answer_dict()
    predicts = {k: v for (k, v) in zip(answer_dict.keys(), prediction.squeeze(0).tolist())}
    print(predicts)
    print(parse_answer(result))


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()

    image_path = args.image_file
    question = args.question

    inference(image_path, question, device)




