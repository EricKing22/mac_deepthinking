import os
import sys
import json
import pickle

import nltk
from tqdm import tqdm
import argparse
from PIL import Image

def process_question(root, split, word_dic=None, answer_dic=None, set='org'):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    if set == 'human':
        with open(os.path.join(root, 'questions', f'CLEVR-Humans-{split}.json')) as f:
            data = json.load(f)
    elif set == 'hard':
        with open(os.path.join(root, 'questions', f'CLEVR_{split}_hard.json')) as f:
            data = json.load(f)
    elif set == 'org':
        with open(os.path.join(root, 'questions', f'CLEVR_{split}_questions.json')) as f:
            data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0

    #  jason example:
    #  {"image_index": 14999,
    #  "split": "test",
    #  "image_filename": "CLEVR_test_014999.png",
    #  "question_index": 149985,
    #  "question": "What number of objects are gray spheres or gray things on the right side of the small gray ball?"}

    # pickle file storage format:
    # image_filename::str; the name of the image file
    # question::[int]; a list of words index from word_dic
    # answer:: int; an index from answer_dic. All possible answers are stored in answer_dic.
    for question in tqdm(data['questions'], desc=f'Processing {split} dataset', total=len(data['questions'])):
        # tokenize the string into a list of words [word]
        words = nltk.word_tokenize(question['question'])

        question_token = []

        for word in words:
            # store every question word as index
            try:
                question_token.append(word_dic[word])
            # add word to word_dic if it is not in the dictionary
            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer'] #  a sentence of answer, don't vary much

        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((question['image_filename'], question_token, answer))

    #  pickle file to store python objects (list: result) in binary format
    with open(f'../data/{split}_{set}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument('-d', '--data', default='D:\\University\\Project\\CLEVR_v1.0')
    parser.add_argument('-set', '--set', default='org', choices=['org', 'hard','human'])
    parser.add_argument('-split', '--split', default='train')
    args = parser.parse_args()

    root = args.data
    if args.set == 'org' and not os.path.exists(os.path.join(os.pardir, "data", "dic.pkl")):
        word_dic, answer_dic = process_question(root, 'train')
        process_question(root, args.split, word_dic, answer_dic)

        with open('../data/dic.pkl', 'wb') as f:
            pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)

    # using the preprocessed dictionary obtained from original CLEVR dataset to process human / hard dataset
    else:
        dicts = pickle.load(open(os.path.join(os.pardir, "data", "dic.pkl"), "rb"))
        word_dict, answer_dict = dicts["word_dic"], dicts["answer_dic"]

        process_question(root, args.split, word_dict, answer_dict, set=args.set)

