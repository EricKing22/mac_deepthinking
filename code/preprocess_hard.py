import argparse
import json
from collections import defaultdict, Counter
import os
import pickle
import tqdm
import nltk



def process_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(os.path.join(root, 'questions', f'CLEVR_{split}_hard.json')) as f:
        data = json.load(f)

    result = []

    unseen = set()

    for question in tqdm.tqdm(data['questions'], desc=f'Processing {split} dataset', total=len(data['questions'])):
        # tokenize the string into a list of words [word]
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])
            except:
                unseen.add(word)
                continue

        answer_word = question['answer'] #  a sentence of answer, don't vary much

        try:
            answer = answer_dic[answer_word]
        except:
            print(f"Answer: \"{answer_word}\" not found in ans dictionary")
            continue


        result.append((question['image_filename'], question_token, answer))

    #  pickle file to store python objects (list: result) in binary format
    with open(f'../data/{split}_hard.pkl', 'wb') as f:
        pickle.dump(result, f)



    return word_dic, answer_dic

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="D:\\University\\Project\\CLEVR_v1.0")
    parser.add_argument("-s", "--split", type=str, default="val")
    args = parser.parse_args()


    dicts = pickle.load(open(os.path.join(os.pardir,"data","dic.pkl"), "rb"))
    word_dict, answer_dict = dicts["word_dic"], dicts["answer_dic"]



    process_question(args.data_dir, args.split, word_dict, answer_dict)