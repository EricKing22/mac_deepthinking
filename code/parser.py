
import pickle


# Parse the question token back into question string
# question_token: list of question word index
def parse_question(question_token):
    with open("../data/dic.pkl", "rb") as f:
        dictionaries = pickle.load(f)
        word_dict = dictionaries["word_dic"]
        reverse_word_dict = {v: k for k, v in word_dict.items()}

    question = ""
    for token in question_token:
        question += reverse_word_dict[token] + " "

    return question.strip()

# Parse the answer token back into answer string
# answer_token: answer index
def parse_answer(answer_token):
    with open("../data/dic.pkl", "rb") as f:
        dictionaries = pickle.load(f)
        answer_dict = dictionaries["answer_dic"]
        reverse_answer_dict = {v: k for k, v in answer_dict.items()}

    return reverse_answer_dict[answer_token]

def get_answer_dict():
    with open("../data/dic.pkl", "rb") as f:
        dictionaries = pickle.load(f)
        answer_dict = dictionaries["answer_dic"]

    return answer_dict

if __name__ == "__main__":
    with open("../data/train.pkl", "rb") as f:
        data = pickle.load(f)
        img_file, example_question, example_answer = data[0]



    print(img_file)
    print(parse_question(example_question))
    print(parse_answer(example_answer))