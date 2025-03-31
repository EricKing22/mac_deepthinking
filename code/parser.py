
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
        if token == 0:
            continue
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
    with open("../data/train_long.pkl", "rb") as f:
        data = pickle.load(f)
        img_file, question_token, answer_token = data[0]

    print(parse_question(question_token))
    print(parse_answer(answer_token))
    # with open("../data/train.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     print(f"There are {len(data)} questions in the training set.")
    #
    # short_questions = []
    # long_questions = []
    # avg_question_length = 19
    #
    # for img_file, question, answer in data:
    #     question_length = len(question)
    #     if question_length <= avg_question_length:
    #         short_questions.append((img_file, question, answer))
    #     elif question_length > avg_question_length:
    #         long_questions.append((img_file, question, answer))
    #
    # with open("../data/train_short.pkl", "wb") as f:
    #     pickle.dump(short_questions, f)
    #
    # with open("../data/train_long.pkl", "wb") as f:
    #     pickle.dump(long_questions, f)
    #
    # print(f"There are {len(short_questions)} short questions and {len(long_questions)} long questions in the training set.")



