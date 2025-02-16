import json
# question1 = How many objects are the same color as the large cylinder?
# question2 =

with open("D:\\University\\Project\\CLEVR_v1.0\\questions\\CLEVR_val_hard.json") as f:
    data = json.load(f)

questions = data["questions"]

num_questions = len(questions)

for (i,question) in enumerate(questions):
    print("Question " + str(i) + ": " + question["question"] + "\n" + "Answer: " + question["answer"] )
