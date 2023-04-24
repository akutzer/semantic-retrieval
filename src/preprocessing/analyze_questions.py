import json
from collections import defaultdict



def count_qa_pairs(data):
    n_qa_pairs = 0

    for page in data:
        parags = page["text"]
        n_qa_pairs += len(parags)

    return n_qa_pairs


def count_question_words(data):
    n_qa_pairs = count_qa_pairs(data)
    question_words_count = defaultdict(int)

    for page in data:
        parags = page["text"]
        for q, a in parags:
            q_word = q.split(" ")[0].lower()
            question_words_count[q_word] += 1
    
    question_words_count = list(sorted(question_words_count.items(), key=lambda k_v: k_v[1], reverse=True))

    for i, (q_word, count) in enumerate(question_words_count):
        q_word_perc = round(count / n_qa_pairs * 100, 3) if n_qa_pairs > 0 else 0
        question_words_count[i] = (q_word, count, q_word_perc)
    
    return n_qa_pairs, question_words_count



if __name__ == "__main__":
    path_to_dataset = "../../data/fandom-qa/harry_potter_qa.json"
    with open(path_to_dataset, mode="r") as f:
        data = json.load(f)

    n_qa_pairs, question_words_count = count_question_words(data)

    print(f"Number of QA pairs: {n_qa_pairs}")
    s = 0
    for i, (q_word, count, q_word_perc) in enumerate(question_words_count):
        print(f"'{q_word}':  {count}\t({q_word_perc}%)")
        s += q_word_perc
        if i >= 15:
            break
    print(s)
