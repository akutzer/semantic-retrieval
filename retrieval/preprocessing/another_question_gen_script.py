import gpt4free
from gpt4free import Provider, quora, forefront
import time
import json
import re
from tqdm import tqdm



def create_token(attempts=5, provider=forefront):
    token = False
    for _ in range(attempts):
        try:
            print("Creating token")
            token = provider.Account.create(logging=False)
            break
        except Exception as e:
            print(e)
    
    return token


def extract_questions(answer):
    positive, negative = [], [] 
    all_positive = False

    question_pattern = r"^[123]\.\s?(?P<question>.*?)$"
    reg_ex = re.compile(question_pattern)

    for line in answer.split("\n"):
        re_match = reg_ex.match(line)
        if re_match:
            question = re_match.group("question")
            if not all_positive:
                positive.append(question)
            else:
                negative.append(question)
        if line == "---" or len(positive) == 3:
            all_positive = True
    
    return positive, negative


def generate_prompt(n_pos_questions, n_neg_questions):
    PROMPT = f"""You are given this passage from a fandom article:
    
%s

Please generate {n_pos_questions} questions that are answered by the passage paragraph and {n_neg_questions} questions that are not answered by it.
The output !!MUST!! be in the following form:
"""

    for i in range(1, n_pos_questions+1):
        PROMPT += f"{i}. <answered-question-{i}>\n"

    PROMPT += "---\n"

    for i in range(1, n_neg_questions+1):
        PROMPT += f"{i}. <unanswered-question-{i}>\n"

    return PROMPT




if __name__ == "__main__":
    with open("witcher.json", mode="r", encoding="utf-8") as f:
        wiki = json.load(f)
    
    PROMPT = generate_prompt(2, 2)
    qa_wiki = []

    for i, page in enumerate(tqdm(wiki)):
        print(f"Quering {page['title']} article.")
        passages = page["text"]
        new_passages = []

        for j, passage in enumerate(passages):
            retry_count = 0
            skip_passage = False


            while True:
                try:
                    print(f"({j+1}/{len(passages)}) Awaiting response")

                    passage_prompt = PROMPT % passage

                    # usage You
                    response = gpt4free.Completion.create(
                        Provider.You, prompt=passage_prompt
                        )

                    # usage Poe
                    # token = quora.Account.create(logging=False)
                    # response = gpt4free.Completion.create(
                    #     Provider.Poe, prompt=passage_prompt, token=token, model="ChatGPT"
                    # )

                    # usage forefront
                    # token = forefront.Account.create(logging=False)
                    # response = gpt4free.Completion.create(
                    #     Provider.ForeFront, prompt=passage_prompt, model="gpt-4", token=token
                    # )

                    # usage theb
                    # response = gpt4free.Completion.create(
                    #     Provider.Theb, prompt=passage_prompt
                    # )

                    # usage cocalc
                    # response = gpt4free.Completion.create(
                    #     Provider.CoCalc, prompt=passage_prompt, cookie_input=""
                    # )



                    if "Please try again." not in response:
                        break
                except Exception as e:
                    print(e)
                
                retry_count += 1
                if retry_count > 3:
                    skip_passage = True
                    break  

                # sleep for a few seconds before requesting again
                time.sleep(3)
            
            if skip_passage:
                continue
            
            positive, negative = extract_questions(response)
            new_passages.append({
                "positive": positive,
                "negative": negative,
                "passage": passage
            })
                    
        wiki[i]["text"] = new_passages
        qa_wiki.append(wiki[i])

        # dump the data to make sure nothing is lost, if the script fails
        with open("witcher_qa.json", mode="w", encoding="utf-8") as f:
            json.dump(qa_wiki, f, indent=1)

    with open("witcher_qa.json", mode="w", encoding="utf-8") as f:
        json.dump(qa_wiki, f, indent=1)
