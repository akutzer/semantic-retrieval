import re
import random
import json
import requests
import re
import time
import os
import sys
import pandas as pd


def getProxyList():
    # opening the file in read mode
    my_file = open("http.txt", "r")
    
    # reading the file
    data = my_file.read()
    
    # replacing end splitting the text 
    # when newline ('\n') is seen.
    data_into_list = data.split("\n")
    
    my_file.close()

    return data_into_list

PROXIES = getProxyList()
PROXIES_TRIED = ['not tried']*len(PROXIES)
current_ind = -1


def setFalseProxies(ind:int):
    global PROXIES_TRIED
    PROXIES_TRIED[ind] = 'doesnt work'

def setTrueProxies(ind:int):
    global PROXIES_TRIED
    PROXIES_TRIED[ind] = 'works'

def getProxy():
    global current_ind
    global PROXIES_TRIED
    global PROXIES
    if "works" in PROXIES_TRIED:
        index = PROXIES_TRIED.index("works")
        return PROXIES[index]
    
    possible_ind = [i for i in range(len(PROXIES_TRIED)) if PROXIES_TRIED[i] != 'doesnt work']
    ind = possible_ind[random.randint(0,len(possible_ind) - 1)]
    proxy = PROXIES[ind]
    
    current_ind = ind

    return proxy



class Completion:
    global current_ind
    global PROXIES
    headers = {
        "authority": "ai.usesless.com",
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.5",
        "cache-control": "no-cache",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0",
    }

    @staticmethod
    def create(
        systemMessage: str = "You are a helpful assistant",
        prompt: str = "",
        parentMessageId: str = "",
        presence_penalty: float = 1,
        temperature: float = 1,
        model: str = "gpt-3.5-turbo",
    ):
        print(parentMessageId, prompt)

        json_data = {
            "openaiKey": "",
            "prompt": prompt,
            "options": {
                "parentMessageId": parentMessageId,
                "systemMessage": systemMessage,
                "completionParams": {
                    "presence_penalty": presence_penalty,
                    "temperature": temperature,
                    "model": model,
                },
            },
        }



        url = "https://ai.usesless.com/api/chat-process"

        # added 11.05
        

        while True:
            https = getProxy()
            print(https)

            proxies= {
                    "https": https
                    #'https': PROXIES[2416]
                }
            try:
                request = requests.post(url, headers=Completion.headers, json=json_data, proxies=proxies, timeout=12)
                break
            except KeyboardInterrupt:
                # quit
                sys.exit()
            except:
                setFalseProxies(current_ind)
                print("timeout")
                

        # ---------
        content = request.content

        response = Completion.__response_to_json(content)
        return response

    @classmethod
    def __response_to_json(cls, text) -> dict:
        text = str(text.decode("utf-8"))

        split_text = text.rsplit("\n", 1)[1]
        to_json = json.loads(split_text)
        return to_json


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




def mainLoop(import_path, export_file):
    global current_ind

    '''
    if not os.path.isfile(export_file):
        with open(export_file, "w") as text_file:
            text_file.write(",i,j,positive,negative,passage")
    '''

    f=import_path
    df = pd.read_json(f, orient ='records')


    pairs_ind = [ (i,j) for i in range(df.shape[0]) for j in range(len(df["text"].iloc[i]))]

    if os.path.isfile(export_file):
        df_prev = pd.read_csv(export_file)
        already_done = [ (row[1], row[2]) for row in list(df_prev.itertuples(index=False))]
    else:
        already_done = []
        df_prev = pd.DataFrame()

    pairs_ind = [pair for pair in pairs_ind if pair not in already_done]

    
    new_df = df_prev

        
    PROMPT = generate_prompt(2, 2)
    new_passages = []

    while pairs_ind:
        i,j = pairs_ind.pop(0)

        retry_count = 0
        skip_passage = False
        passage = df.iloc[i]["text"][j]

        while True:
            try:
                passage_prompt = PROMPT % passage

                # usage You
                message_id=""
                response = Completion.create(prompt=passage_prompt, parentMessageId=message_id)
  
                if "Please try again." not in response:
                    break
            except Exception as e:
                time.sleep(3)
                print(e)
            
            retry_count += 1
            if retry_count > 3:
                skip_passage = True
                setFalseProxies(current_ind)
                break  

            # sleep for a few seconds before requesting again
            time.sleep(3)
        

        i = i + 1
        if skip_passage:
            pairs_ind.append((i,j))
            continue

        setTrueProxies(current_ind)
        
        positive, negative = extract_questions(response['text'])
        new_passages.append({
            "i": i,
            "j": j,
            "positive": positive,
            "negative": negative,
            "passage": passage
        })
        
        if(new_passages):
            new_df = pd.concat([new_df, pd.DataFrame(new_passages)], ignore_index=True)
            new_df.to_csv(export_file, index=False)

        new_passages = []

    new_df



if __name__ == "__main__":
    mainLoop(import_path="../../data/fandoms/harry_potter.json", export_file="harry_potter.csv")
