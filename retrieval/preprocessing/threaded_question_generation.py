import re
import json
import requests
import re
import time
import os
import sys
import pandas as pd
import tqdm
import numpy as np
import threading
import ast
import random

'''
Problem: Server denies further requests if too many requests were sent. 
Possible Solution: Use Proxies
Problem: Server blocked some and some dont work
Solutions:
Threads are used for generating questions for multiple passages at once
Threads are used for trying to request answers from the servers at the same time

Problem: still slow

Script tries to ask for different question words if frequency of #what# gets too hight - does not always work
'''



# surpress warings
import warnings
warnings.filterwarnings("ignore")
import sys
sys.tracebacklimit = 0

# to get the proxy file use this command: curl https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt -o http.txt

# specify timeout interval here
TIMEOUT=10
thread_value=None
lock = threading.Lock()
lock_outter = threading.Lock()
threads_passages = []


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


# class copied and then adjusted from https://github.com/xtekky/gpt4free
class Completion:
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
        proxy_https : str = 'empty',
    ):
        # print(parentMessageId, prompt)

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
        
        limit = 1
        ii = 0
        while True:
            
            proxies= {
                    "https": proxy_https
                }
            if ii >= limit:
                return None
            try:
                ii = ii + 1
                # print(proxy_https)
                request = requests.post(url, headers=Completion.headers, json=json_data, proxies=proxies, timeout=TIMEOUT, verify=True)
                break
            except KeyboardInterrupt:
                # quit
                sys.exit()
            except Exception as e:
                pass
                # print(e)
                
                

        # ---------
        # print("success")
        content = request.content

        response = Completion.__response_to_json(content)
        # lock.acquire()
        # global thread_value
        # thread_value=response
        # lock.release()
        return response

    @classmethod
    def __response_to_json(cls, text) -> dict:
        text = str(text.decode("utf-8"))

        split_text = text.rsplit("\n", 1)[1]
        to_json = json.loads(split_text)
        return to_json
    

# function copied from aarons example script
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

# function copied from aarons example script
def generate_prompt(n_pos_questions, n_neg_questions, what=False):

    if what:
        PROMPT = f"""You are given this passage from a fandom article:
        
        %s

        Please generate {n_pos_questions} questions that are answered by the passage paragraph and {n_neg_questions} questions that are not answered by it.
        Try to use synonyms in your questions.
        The output !!MUST!! be in the following form:
        """
    else:
        PROMPT = f"""You are given this passage from a fandom article:
        
        %s

        Please generate {n_pos_questions} questions that are answered by the passage paragraph and {n_neg_questions} questions that are not answered by it.
        Try to use synonyms in your questions.
        The questions should NOT begin with what.
        The output !!MUST!! be in the following form:
        """


    for i in range(1, n_pos_questions+1):
        PROMPT += f"{i}. <answered-question-{i}>\n"

    PROMPT += "---\n"

    for i in range(1, n_neg_questions+1):
        PROMPT += f"{i}. <unanswered-question-{i}>\n"

    return PROMPT


def getResponse(df,i,j,start_ind, end_ind, what_prop=0.5, what_prop_limit=0.5):
    # print("current indices: " + str(i) +" and "+ str(j))

    if what_prop > what_prop_limit:
        PROMPT = generate_prompt(2, 2, what=False)
    else:
        PROMPT = generate_prompt(2, 2, what=True)

    skip_passage = False
    passage = df.iloc[i]["text"][j]

    global threads_passages
    try:
        
        passage_prompt = PROMPT % passage

        # usage You
        message_id=""
        threads = []
        global PROXIES

        results = []
        for https in PROXIES[start_ind:end_ind]:
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(processes=1)
            async_result = pool.apply_async(Completion.create, kwds={'prompt':passage_prompt, "parentMessageId": message_id, "proxy_https": https})
            results.append(async_result)
            time.sleep(0.4)
        
        response=None
        for result in results:
            res=result.get()
            if res != None:
                response = res
            

        if response == None:
            skip_passage = True


    #     if "Please try again." not in response:
    #         pass
            
    except Exception as e:
        # print(e)
        response = None
        pass


    global threads_passages
    global lock_outter

    if skip_passage:
        lock_outter.acquire()
        threads_passages.append(({},i,j))
        lock_outter.release()
        return
        

    positive, negative = extract_questions(response['text'])
    lock_outter.acquire()
    threads_passages.append(({
        "i": i,
        "j": j,
        "positive": positive,
        "negative": negative,
        "passage": passage
    },i,j))
    lock_outter.release()


def getDistributionQuestionWords(df):
    list_pos = df["positive"].values.tolist()
    list_neg = df['negative'].values.tolist()

    list_com = list_neg + list_pos
    list_com =  [ast.literal_eval(s) for s in list_com]
    list_com = list(np.concatenate(list_com).flat)
    q_words = [sentence.split(" ")[0].lower() for sentence in list_com if sentence]
    series =  pd.Series(q_words).value_counts(normalize=True)
    return series
    



def mainLoop(import_path, export_file,num_threads=1):
    f=import_path
    df = pd.read_json(f, orient ='records')

    pairs_ind = [ (i,j) for i in range(df.shape[0]) for j in range(len(df["text"].iloc[i]))]

    if os.path.isfile(export_file):
        df_prev = pd.read_csv(export_file)
        already_done = [ (row[0], row[1]) for row in list(df_prev.itertuples(index=False))]
    else:
        already_done = []
        df_prev = pd.DataFrame()

    pairs_ind = [pair for pair in pairs_ind if pair not in already_done]
    
    new_df = df_prev

    # print(getDistributionQuestionWords(new_df)['what'])
    

    new_passages = []
    pbar = tqdm.tqdm(total=len(pairs_ind))
    global PROXIES
    proxies_ind = random.randint(0,len(PROXIES))

    # amount of threads for different proxies
    step = num_threads

    # number of threads for different passages
    num_threads_outer=2000


    threads=[]

    # limit of what prob
    what_limit_prop = 0.25

    while pairs_ind:
        if not df_prev.empty:
            try:
                what_prop = getDistributionQuestionWords(new_df)['what']
            except:
                what_prop = 0.0
        else:
            what_prop = 0.0
        
        for _ in range(num_threads_outer):
            i,j = pairs_ind.pop(0)
            t = threading.Thread(target=getResponse, kwargs={'i':i, "j": j, "df": df, "start_ind": proxies_ind, "end_ind": proxies_ind+step, "what_prop": what_prop, "what_prop_limit":what_limit_prop})
            threads.append(t)
            t.start()
            proxies_ind = (proxies_ind + step) % len(PROXIES)
            time.sleep(0.05)
        for t in threads:
            t.join()

        global threads_passages

        # if not successful append it to list else add it to the output list
        for dict, i,j in threads_passages:
            if not dict:
                pairs_ind.append((i,j))
            else:
                new_passages.append(dict)

        threads_passages = []
        
        if(new_passages):
            new_df = pd.concat([new_df, pd.DataFrame(new_passages)], ignore_index=True)
            new_df.to_csv(export_file, index=False)

        pbar.update(len(new_passages))
        if(len(new_passages) == 0):
            time.sleep(1800)
        new_passages = []

    pbar.close()



if __name__ == "__main__":
    mainLoop(import_path="../../data/fandoms/harry_potter.json", export_file="harry_potter_questions_a.csv")
