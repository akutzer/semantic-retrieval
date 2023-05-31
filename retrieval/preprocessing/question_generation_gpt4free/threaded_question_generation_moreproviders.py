import re
import json
import requests
import time
import os
import sys
import pandas as pd
import tqdm
import numpy as np
import threading
import ast
import random
from multiprocessing.pool import ThreadPool
from gpt4free import deepai, you, usesless

'''
THIS FILE HAS TO BE PUT INTO THE CLONED GPT4FREE FOLDER TO WORK
FURTHERMORE:
- in the deepai folder in the gpt4free the init has to be changed to the init in the deepai folder in this directory



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
THREADS_DIFFERENT_PARAGRAPHS = 1000
THREADS_DIFFERENT_PROXIES_FOR_PARAGRAPH = 1
TIMEOUT=10

# global variables used by threads
lock_outter = threading.Lock()
threads_passages = []

POS, NEG = (2,2)


def getProxyList():
    proxies = []
    regex = ">([0-9]*.[0-9]*.[0-9]*.[0-9]*.[0-9]*)</td>"
    files = ['https://github.com/TheSpeedX/PROXY-List/blob/master/http.txt','https://github.com/TheSpeedX/PROXY-List/blob/master/socks4.txt', 'https://github.com/TheSpeedX/PROXY-List/blob/master/socks5.txt']
    # files = ['https://github.com/mertguvencli/http-proxy-list/blob/main/proxy-list/data.txt']

    for file in files:
        res = requests.get(file)
        proxies = proxies + re.findall(regex, res.text)

    return proxies
    

# function copied from aarons example script
def extract_questions(answer,n):
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
        if line == "---" or len(positive) == n:
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


def getResponse(df,i,j,start_ind, end_ind ,proxies, what_prop=0.5, what_prop_limit=0.5, provider = 0):
    n,m = (POS, NEG)
    # print("current indices: " + str(i) +" and "+ str(j))

    if what_prop > what_prop_limit:
        PROMPT = generate_prompt(n, m, what=False)
    else:
        PROMPT = generate_prompt(n, m, what=True)

    skip_passage = False
    passage = df.iloc[i]["text"][j]

    global threads_passages
    try:
        
        passage_prompt = PROMPT % passage

        # usage You
        results = []
        for https in proxies[start_ind:end_ind]:
            pool = ThreadPool(processes=1)
            if provider == 1:
                # you doesnt work anymore
                continue
            if provider == 0:
                async_result = pool.apply_async(deepai.Completion.create, kwds={'messages':passage_prompt, "proxy_https": https, 'timeout': TIMEOUT})
            elif provider == 1:
                async_result = pool.apply_async(you.Completion.create, kwds={'prompt':passage_prompt, "proxy": https})
            elif provider == 2:
                async_result = pool.apply_async(usesless.Completion.create, kwds={'prompt':passage_prompt, "parentMessageId": '', "proxy_https": https})

            results.append(async_result)
        time.sleep(0.4)
        
        response=None
        for result in results:
            res=result.get()
            if res != None:
                if provider == 2:
                    response = res['text']
                else:
                    response = res

            

        if response == None:
            skip_passage = True

            
    except Exception as e:
        response = None
        skip_passage = True


    global threads_passages
    global lock_outter

    if skip_passage:
        lock_outter.acquire()
        threads_passages.append(({},i,j))
        lock_outter.release()
        return
        

    positive, negative = extract_questions(response,n)

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
    



def mainLoop(import_path, export_file):
    f=import_path
    df = pd.read_json(f, orient ='records')
    # added to not process bad sentences not filtered out of preprocessing
    pairs_ind = [ (i,j) for i in range(df.shape[0]) for j in range(len(df["text"].iloc[i])) if not df["text"].iloc[i][j].endswith(' .')]

    if os.path.isfile(export_file):
        df_prev = pd.read_csv(export_file)
        already_done = [ (row[0], row[1]) for row in list(df_prev.itertuples(index=False))]
    else:
        already_done = []
        df_prev = pd.DataFrame()

    pairs_ind = [pair for pair in pairs_ind if pair not in already_done]
    
    new_df = df_prev    

    new_passages = []
    pbar = tqdm.tqdm(total=len(pairs_ind), position=1)
    proxies = getProxyList()
    proxies_ind = random.randint(0,len(proxies))

    # amount of threads for different proxies
    global THREADS_DIFFERENT_PROXIES_FOR_PARAGRAPH, THREADS_DIFFERENT_PARAGRAPHS

    step = THREADS_DIFFERENT_PROXIES_FOR_PARAGRAPH
    # number of threads for different passages
    num_threads_outer= THREADS_DIFFERENT_PARAGRAPHS

    # limit of what prob
    what_limit_prop = 0.25

    b = 0
    num_providers = 3

    while pairs_ind:

        threads=[]
        b = b + 1
        if (b % 30) == 0:
            print('now sleeping for 1000s')
            time.sleep(1000)

        if not df_prev.empty:
            try:
                what_prop = getDistributionQuestionWords(new_df)['what']
            except:
                what_prop = 0.0
        else:
            what_prop = 0.0
        
        for _ in tqdm.tqdm(range(num_threads_outer),position=0):

            if len(pairs_ind) == 0:
                break


            for provider_i in range(num_providers):
                # continue if list empty
                if not pairs_ind:
                    continue
                i,j = pairs_ind.pop(0)

                t = threading.Thread(target=getResponse, kwargs={'i':i, "j": j, "df": df, "start_ind": proxies_ind, "end_ind": proxies_ind+step, "what_prop": what_prop, "proxies": proxies, "what_prop_limit":what_limit_prop, "provider": provider_i})
                threads.append(t)
                t.start()
            time.sleep(0.04)

            proxies_ind = (proxies_ind + step) % len(proxies)
        for t in threads:
            t.join()

        global threads_passages

        # if not successful append it to list else add it to the output list
        for dict, i,j in threads_passages:
            if not dict:
                pairs_ind.append((i,j))
            else:
                if dict['positive']:
                    new_passages.append(dict)

        threads_passages = []
        
        if(new_passages):
            new_df = pd.concat([new_df, pd.DataFrame(new_passages)], ignore_index=True)
            new_df.to_csv(export_file, index=False)

        pbar.update(len(new_passages))
        # if(len(new_passages) == 0):
        #     time.sleep(1800)
        new_passages = []


    pbar.close()



if __name__ == "__main__":
    # print(deepai.Completion.create(messages='hi wie gehts', proxy_https=getProxyList()[0], timeout=TIMEOUT))


    rn = random.randint(0,2)
    rn = 2
    if rn == 0:
        print('elder')
        mainLoop(import_path="../../../../data/fandoms/elder_scrolls.json", export_file="../../elder_scrolls_qa.csv")
    elif rn == 0:
        print('star wars')
        mainLoop(import_path="../../../../data/fandoms/starwars.json", export_file="../../starwars_qa.csv")
    elif rn == 1:
        print('witcher')
        mainLoop(import_path="../../../../data/fandoms/witcher.json", export_file="../../witcher_qa.csv")
    elif rn == 2:
        print('harry')
        mainLoop(import_path="../../../../data/fandoms/harry_potter.json", export_file="../../harry_potter_question_para.csv")
    elif rn == 3:
        print('marvel')
        mainLoop(import_path="../../../../data/fandoms/marvel.json", export_file="../../marvel_qa.csv")
    # elif rn == 5:
    #     mainLoop(import_path="../../../../data/flo/harry_pqq_blueprint1_.json", export_file="../../harry_pqq_blueprint1_flo_qa.csv")


