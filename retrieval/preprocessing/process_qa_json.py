#!/usr/bin/env python3
import glob
import os
from tqdm import tqdm
import pandas as pd
import json
import shutil

preprocessed_qa_json_path_dir = '../../data/fandoms_qa/'
output_path_dir = "../../data/fandoms_qa/"

# if true take all wikis
COMPLETE = True

# if true split into train, test, val
ALL = True

# fractions of ['train', 'test', 'val']
FRACS = [0.8, 0.1, 0.1]

# # endings of individual files
# # endings = [".train", ".test", ".val"]

# # if no file endings for individual files wanted:
# endings = ['']*len(fracs)

# # folder names for each project
# endings_dir = ['train', 'test', 'val']



if __name__ == "__main__":
    if ALL:
        fracs = [1.0]
        endings_dir = ['all']
    else:
        fracs = FRACS
        endings_dir = ['train', 'test', 'val']
        # endings = [".train", ".test", ".val"]

    endings = ['']*len(fracs)


    if COMPLETE:
        dft = pd.DataFrame()
        for file in os.listdir(preprocessed_qa_json_path_dir):
            if file.endswith(".json"):
                dfnew = pd.read_json(preprocessed_qa_json_path_dir + file, orient ='records')
                dft = pd.concat([dft, dfnew], ignore_index=True)
        preprocessed_qa_json_path_dir = preprocessed_qa_json_path_dir+ 'temp/'
        os.makedirs(preprocessed_qa_json_path_dir, exist_ok=True)
        dft.to_json(preprocessed_qa_json_path_dir+ "complete_qa"+".json", orient='records', indent=4)
        

    for file in os.listdir(preprocessed_qa_json_path_dir):
        if file.endswith(".json"):
            current_frac = 0
            f = file

            df2 = pd.read_json(preprocessed_qa_json_path_dir + f, orient ='records')

            if COMPLETE:
                shutil.rmtree(preprocessed_qa_json_path_dir)

            df2.columns

            # %%
            queries = []
            qids = []
            pids = []
            pid_wid = []
            passages = []
            qid_id = 0
            pid_id = 0
            triples = []
            wikis = []
            inconsistent = 0
            exclude = 0
            total = 0
            last_ind = 0

            df2 = df2.sample(frac=1)

            total_p = len([(i,j,k) for i in tqdm(range(len(df2))) for j in range(len(df2.iloc[i]['text'])) for k in range(min(len(df2.iloc[i]['positive'][j]), len(df2.iloc[i]['negative'][j]))) if not df2.iloc[i]['text'][j].endswith(' .')])

            ii = 0
            for i in tqdm(range(len(df2))):
                row = df2.iloc[i]
                wiki_pids = []

                for j in range(len(row['text'])):
                    total = total + 1
                    # added to exclude wrong sentences of wikiextractor that were not filtered out during preprocessing
                    if row['text'][j].endswith(' .'):
                        exclude = exclude + 1
                        continue

                    pids.append(pid_id)
                    passages.append(row['text'][j])
                    pid_wid.append(row['id'])
                    wiki_pids.append(pid_id)

                    pid_id = pid_id + 1


                    if len(row['positive'][j]) != len(row['negative'][j]):
                        inconsistent = inconsistent + 1


                    for k in range(min(len(row['positive'][j]), len(row['negative'][j]))):
                        ii = ii + 1
                        # added to not include passages where information was lost due to wikiextractor
                        queries.append(row['positive'][j][k])
                        qids.append(qid_id)
                        qid_id = qid_id + 1


                        queries.append(row['negative'][j][k])
                        qids.append(qid_id)
                        qid_id = qid_id + 1
                        triples.append(( qid_id - 2, qid_id - 1,pid_id - 1))

                wikis.append(wiki_pids)

                if 1.0*ii/total_p >= sum(fracs[:current_frac + 1]):
                    if current_frac != len(fracs):
                        out_dir = output_path_dir + file.split('/')[-1][:-8] + '/' + endings_dir[current_frac] + '/'

                        os.makedirs(out_dir,exist_ok=True)
                    else:
                        continue


                    pid_p_df = pd.DataFrame(zip(pids,passages, pid_wid), columns=['PID', 'passage', 'WID'])
                    qid_q_df = pd.DataFrame(zip(qids,queries), columns=['QID', 'query'])
                    triples_df = pd.DataFrame(triples, columns=['QID+', 'QID-', 'PID'])

                    queries = []
                    qids = []
                    pids = []
                    pid_wid = []
                    passages = []
                    qid_id = 0
                    pid_id = 0
                    triples = []
   
                    df2n = df2.iloc[last_ind:(i + 1)]
                    df2n = df2n.drop(['text', 'positive', 'negative'], axis=1)

                    df2n['PIDs'] = wikis

                    last_ind = i + 1

                    output_path = out_dir

                    triples_df.to_csv(output_path + 'triples' + endings[current_frac]+ '.tsv', index=False, sep="\t")
                    pid_p_df.to_csv(output_path +  'passages' + endings[current_frac] +'.tsv', index=False, sep="\t")
                    qid_q_df.to_csv(output_path + 'queries' + endings[current_frac] +'.tsv', index=False, sep="\t")



                    data = df2n.to_dict('index')

                    # convert the dictionary to the desired JSON format
                    output = {}
                    for key, value in data.items():
                        output[int(value['id'])] = {
                            'revid': value['revid'],
                            'url': value['url'],
                            'title': value['title'],
                            'PIDs': value['PIDs']
                        }


                    # convert the output dictionary to JSON
                    # json_output = json.dumps(output)
                    with open(output_path + 'wiki'+ endings[current_frac] +'.json', 'w') as f:
                        json.dump(output, f, indent=4)
                    
                    wikis = []
                    
                    current_frac = current_frac + 1
                

            print('inconsistent: ' + str(inconsistent))
            print('excluded: ' + str(1.0*exclude/total))
