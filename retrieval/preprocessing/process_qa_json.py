#!/usr/bin/env python3
import glob
import os
from tqdm import tqdm
import pandas as pd
import json
import shutil
import re

preprocessed_qa_json_path_dir = '../../data/fandoms_qa/'
output_path_dir = "../../data/fandoms_qa/"


def transform_triple(df, len_p,len_q):
    df['QID+'] = df['QID+'].apply(lambda row : row + len_q)
    df['QID-'] = df['QID-'].apply(lambda row : row + len_q)
    df['PID'] = df['PID'].apply(lambda row : row + len_p)
    return df

def transform_wiki(json_dic, len_p):
    json_dic_ret = json_dic
    for key, value in json_dic.items():
        for key2, _ in value.items():
            if key2 == 'PIDs':
                pids = [pid + len_p for pid in json_dic[key][key2]]
                json_dic_ret[key][key2] = pids
    return json_dic_ret

# if true take all wikis
#COMPLETE = False

# if false split into train, test, val
#ALL = False

# fractions of ['train', 'test', 'val']
FRACS = [0.8, 0.1, 0.1]

# # endings of individual files
# # endings = [".train", ".test", ".val"]

# # if no file endings for individual files wanted:
# endings = ['']*len(fracs)

# # folder names for each project
# endings_dir = ['train', 'test', 'val']

REGEX_STRING = r'(.*\?).*'

if __name__ == "__main__":
    for ALL, COMPLETE in [(False, False), (True, False), (True, True)]:
    #for ALL, COMPLETE in [(True, True)]:
        print((ALL,COMPLETE))
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
            dft.to_json(preprocessed_qa_json_path_dir+ "fandoms_all_qa"+".json", orient='records', indent=4)
            

        for file in os.listdir(preprocessed_qa_json_path_dir):
            if file.endswith(".json"):
                current_frac = 0
                f = file

                df2 = pd.read_json(preprocessed_qa_json_path_dir + f, orient ='records')



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

                df2 = df2.sample(frac=1, random_state=42)

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
                            if re.findall(REGEX_STRING,row['positive'][j][k]) and re.findall(REGEX_STRING,row['negative'][j][k]):
                                # added to not include passages where information was lost due to wikiextractor
                                queries.append(row['positive'][j][k])
                                qids.append(qid_id)
                                qid_id = qid_id + 1


                                queries.append(row['negative'][j][k])
                                qids.append(qid_id)
                                qid_id = qid_id + 1
                                triples.append(( qid_id - 2, qid_id - 1,pid_id - 1))
                            else: 
                                exclude = exclude + 1
                            ii = ii + 1

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
                if COMPLETE:
                    shutil.rmtree(preprocessed_qa_json_path_dir)
        
        '''


        j\i        train test eval 
    passages
    queries
    triples
    '''
    # wikis        train test eval 

    train_test_eval_df = [[pd.DataFrame()]*3 for j in range(4)]
    wikis = [{}]*3


    for root, dirs, files in os.walk(output_path_dir):
        for file in sorted(files)[::-1]:
            path = os.path.join(root, file)

            if path.endswith('train/passages.tsv'):
                i, j = (0,0)
                name = 'PID'
                df = pd.read_csv(path, sep ='\t')
                df[name] = df[name].apply(lambda row : row + len(train_test_eval_df[i][j]))
                train_test_eval_df[i][j] = pd.concat([train_test_eval_df[i][j], df], ignore_index=True)
            elif path.endswith('train/triples.tsv'):
                i, j = (0,2)
                train_test_eval_df[i][j] = pd.concat([train_test_eval_df[i][j], transform_triple(pd.read_csv(path, sep ='\t'), len(train_test_eval_df[0][0]), len(train_test_eval_df[0][1]) )], ignore_index=True)
            elif path.endswith('train/queries.tsv'):
                i, j = (0,1)
                name = 'QID'
                df = pd.read_csv(path, sep ='\t')
                df[name] = df[name].apply(lambda row : row + len(train_test_eval_df[i][j]))
                train_test_eval_df[i][j] = pd.concat([train_test_eval_df[i][j], df], ignore_index=True)
            elif path.endswith('test/passages.tsv'):
                i, j = (1,0)
                name = 'PID'
                df = pd.read_csv(path, sep ='\t')
                df[name] = df[name].apply(lambda row : row + len(train_test_eval_df[i][j]))
                train_test_eval_df[i][j] = pd.concat([train_test_eval_df[i][j], df], ignore_index=True)
            elif path.endswith('test/triples.tsv'):
                i, j = (1,2)
                train_test_eval_df[i][j] = pd.concat([train_test_eval_df[i][j], transform_triple(pd.read_csv(path, sep ='\t'), len(train_test_eval_df[1][0]), len(train_test_eval_df[1][1]) )], ignore_index=True)
            elif path.endswith('test/queries.tsv'):
                i, j = (1,1)
                name = 'QID'
                df = pd.read_csv(path, sep ='\t')
                df[name] = df[name].apply(lambda row : row + len(train_test_eval_df[i][j]))
                train_test_eval_df[i][j] = pd.concat([train_test_eval_df[i][j], df], ignore_index=True)
            elif path.endswith('val/passages.tsv'):
                i, j = (2,0)
                name = 'PID'
                df = pd.read_csv(path, sep ='\t')
                df[name] = df[name].apply(lambda row : row + len(train_test_eval_df[i][j]))
                train_test_eval_df[i][j] = pd.concat([train_test_eval_df[i][j], df], ignore_index=True)
            elif path.endswith('val/triples.tsv'):
                i, j = (2,2)
                train_test_eval_df[i][j] = pd.concat([train_test_eval_df[i][j], transform_triple(pd.read_csv(path, sep ='\t'), len(train_test_eval_df[2][0]), len(train_test_eval_df[2][1]) )], ignore_index=True)
            elif path.endswith('val/queries.tsv'):
                i, j = (2,1)
                name = 'QID'
                df = pd.read_csv(path, sep ='\t')
                df[name] = df[name].apply(lambda row : row + len(train_test_eval_df[i][j]))
                train_test_eval_df[i][j] = pd.concat([train_test_eval_df[i][j], df], ignore_index=True)
            elif 'fandoms_all' not in path and path.endswith('train/wiki.json'):
                f = open(path)

                json_f = json.load(f)
                len_p = len(train_test_eval_df[0][0])
                wikis[0] = wikis[0] | transform_wiki(json_f, len_p)

            elif 'fandoms_all' not in path and path.endswith('test/wiki.json'):
                f = open(path)

                json_f = json.load(f)

                len_p = len(train_test_eval_df[1][0])
                wikis[1] = wikis[1] | transform_wiki(json_f, len_p) 
            elif 'fandoms_all' not in path and path.endswith('val/wiki.json'):
                f = open(path)

                json_f = json.load(f)
                len_p = len(train_test_eval_df[2][0])
                wikis[2] = wikis[2] | transform_wiki(json_f, len_p)

    os.makedirs(output_path_dir + '/fandoms_all', exist_ok=True)
    dirs = [output_path_dir + '/fandoms_all/test/', output_path_dir + '/fandoms_all/train/', output_path_dir + '/fandoms_all/val/']
    for dire in dirs:
        os.makedirs(dire, exist_ok=True)
    for i in range(3):
        train_test_eval_df[i][0].to_csv(dirs[i] + 'passages.tsv', index=False, sep="\t")
        train_test_eval_df[i][1].to_csv(dirs[i] + 'queries.tsv', index=False, sep="\t")
        train_test_eval_df[i][2].to_csv(dirs[i] + 'triples.tsv', index=False, sep="\t")
        with open(dirs[i] + 'wiki.json', 'w') as f:
            json.dump(wikis[i], f, indent=4)
