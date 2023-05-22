#!/usr/bin/env python3
import glob
import os


preprocessed_qa_json_path_dir = '../../data/fandoms_qa/'
output_path_dir = "../../data/fandoms_qa/"


if __name__ == "__main__":
    for file in os.listdir(preprocessed_qa_json_path_dir):
        if file.endswith(".json"):
            f = file
            out_dir = output_path_dir + file.split('/')[-1][:-8] + '/'
            os.makedirs(out_dir,exist_ok=True)

            import pandas as pd

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
            import tqdm

            for i in tqdm.tqdm(range(len(df2))):
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
                        # added to not include passages where information was lost due to wikiextractor
                        queries.append(row['positive'][j][k])
                        qids.append(qid_id)
                        qid_id = qid_id + 1


                        queries.append(row['negative'][j][k])
                        qids.append(qid_id)
                        qid_id = qid_id + 1
                        triples.append(( qid_id - 2, qid_id - 1,pid_id - 1))
                wikis.append(wiki_pids)

            print('inconsistent: ' + str(inconsistent))
            print('excluded: ' + str(1.0*exclude/total))

            # %%
            df2 = df2.drop(['text', 'positive', 'negative'], axis=1)


            # %%
            pid_p_df = pd.DataFrame(zip(pids,passages, pid_wid), columns=['PID', 'passage', 'WID'])
            qid_q_df = pd.DataFrame(zip(qids,queries), columns=['QID', 'query'])
            triples_df = pd.DataFrame(triples, columns=['QID+', 'QID-', 'PID'])
            df2['PIDs'] = wikis

            # %%
            triples_df.head(3)

            # %%
            pid_p_df.head(4)

            # %%
            qid_q_df.head(6)

            # %%
            triples_df.head(3)

            # %%
            df2.head(3)

            # %%
            import json
            output_path = out_dir
            triples_df.to_csv(output_path + "triples.tsv", index=False, sep="\t")
            pid_p_df.to_csv(output_path + "passages.tsv", index=False, sep="\t")
            qid_q_df.to_csv(output_path + "queries.tsv", index=False, sep="\t")

            data = df2.to_dict('index')

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
            with open(output_path + 'wiki.json', 'w') as f:
                json.dump(output, f, indent=4)


            # %%
