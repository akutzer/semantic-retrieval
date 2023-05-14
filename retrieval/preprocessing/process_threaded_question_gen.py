preprocessed_json_path = '../../data/fandoms/harry_potter.json'
question_generated__json_path = 'harry_potter_questions_a.csv'
output_path_dir = "./temp/"

if __name__ == "__main__":

    f = preprocessed_json_path
    import pandas as pd
    df = pd.read_json(f, orient ='records')
    df

    import json
    import ast
    f2 = question_generated__json_path
    df_q = pd.read_csv(f2)
    df_q = df_q.sort_values(by=['i', 'j'], ascending=True)

    arr = []
    for list in df_q['positive'].values:
        try:
            arr.append(json.loads(list))
        except:
            try:
                arr.append(ast.literal_eval(list))
            except:
                print("fail")


    arr2 = []
    for list in df_q['negative'].values:
        try:
            arr2.append(json.loads(list))
        except:
            try:
                arr2.append(ast.literal_eval(list))
            except:
                print("fail")

                

    i = 0
    while i < len(arr):
        if len(arr[i]) == 3:
            arr2[i].append(arr[i].pop(-1))
        i = i + 1

    df_q['negative'] = arr2
    df_q['positive'] = arr 

    arr = [len(x) for x in df_q['positive'].values]
    print(len(arr))
    print(arr.count(2))
    #df_q

    # %%
    pd.options.display.max_colwidth = 200
    df_q.head()


    # %%
    arr_base = []
    for i in range(len(df)):
        arr = []
        l = len(df.iloc[i]['text'])
        for j in range(l):
            if df_q.loc[(df_q['i'] == i) & (df_q['j'] == j)].empty:
                arr.append([])
            else:
                y = ([x for x in (df_q.loc[(df_q['i'] == i) & (df_q['j'] == j)]['positive'].values)])
                arr.append(y[0])
        arr_base.append(arr)

    df['positive'] = arr_base 


    arr_base = []
    for i in range(len(df)):
        arr = []
        l = len(df.iloc[i]['text'])
        for j in range(l):
            if df_q.loc[(df_q['i'] == i) & (df_q['j'] == j)].empty:
                arr.append([])
            else:
                y = ([x for x in (df_q.loc[(df_q['i'] == i) & (df_q['j'] == j)]['negative'].values)])
                arr.append(y[0])
        arr_base.append(arr)

    df['negative'] = arr_base 


    # %%
    #print(df.iloc[8]['text'])
    #print(df.iloc[8]['positive'])

    # %%
    pd.options.display.max_colwidth = 400
    df.head(5)

    # %%
    pd.options.display.max_colwidth = 50
    df

    # %%
    df.to_json(output_path_dir + "processed_wiki.json", orient='records', indent=4)
    df2 = df.copy()
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

    for i in range(len(df2)):
        row = df.iloc[i]
        wiki_pids = []

        for j in range(len(row['text'])):
            pids.append(pid_id)
            passages.append(row['text'][j])
            pid_wid.append(row['id'])
            wiki_pids.append(pid_id)

            pid_id = pid_id + 1


            if len(row['positive'][j]) != len(row['negative'][j]):
                inconsistent = inconsistent + 1
            for k in range(min(len(row['positive'][j]), len(row['negative'][j]))):
                queries.append(row['positive'][j][k])
                qids.append(qid_id)
                qid_id = qid_id + 1


                queries.append(row['negative'][j][k])
                qids.append(qid_id)
                qid_id = qid_id + 1
                triples.append(( qid_id - 2, qid_id - 1,pid_id - 1))
        wikis.append(wiki_pids)

    print(inconsistent)

    # %%
    df2 = df2.drop(['text', 'positive', 'negative'], axis=1)


    # %%
    pid_p_df = pd.DataFrame(zip(pids,passages, pid_wid),columns=['PID', 'passage', 'WID'])
    qid_q_df = pd.DataFrame(zip(qids,queries),columns=['QID', 'query'])
    triples_df = pd.DataFrame(triples,columns=['QID+', 'QID-', 'PID'])
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
    output_path = output_path_dir
    triples_df.to_csv(output_path+ "triples.tsv",index=False, sep="\t")
    pid_p_df.to_csv(output_path+ "passages.tsv",index=False, sep="\t")
    qid_q_df.to_csv(output_path+ "queries.tsv",index=False, sep="\t")

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



