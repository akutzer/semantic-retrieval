#!/usr/bin/env python3
import os
import json
import csv


if __name__ == "__main__":
    QA_PATH = "../../data/fandom-qa/"
    MODE = "tsv" # "tsv" or "json"

    for file in os.listdir(QA_PATH):
        path_to_qa = os.path.join(QA_PATH, file)

        # skip directories and non-json files
        if not os.path.isfile(path_to_qa) or not path_to_qa.endswith(".json"):
            continue

        with open(path_to_qa, "r", encoding="utf-8") as f:
            wiki_qa_data = json.load(f)

        triples = []
        qID_2_query = {}
        pID_2_passage = {}
        docID_2_pID = {}
        
        qID, pID = 0, 0
        for page in wiki_qa_data:
            docID = page["id"]

            if docID in docID_2_pID.keys():
                print("Duplicate docID!!!")
                continue

            docID_2_pID[docID] = {
                "revid": page["revid"],
                "url": page["url"],
                "title": page["title"],
                "PIDs": []
            }

            for questions, passage in page["text"]:
                if isinstance(questions, str):
                    questions = [questions]
                for question in questions:
                    # TODO: sample the negative paragraph using BM25
                    triples.append([qID, pID, -1])
                    qID_2_query[qID] = question
                    qID += 1

                pID_2_passage[pID] = passage
                docID_2_pID[docID]["PIDs"].append(pID)
                pID += 1
        
        filename = os.path.splitext(file)[0]
        qa_dataset_path = os.path.join(QA_PATH, filename)
        os.makedirs(qa_dataset_path, exist_ok=True)


        if MODE.lower() == "tsv":
            with open(os.path.join(qa_dataset_path, "triples.train.tsv"), mode="w", encoding="utf-8", newline="") as trip_f:
                writer = csv.writer(trip_f, delimiter="\t", lineterminator="\n")
                writer.writerows(triples)
            
            with open(os.path.join(qa_dataset_path, "queries.train.tsv"), mode="w", encoding="utf-8", newline="") as q_f:
                writer = csv.writer(q_f, delimiter="\t", lineterminator="\n")
                writer.writerows(qID_2_query.items())
            
            with open(os.path.join(qa_dataset_path, "passages.train.tsv"), mode="w", encoding="utf-8", newline="") as p_f:
                writer = csv.writer(p_f, delimiter="\t", lineterminator="\n")
                writer.writerows(pID_2_passage.items())

        elif MODE.lower() == "json":
            with open(os.path.join(qa_dataset_path, "triples.train.json"), mode="w", encoding="utf-8") as trip_f:
                json.dump(triples, trip_f, indent=0)
            
            with open(os.path.join(qa_dataset_path, "queries.train.json"), mode="w", encoding="utf-8") as q_f:
                json.dump(qID_2_query, q_f, indent=0)
            
            with open(os.path.join(qa_dataset_path, "passages.train.json"), mode="w", encoding="utf-8") as p_f:
                json.dump(pID_2_passage, p_f, indent=0)


        with open(os.path.join(qa_dataset_path, "docs.train.json"), mode="w", encoding="utf-8", newline="") as docs_f:
            json.dump(docID_2_pID, docs_f, indent=0)

    
