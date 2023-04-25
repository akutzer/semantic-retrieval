import os
import json


if __name__ == "__main__":
    QA_PATH = "../../data/fandom-qa/"
    for file in os.listdir(QA_PATH):
        if not "witcher" in file:
            continue
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
                "pIDs": []
            }

            for questions, passage in page["text"]:
                for question in questions:
                    # TODO: sample the negative paragraph using BM25
                    triples.append([qID, pID, -1])
                    qID_2_query[qID] = question
                    qID += 1

                pID_2_passage[pID] = passage
                docID_2_pID[docID]["pIDs"].append(pID)
                pID += 1
        
        filename = os.path.splitext(file)[0]
        qa_dataset_path = os.path.join(QA_PATH, filename)
        os.makedirs(qa_dataset_path, exist_ok=True)

        with open(os.path.join(qa_dataset_path, "triples.train.json"), mode="w", encoding="utf-8") as f:
            json.dump(triples, f)
        
        with open(os.path.join(qa_dataset_path, "queries.train.json"), mode="w", encoding="utf-8") as f:
            json.dump(qID_2_query, f, indent=0)
        
        with open(os.path.join(qa_dataset_path, "passages.train.json"), mode="w", encoding="utf-8") as f:
            json.dump(pID_2_passage, f, indent=0)
        
        with open(os.path.join(qa_dataset_path, "docs.train.json"), mode="w", encoding="utf-8") as f:
            json.dump(docID_2_pID, f, indent=0)

    
