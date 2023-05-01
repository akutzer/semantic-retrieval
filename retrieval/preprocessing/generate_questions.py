#!/usr/bin/env python3
import os
import json
import re
from tqdm import tqdm
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig
from torch.utils.data import Dataset, DataLoader



class QADataset(Dataset):
    def __init__(self, path_to_wiki_qa):
        with open(path_to_wiki_qa, "r", encoding="utf-8") as f:
            data = json.load(f)

        # data elements are of form (doc_idx, parag_idx, parag)
        self.data = []
        for i, wiki_page in enumerate(data):
            for j, parag in enumerate(wiki_page["text"]):
                self.data.append((i, j, parag))

        # TODO: sort data by length of paragraph
        self.data.sort(key=lambda x: (len(x[2].split(" ")), x[0], x[1]), reverse=True)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collate_fn_wrapper(tokenizer, max_length=512):
    def collate_fn(batch):
        doc_ids, parag_ids, parags = zip(*batch)
        parags_token = tokenizer(
            parags, padding="longest", truncation="longest_first",
            return_tensors="pt", max_length=max_length)
        return doc_ids, parag_ids, parags, parags_token
    
    return collate_fn


def main():
    # create new directories for the data dumps in the data/ directory
    WIKI_DUMPS_PATH = "../../data/fandoms/"
    DATASET_PATH = "../../data/fandom-qa/"
    os.makedirs(DATASET_PATH, exist_ok=True)

    # dataloading parameters
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    pin_memory = DEVICE.startswith("cuda")
    BATCH_SIZE = 64 #255
    N_WORKERS = 0   # TODO: check if workers actually do anything
    
    # question generation parameters
    #TODO: play around with parameters
    #NOTE: computation seems to scales more or less linear with the number of beam, so decrease batch_size
    #      wait longer for better results :^)
    # see more: https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationConfig
    generation_config = GenerationConfig(
        max_new_tokens=50,
        num_beams=4,
        num_beam_groups = 1,
        temperature = 1.0,
    )

    # created tokenizer + model, load the model onto the selected device
    model_name = "allenai/t5-small-squad2-question-generation"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(DEVICE)

    collate_fn = collate_fn_wrapper(tokenizer)

    # for filename in os.listdir(WIKI_DUMPS_PATH):
    for filename in ["witcher.json", "harry_potter.json", "elder_scrolls.json"]:    
        path_to_wiki = os.path.join(WIKI_DUMPS_PATH, filename)
        if not os.path.isfile(path_to_wiki) or not path_to_wiki.endswith(".json"):
            continue

        dataset = QADataset(path_to_wiki)
        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS,
            pin_memory=pin_memory, collate_fn=collate_fn
        )

        question_answer = []
        for doc_idcs, parag_idcs, _, parags_token in tqdm(dataloader):
            for key, tensor in parags_token.items():
                parags_token[key] = tensor.to(DEVICE)

            out = model.generate(**parags_token, generation_config=generation_config)
            questions = tokenizer.batch_decode(out, skip_special_tokens=True)
            question_answer.extend(zip(doc_idcs, parag_idcs, questions))
        

        with open(path_to_wiki, "r", encoding="utf-8") as f:
            wiki_data = json.load(f)
        
        for doc_idx, parag_idx, question in question_answer:
            parag = wiki_data[doc_idx]["text"][parag_idx]
            if not question.endswith("?") or question[:-1] in parag:
                wiki_data[doc_idx]["text"][parag_idx] = [[], parag]
            else:
                wiki_data[doc_idx]["text"][parag_idx] = [[question], parag]
        

        path_to_wiki_qa = os.path.splitext(os.path.basename(path_to_wiki))[0] + "_qa.json"
        path_to_wiki_qa = os.path.join(DATASET_PATH, path_to_wiki_qa)
        print(f"Writing {path_to_wiki_qa}")
        with open(path_to_wiki_qa, mode="w", encoding="utf-8") as f:
            json.dump(wiki_data, f, indent=1)



if __name__ == "__main__":
    main()
