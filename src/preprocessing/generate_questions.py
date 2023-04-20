import os
import json
import re
from tqdm import tqdm
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer



def split_page_in_paragraphs(wiki_page, heading_max_length=40):
    parags = wiki_page.split("\n")

    # first iteration: drop links and empty lines
    clean_parags = []
    for parag in parags:
        parag = parag.strip()
        if parag == "":
            continue

        if re.match("&lt;", parag):
            continue
        
        clean_parags.append(parag)
    

    parags, clean_parags = clean_parags, clean_parags[:1]

    # second iteration: merge a line with it's previous line, in case the former was a heading
    last_parag_heading = ""      
    for j, parag in enumerate(parags[1:], start=1):
        prev_parag = parags[j - 1]
        if len(prev_parag) <= heading_max_length:
            # remove the last line, since it was a heading, and merge it with
            # the current line
            clean_parags.pop(-1)
            clean_parags.append(f"{prev_parag[:-1]}: {parag}")
            last_parag_heading = prev_parag[:-1]
        else:
            if last_parag_heading:
                clean_parags.append(f"{last_parag_heading}: {parag}")
            else:
                clean_parags.append(parag)


    return clean_parags


def generate_questions(parags, batch_size=16, device="cpu", **generator_args):
    question_answer = []

    model_name = "allenai/t5-small-squad2-question-generation"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # TODO: choose batch_size depending on the sequence length to prevent
    # out-memory-errors when processing long sequences
    for i in range(0, len(parags), batch_size):
        parags_batch = parags[i: i+batch_size]
        input_ids = tokenizer.batch_encode_plus(parags_batch, return_tensors="pt", padding=True)
        
        for key, tensor in input_ids.items():
            input_ids[key] = tensor.to(device)

        out = model.generate(**input_ids, **generator_args)
        questions = tokenizer.batch_decode(out, skip_special_tokens=True)
        question_answer.extend(zip(questions, parags_batch))
    
    return question_answer


    


def main():
    # create new directories for the data dumps in the data/ directory
    wiki_dumps_path = "../../data/preprocessing/"
    dataset_path = "../../data/fandom-qa/"
    os.makedirs(dataset_path, exist_ok=True)

    PAGES = 100
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    for filename in os.listdir(wiki_dumps_path):
        file_path = os.path.join(wiki_dumps_path, filename)
        if not os.path.isfile(file_path) or not file_path.endswith(".json"):
            continue
        
        with open(file_path, mode="r") as f:
            data = json.load(f)    
        
        for i, wiki_page in enumerate(tqdm(data[:PAGES])):
            wiki_page["text"] = split_page_in_paragraphs(wiki_page["text"])
            wiki_page["text"] = generate_questions(wiki_page["text"], batch_size=192, device=DEVICE, max_new_tokens=50)

        
        file_name = os.path.splitext(os.path.basename(file_path))[0] + "_qa.json"
        file_path = os.path.join(dataset_path, file_name)
        print(f"Writing {file_name}")
        with open(file_path, mode="w") as f:
            json.dump(data[:PAGES], f)


if __name__ == "__main__":
    main()