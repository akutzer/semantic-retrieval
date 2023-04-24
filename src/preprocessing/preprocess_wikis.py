import os
import re
import json
import numpy as np

import nltk
nltk.download("punkt")



def clean_wiki(wiki, min_length=40, page_regex=None, line_regex=None):
    # removes wiki pages which are only 50 symbols long or which start with an URL
    # important: this is an inplace operation
    raw_wiki[:] = [wiki_page for wiki_page in raw_wiki if filter_wiki(wiki_page, min_length=50, regex=page_regex)]

    for i, wiki in enumerate(raw_wiki):
        wiki_text = wiki["text"]
        parags = split_page_in_paragraphs(wiki_text, heading_max_length=40, regex=line_regex)
        wiki["text"] = parags
        raw_wiki[i] = wiki
    
    # remove wiki pages without paragraphs after cleaning
    raw_wiki[:] = [wiki_page for wiki_page in raw_wiki if len(wiki_page["text"]) > 0]
    print_stats(raw_wiki)

    return raw_wiki


def print_stats(wiki):
    n_pages = len(wiki)
    n_parags_per_page = [len(wiki_page["text"]) for wiki_page in wiki]
    n_words_per_parags = [len(parag.split(" ")) for wiki_page in wiki for parag in wiki_page["text"]]

            
    print("~~~~~~~~~~~   Quick Stats   ~~~~~~~~~~~")
    print(f"Wiki contains {n_pages} pages.")
    print(f"Wiki contains {sum(n_parags_per_page)} paragraphs.", end="\n\n")
    print(f"Parags per page:")
    print(f"  mean: {np.mean(n_parags_per_page)}")
    print(f"  min: {np.min(n_parags_per_page)}")
    print(f"  5%: {np.percentile(n_parags_per_page, 0.05)}")
    print(f"  25%: {np.percentile(n_parags_per_page, 0.25)}")
    print(f"  50%: {np.percentile(n_parags_per_page, 0.5)}")
    print(f"  75%: {np.percentile(n_parags_per_page, 0.75)}")
    print(f"  95%: {np.percentile(n_parags_per_page, 0.95)}")
    print(f"  max: {np.max(n_parags_per_page)}", end="\n\n")

    print(f"Words per paragraph:")
    print(f"  mean: {np.mean(n_words_per_parags)}")
    print(f"  min: {np.min(n_words_per_parags)}")
    print(f"  5%: {np.percentile(n_words_per_parags, 0.05)}")
    print(f"  25%: {np.percentile(n_words_per_parags, 0.25)}")
    print(f"  50%: {np.percentile(n_words_per_parags, 0.5)}")
    print(f"  75%: {np.percentile(n_words_per_parags, 0.75)}")
    print(f"  95%: {np.percentile(n_words_per_parags, 0.95)}")
    print(f"  max: {np.max(n_words_per_parags)}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", end="\n\n")



def filter_wiki(wiki_page, min_length, regex=None):
    wiki_text = wiki_page["text"].strip()
    if len(wiki_text) < min_length:
        return False
    
    if regex is not None and regex.match(wiki_text):
        return False 
    
    return True


def split_paragraphs(paragraph, max_words_per_line):
    sentences = nltk.sent_tokenize(paragraph)
    sub_paragraphs = []
    current_sub_paragraph = ""
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        sub_paragraph_length = len(current_sub_paragraph.split())

        if sub_paragraph_length + sentence_length <= max_words_per_line:
            current_sub_paragraph += sentence + " "
        else:
            if 2*(max_words_per_line - sub_paragraph_length) >= sentence_length:
                current_sub_paragraph += sentence
                sub_paragraphs.append(current_sub_paragraph.strip())
                current_sub_paragraph = ""
            else:
                if current_sub_paragraph.strip():
                    sub_paragraphs.append(current_sub_paragraph.strip())
                current_sub_paragraph = sentence + " "
    
    if current_sub_paragraph:
        sub_paragraphs.append(current_sub_paragraph.strip())
    
    return sub_paragraphs


def split_page_in_paragraphs(wiki_page, heading_max_length=40, max_words_per_line=120, regex=None):
    parags = wiki_page.split("\n")

    # first iteration: drop links and empty lines
    clean_parags = []
    for parag in parags:
        parag = parag.strip()
        if parag == "":
            continue

        if regex is not None and regex.match(parag):
            continue
        
        clean_parags.append(parag)
    

    parags, clean_parags = clean_parags, clean_parags[:1]

    # second iteration:
    # (1) merge a line with it's previous line, in case the former was a heading
    # (2) creates a new paragraph after `max_words_per_line`, but still allows to
    #     finish the sentence, so lines can be longer
    # TODO: merge short paragraphs
    last_parag_heading = ""
    for j, parag in enumerate(parags[1:], start=1):
        prev_parag = parags[j - 1]
        sub_parags = split_paragraphs(parag, max_words_per_line=100)
        if len(prev_parag) <= heading_max_length:
            # remove the last line, since it was a heading, and merge it with
            # the current line
            clean_parags.pop(-1)
            clean_parags.extend([f"{prev_parag[:-1]}: {sub_parag}" for sub_parag in sub_parags])
            last_parag_heading = prev_parag[:-1]
        else:
            if last_parag_heading:
                clean_parags.extend([f"{last_parag_heading}: {sub_parag}" for sub_parag in sub_parags])
            else:
                clean_parags.extend(sub_parags)
    
    # drop empty sub paragraphs just in case
    clean_parags[:] = [parag.strip() for parag in clean_parags if len(parag.strip())]

    return clean_parags



if __name__ == "__main__":

    WIKI_PATHS = "../../data/preprocessing/dumps"
    LINE_ANTI_PATTERN = "^&lt;"
    line_regex = re.compile(LINE_ANTI_PATTERN)

    for file in os.listdir(WIKI_PATHS):
        path_to_wiki = os.path.join(WIKI_PATHS, file)

        # skip directories and non-json files
        if not os.path.isfile(path_to_wiki) or not path_to_wiki.endswith("_raw.json"):
            continue

        print("#"*50)
        print(f"Starting to clean {file.replace('_raw', '')} wiki.")
        print("#"*50, end="\n\n")        

        with open(path_to_wiki, mode="r", encoding="utf-8") as f:
            raw_wiki = json.load(f)        
        
        cleaned_wiki = clean_wiki(raw_wiki, min_length=40, line_regex=line_regex)

        path_to_cleaned_wiki = os.path.join(os.path.split(WIKI_PATHS)[0], file.replace("_raw", ""))
        with open(path_to_cleaned_wiki, mode="w", encoding="utf-8") as f:
            json.dump(cleaned_wiki, f, indent=1)
