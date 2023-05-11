#!/usr/bin/env python3
import os
import re
import json
import math
import numpy as np

import nltk
nltk.download("punkt")



def clean_wiki(wiki, min_length=0, page_regex=None, parag_regex=None,
               max_heading_length=5, max_words_per_parag=250, min_words_per_parag=20,
               print_statistics=True):
    # removes wiki pages which are only 50 symbols long or which start with an URL(be careful) or is a Mainpage ("__NOEDIT__")
    # important: this is an inplace operation
    raw_wiki[:] = [wiki_page for wiki_page in raw_wiki if wiki_filter(wiki_page, min_length=min_length, regex=page_regex)]

    # split to long paragraphs into seperate sub-paragraphs and adds the last
    # heading at the beginning of each new sub-paragraph
    for i, wiki in enumerate(raw_wiki):
        wiki_text = wiki["text"]
        parags = split_page_in_paragraphs(
            wiki_text, max_heading_length=max_heading_length,
            max_words_per_parag=max_words_per_parag, min_words_per_parag=min_words_per_parag,
            regex=parag_regex
        )
        wiki["text"] = parags
        raw_wiki[i] = wiki
    
    # remove wiki pages without paragraphs after cleaning
    raw_wiki[:] = [wiki_page for wiki_page in raw_wiki if len(wiki_page["text"]) > 0]

    if print_statistics:
        print_stats(raw_wiki)

    return raw_wiki


def wiki_filter(wiki_page, min_length, regex=None):
    wiki_text = wiki_page["text"].strip()
    if len(wiki_text.split(" ")) < min_length:
        return False
    
    if regex is not None and regex.match(wiki_text):
        print(wiki_text)
        return False
    
    return True


def split_page_in_paragraphs(wiki_page, max_heading_length=5, max_words_per_parag=250, min_words_per_parag=20, regex=None):
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
    # (2) creates a new paragraph after `max_words_per_parag`, but still allows to
    #     finish the sentence, so lines can be longer
    last_parag_heading = ""
    for j, parag in enumerate(parags[1:], start=1):
        prev_parag = parags[j - 1]
        sub_parags = split_paragraphs(parag, max_words_per_parag=max_words_per_parag, min_words_per_parag=min_words_per_parag)
        if len(prev_parag.split(" ")) <= max_heading_length:
            # remove the last line, since it was a heading, and merge it with
            # the current line
            clean_parags.pop(-1)
            last_parag_heading = prev_parag[:-1]
            clean_parags.extend([f"[{last_parag_heading}] {sub_parag}" for sub_parag in sub_parags])
            
        else:
            # merge paragraphs inside a heading if the last sub-paragraph was shorter than
            # the required minimal length of a sub-paragraph and the current paragraph is no heading
            is_heading = len(parag.split()) <= max_heading_length
            if len(clean_parags[-1].split(" ")) < min_words_per_parag and not is_heading:
                last_sub_parag = clean_parags.pop(-1).replace(f"[{last_parag_heading}] ", "")
                sub_parags = split_paragraphs(f"{last_sub_parag} {parag}", max_words_per_parag=max_words_per_parag, min_words_per_parag=min_words_per_parag)

            if last_parag_heading:
                clean_parags.extend([f"[{last_parag_heading}] {sub_parag}" for sub_parag in sub_parags])
            else:
                clean_parags.extend(sub_parags)
    
    # drop empty sub paragraphs just in case
    clean_parags[:] = [parag.strip() for parag in clean_parags if len(parag.strip())]

    # merge short paragraphs
    parags = []
    long_parag = ""
    for parag in clean_parags:
        if len(long_parag.split()) + len(parag.split()) < min_words_per_parag:
            long_parag += parag + " "
        else:
            if long_parag:
                parags.append(long_parag.strip())
            long_parag = parag

    if long_parag:
        if parags and len(long_parag.split(" ")) < min_words_per_parag:
            parags[-1] = f"{parags[-1]} {long_parag}".strip()
        else:
            parags.append(long_parag.strip())
    
    clean_parags = parags

    return clean_parags


def split_paragraphs(paragraph, max_words_per_parag, min_words_per_parag):
    sentences = nltk.sent_tokenize(paragraph)
    sub_paragraphs = []
    current_sub_paragraph = ""

    for sentence in sentences:
        sentence_length = len(sentence.split())
        sub_paragraph_length = len(current_sub_paragraph.split())

        if sub_paragraph_length + sentence_length <= max_words_per_parag:
            current_sub_paragraph += sentence + " "
        else:
            if sentence_length > max_words_per_parag:
                sub_paragraphs.append(current_sub_paragraph.strip())
                sub_paragraphs.extend(split_sentence(sentence, max_words_per_parag, min_words_per_parag))
                current_sub_paragraph = ""
            elif 2*(max_words_per_parag - sub_paragraph_length) >= sentence_length:
                current_sub_paragraph += sentence
                sub_paragraphs.append(current_sub_paragraph.strip())
                current_sub_paragraph = ""
            else:
                if current_sub_paragraph.strip():
                    sub_paragraphs.append(current_sub_paragraph.strip())
                current_sub_paragraph = sentence + " "
    
    if len(current_sub_paragraph.split()) < min_words_per_parag and sub_paragraphs:
        last_parag = sub_paragraphs.pop(-1)
        sub_paragraphs.append(last_parag + " " + current_sub_paragraph)
    elif current_sub_paragraph:
        sub_paragraphs.append(current_sub_paragraph.strip())
    
    return sub_paragraphs


def split_sentence(sentence, max_words_per_parag, min_words_per_parag):
    # if sentence is longer than max_words_per_parag it is split into
    # equally long subsentences
    words = sentence.split(" ")
    n_parts = math.ceil(len(words) / max_words_per_parag)
    parts_len = math.ceil(len(words) / n_parts)

    sentence_parts = [" ".join(words[i:i+parts_len]) for i in range(0, len(words), parts_len)]

    return sentence_parts


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




if __name__ == "__main__":
    # directory containing the wikis, which should be preprocessed
    WIKI_PATHS = "../../data/fandoms/dumps"

    # regex pattern for wiki pages, which should be removed, applies to the whole page
    PAGE_ANTI_PATTERN = r"__[\w]*__"
    page_regex = re.compile(PAGE_ANTI_PATTERN)

    # regex pattern for paragraphs, which should be removed, will be applied to each paragraph
    # in a page seperatly
    PARAGRAPH_ANTI_PATTERN = r"(&lt;|__[\w]*__)"
    parag_regex = re.compile(PARAGRAPH_ANTI_PATTERN)

    # wikis with less than `MIN_LENGTH` words will be discarded
    MIN_LENGTH = 8

    # if a paragraph contains at most `MAX_HEADING_LENGTH` words, then it is considered a
    # paragraph heading and is append to the next paragraphs
    MAX_HEADING_LENGTH = 8
    
    # soft upper bound for paragraphs, which can be exceeded
    MAX_WORDS_PER_PARAG = 120

    # lower bound for paragraphs
    MIN_WORDS_PER_PARAG = 20

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
        
        cleaned_wiki = clean_wiki(
            raw_wiki, min_length=MIN_LENGTH, 
            page_regex = page_regex, 
            parag_regex=parag_regex,
            max_heading_length=MAX_HEADING_LENGTH,
            max_words_per_parag=MAX_WORDS_PER_PARAG, 
            min_words_per_parag=MIN_WORDS_PER_PARAG,
            print_statistics=True)

        path_to_cleaned_wiki = os.path.join(os.path.split(WIKI_PATHS)[0], file.replace("_raw", ""))
        with open(path_to_cleaned_wiki, mode="w", encoding="utf-8") as f:
            json.dump(cleaned_wiki, f, indent=1)
