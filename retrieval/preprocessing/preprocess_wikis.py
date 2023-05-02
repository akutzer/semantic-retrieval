#!/usr/bin/env python3
import os
import re
import json
import numpy as np

import nltk
nltk.download("punkt")



def clean_wiki(wiki, min_length=0, page_regex=None, parag_regex=None,
               max_heading_length=5, max_words_per_parag=250, min_words_per_parag=20,
               print_statistics=True):
    # removes wiki pages which are only 50 symbols long or which start with an URL(be careful)
    # important: this is an inplace operation
    raw_wiki[:] = [wiki_page for wiki_page in raw_wiki if wiki_filter(wiki_page, min_length=min_length, regex=page_regex)]

    # split to long paragraphs into seperate sub-paragraphs and adds the last
    # heading at the beginning of each new sub-paragraph
    # TODO: maybe merge to short paragraphs, since sometime more than 95% of all
    # paragraphs are shorter than 6 words
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
    # TODO:
    # (a) maybe merge short paragraphs?
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
            if len(prev_parag.split(" ")) <= min_words_per_parag and len(parag) > max_heading_length:
                clean_parags.pop(-1)
                sub_parags = split_paragraphs(prev_parag + " " + parag, max_words_per_parag=max_words_per_parag, min_words_per_parag=min_words_per_parag)
            if last_parag_heading:
                clean_parags.extend([f"[{last_parag_heading}] {sub_parag}" for sub_parag in sub_parags])
            else:
                clean_parags.extend(sub_parags)
    
    # drop empty sub paragraphs just in case
    clean_parags[:] = [parag.strip() for parag in clean_parags if len(parag.strip())]

    return clean_parags

# TODO: add min_words_per_parag
def split_paragraphs(paragraph, max_words_per_parag, min_words_per_parag):
    sentences = nltk.sent_tokenize(paragraph)
    sub_paragraphs = []
    current_sub_paragraph = ""

    #TODO:
    # (a) if a sentence is 50% longer than max_words_per_parag or so,
    #     it must be split into sub-sentences
    for sentence in sentences:
        sentence_length = len(sentence.split())
        sub_paragraph_length = len(current_sub_paragraph.split())

        if sub_paragraph_length + sentence_length <= max_words_per_parag:
            current_sub_paragraph += sentence + " "
        else:
            if sentence_length > max_words_per_parag:
                sub_paragraphs.append(current_sub_paragraph.strip())
                sub_paragraphs.extend(split_sentence(sentence))
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

# assumption: very long sentences use "," as separator
# if not, then sentences will be doubled
def split_sentence(sentence):
    half_sentence = ""
    short_sentences = []
    sentence_halvelength = len(sentence.split())/2
    for subclause in sentence.split(", "):
        if len(half_sentence.split()) <= sentence_halvelength:
            half_sentence += subclause + ", "

    short_sentences.append(half_sentence)
    short_sentences.append(sentence.replace(half_sentence, ""))

    return short_sentences


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
    PAGE_ANTI_PATTERN = ""
    page_regex = re.compile(PAGE_ANTI_PATTERN)

    # regex pattern for paragraphs, which should be removed, will be applied to each paragraph
    # in a page seperatly
    PARAGRAPH_ANTI_PATTERN = "^&lt;"
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
            raw_wiki, min_length=MIN_LENGTH, parag_regex=parag_regex,
            max_heading_length=MAX_HEADING_LENGTH,
            max_words_per_parag=MAX_WORDS_PER_PARAG, 
            min_words_per_parag=MIN_WORDS_PER_PARAG,
            print_statistics=True)

        path_to_cleaned_wiki = os.path.join(os.path.split(WIKI_PATHS)[0], file.replace("_raw", ""))
        with open(path_to_cleaned_wiki, mode="w", encoding="utf-8") as f:
            json.dump(cleaned_wiki, f, indent=1)
