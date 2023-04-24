 
- [download_wikis.py](/src/preprocessing/download_wikis.py):
    - downloads and extracts the wiki dumps into the `semantic-retrieval/data/preprocessing/dumps/` directory
    - the final wiki is save as a json file in the same directory

- [preprocess_wiki.py](/src/preprocessing/preprocess_wiki.py):
    - splits the wiki-pages into paragraphs and sub-paragraphs
    - appends the heading of the paragraph in front of each sub-paragraph
    - removes to short wiki-pages or wiki-pages of a certain form (containing only a link for example)
    - the cleaned wiki is stored in the `semantic-retrieval/data/preprocessing/` directory

- [generate_questions.py](/src/preprocessing/generate_questions.py):
    - generates a question for each sub-paragraph
    - the question-answer pairs are stored in the `semantic-retrieval/data/fandom-qa/` directory