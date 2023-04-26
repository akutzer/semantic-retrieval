 
### Descriptions

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

- [generate_qa_dataset.py](/src/preprocessing/generate_qa_dataset.py):
    - splits the JSON file containing the question-answer pairs into smaller JSON or TSV files (triples, queries, passages, docs), which reduces redundant information and fits the data format required for ColBERT & co.


### How to run

To download and clean the wikis run the following command: \
*NOTE: Make sure to be in the `/src/preprocessing/` directory, since the download paths are relative to the directory from which the script got executed*
```bash
python3 download_wikis.py
python3 preprocess_wikis.py
```
**TODO:** We could also use [argparser](https://docs.python.org/3/library/argparse.html), so we got a user-friendly command-line interfaces for executing the scripts (e.g. global constants, like the download directory, could be given as optional arguments when executing the script)
