Dummy directory. Keep data locally in this directory or at the HPC.

# Notes on pulling qa files from the HPC:
- navigate to data folder in terminal
- run: ```sftp -acc_name-@taurus.hrsk.tu-dresden.de```
- run: ```get -r /scratch/ws/0/tidi938c-datastoragefiles/fandoms_qa```

# Notes on further processing
- The script [process_qa_json.py](/retrieval/preprocessing/process_qa_json.py) can be run next to create triples.tsv, passages.tsv, queries.tsv and wiki.json