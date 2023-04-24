## Information Retrieval

The goal of the task is to develop a **semantic retrieval system for wikis**. The system should be able to be connected to any wiki (preferably language-independent). **Therefore, it must not be adapted to the concrete dataset.** The challenge here is also that generally there are no or only a few queries.

As an example wiki, the Wikia datasets are to be worked with in the context of the project. These are available in different sizes and for different contents.

A rough work plan could look like this:
- read in, understand and preprocess the dataset
- generate queries to the document (e.g. with Doc2Query)
- train a semantic retrieval system (e.g. ColBERT, DSI, Splade, etc.)
- develop a prototypical frontend or connect it to an already existing one

---

**Setting:** 
Find the matching passage (paragraph, sentence, up to you) that answers a question written by
the user. We consider only questions as queries. The data source is written in Wiki style. We typically
return multiple (N) candidate passages to the user (how large N is is up to you, could be 5, 10, or 100)

---


### **Tasks:**
*task schedule will be iterativly improved by all of use :3*

#### 1. Crawling data
- [x] collect xml files from various fandoms
- [x] preprocess the data using [WikiExtractor](https://github.com/attardi/wikiextractor) (e.g. clean from wiki syntax, remove tables, ...)
- [:hourglass:] bringing the data into a fitting file format and file structure
    - communicate together with person responsible for dataloading
    - have a look at benchmark datasets
- [ ] extract multi-linguistic wikis
    - we will probably focus first on english pages and later traing a model on multiple languages
- responsible: Till, Aaron


#### 2. Generating dataset
- [:hourglass:] auto-generate questions
    - if possible: get the text span of the answers for a given paragraph as well
    - if possible: generate questions, which aren't answered in the paragraph
    - currently using [T5-small](https://huggingface.co/allenai/t5-small-squad2-question-generation), but we should test also other models
    - look into: Doc2Query and DocTTTTTQuery
- [ ] create a small human-generated dataset (~1k Questions)
    - can be co-created together with ChatGPT
    - focus on long an diverse questions (variety of different questions) and use different words than the paragraph
- [ ] finding benchmarks/ gold standard datasets for evaluation the final models performance
    - maybe: SQuAD 2.0, MS MARCO Ranking, TREC CAR... ?
- responsible: Till, Aaron


#### 3. Retrieval System
- [:hourglass:] read into different neural retrieval systems and select one or two retrieval approaches + a baseline model
    - baseline: TF-IDF or BM25?
    - first approach: ColBERT?
    - second approach: ???
    - search for the code to the paper (e.g. https://paperswithcode.com/) or implement the model yourself using PyTorch (finding parameters would be very helpful for quicker training)
    - BERT paper describes a possible baseline, where, for a given question and wikipedia paragraph, the answer text span is predicted (on the SQuAD dataset); alternativly one can also just calculate a similarity score for the question and paragraph
    - the model could either predict the text span to the answer or just a similarity score (both ways would be interesting)
    - re-ranking or full retrieval?
    - other exotic approaches can be interesting (probably not big problem if it doesn't outperform baseline) or a performance oriented approach ("model performance"/FLOPs, "model performance"/inference time [µs])



#### 4. Training pipeline & Evaluation
- [ ] implement a simple inference pipeline with the baseline models, since they don't need training
- [ ] implementation of the first approach:
    - [ ] got a running model
    - [ ] dataloading + tokenizing
    - [ ] got a running trainings pipeline (training loop + tracking of the training + checkpoints)
    - [ ] implementation of different evaluation methods/metrics and the collection of the results of the different retrieval system in a meaningful way for the final paper
    - etc.
- [ ] implementation of the second approach:
    - [ ] got a running model
- responsible: ...


#### 5. Technical Demonstration
- demonstration of the model (done however you like; website, colab, application, ...)
- maybe some inference optimizations & pruning if the person in charge is interested in it and there is time
- responsible: ...


#### 6. Paper
- final paper blablabla
- responsible: ...