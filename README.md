# Semantic Retrieval in FANDOM Wikis

## Task

The goal is to develop a **semantic retrieval system for wikis**. The system should be able to be connected to any wiki (preferably language-independent). **Therefore, it must not be adapted to the concrete dataset.** The challenge here is also that generally there are no or only a few queries.

---

**Setting:** 
Find the matching passage (paragraph, sentence, up to you) that answers a question written by
the user. We consider only questions as queries. The data source is written in Wiki style. We typically
return multiple (N) candidate passages to the user (how large N is is up to you, could be 5, 10, or 100)

---

## Demo Version

We developed a [demo version](https://colab.research.google.com/drive/1cfob0zBghF2vRkeR9696YRHmD2mjT-WU?usp=sharing) where you can test our retrieval system.

## Installation

If you want to run our project locally you first need to install all requirements:
```python
pip3 install -r requirements.txt
pip3 install -e .
```

If you want to **train** a model yourself, have a look at the [train_ms_marco.sh](retrieval/training/train_ms_marco.sh) script.

For **evaluation**, have a look at the [evaluate_retrieval.sh](retrieval/evaluation/evaluate_retrieval.sh) script.