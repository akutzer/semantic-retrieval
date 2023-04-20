## Ressources


### Links

Blog mit schicken Visualisierungen zu Deep Learning: https://jalammar.github.io/ \
Advanced Information Retrieval Kurs (nebst Aufzeichnungen): https://github.com/sebastian-hofstaetter/teaching#lectures \
Survey Paper zu Transformer für Information Retrieval: https://arxiv.org/abs/2010.06467 \
Paper der größten Information Retrieval Konferenz (SIGIR) als Inspiration: https://dl.acm.org/doi/proceedings/10.1145/3477495 



### Libs

Potentiell nützliche Software: \
Natural Language Toolkit: https://www.nltk.org \
Gensim - Word Embeddings (Semantic Vector Representations of Words): https://radimrehurek.com/gensim/ \
Pipelines für NLP (NER or POS Tagging, Parsing etc.): https://spacy.io \
Simple Transformers: https://simpletransformers.ai \
Transformers & Tokenizers: https://huggingface.co/docs \
PyTorch (DeepLearning Framework): https://pytorch.org/ 


### Reference table for papers

| file name                                 | description                          | useful for/ used in                         |
|-------------------------------------------|--------------------------------------|---------------------------------------------|
| [1706.03762.pdf](papers/1706.03762.pdf)   | Vanilla Transformer Paper            | Introduction to Transformer <br> *(alternativly: https://nlp.seas.harvard.edu/annotated-transformer/  describes the paper more detailed and with PyTorch code)*  |
| [1810.04805.pdf](papers/1810.04805.pdf)   | BERT Paper                           | useful Base-Model; paper also describes methodes to mark the answer to the given question in the given paragraph (see 4.2 SQuAD v1.1 and 4.3 SQuAD v2.0) |
| [2004.12832.pdf](papers/2004.12832.pdf)   | ColBERT Paper                        | BERT based model for efficient passage search; focuses on computational efficiency; describes a re-ranking and a full retrieval approach|
| [1903.06902v3.pdf](papers/1903.06902v3.pdf)   | Deep Look into Neural Ranking Models (NRMs) for IR                            | Survey paper; describes tasks & architecture types of NRMs as well as formalizing the learning problem; doesn't include (more modern) Transformer-based architecures, but still a good first overview; problem: it feels a bit out of date and some categories or assumptions don't really extrapolate to transformer based models, which undermines the idea of "gain[ing] some insights for future development" |
| [1611.09268.pdf](papers/1611.09268.pdf)   | MS MARCO Paper                        | Describes how they gathered the dataset, how it is structured and for what tasks it could be used (e.g predicting if the answer is in a passage, predict the text span of the answer or generate a well-formed answer  |
| [2202.06991.pdf](papers/2202.06991.pdf)   | Differentiable Search Index (DSI)     | Goal: Instead of indexing the documents and queries and calculating the similarity between them and ranking them based on that, the query is directly mapped to the document identifier in an end-to-end way; so all the information of the training corpus is encoded in the parameters of the model  |
| ...                                       |  ...                                 |...                                          |


