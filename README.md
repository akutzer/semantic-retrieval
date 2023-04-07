## Information Retrieval

The goal of the task is to develop a **semantic retrieval system for wikis**. The system should be able to be connected to any wiki (preferably language-independent). **Therefore, it must not be adapted to the concrete dataset.** The challenge here is also that generally there are no or only a few queries.

As an example wiki, the Wikia datasets are to be worked with in the context of the project. These are available in different sizes and for different contents.

A rough work plan could look like this:
- read in, understand and preprocess the dataset
- generate queries to the document (e.g. with Doc2Query)
- train a semantic retrieval system (e.g. ColBERT, DSI, Splade, etc.)
- develop a prototypical frontend or connect it to an already existing one