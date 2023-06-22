# Model understanding

## 1.	Test if it's possible to extract roughly position of the answer. 
for example: query is encoded as 32 vectors. For each vector find the most similar passage vectors and visualize 
those 32 token in the passage string, does it correlate with the answer?
- [CLS] and [D] (Document/Passage) token often have a high priority → ignore in KDE
- Often the answer is not directly recognisable by comparing the passage embedding with the query embedding,
  Most of the time the words in the passage that lead to the answer will have the highest similarity with the query. 
  For example: query = "Who won the football championship in 2006?" and a 
  passage = "The football championship in the year 2006 was a great sports event that was won by Italy." 
  The highest similarity will be found in the words "championship", "2006", "was", "won" and "by" instead 
  of the actual answer "Italy", because the actual answer is not relevant to decide whether the question is answered 
  in the passage. Any other word could be used in place of Italy and it would hardly change the likelihood of the 
  passage answering the question.

visualize the unsmoothed and smoothed (KDE or whatever) results

can we make assumptions about how ColBERT might work?

## 2.	Find out what ColBERT is capable of doing that TF-IDF can't do. Compare queries that ColBERT answered successfully but TF-IDF failed. Maybe you can find a pattern? Synonyms, ...?
###example 1
- question: 'How many series of comic books have been published in Poland between 1993 and 1995?'
- passage: "There have been three series of comic books based on Andrzej Sapkowski's . A series published in Poland 
between 1993 and 1995, a 2011 miniseries published by Egmont, and the currently ongoing series published by Dark
Horse Comics, started in 2014. The latter two series are part of ."
- colbert: 3
- tf idf: over 1000 (not exactly known currently) 
###example 2
- query: "Who leads the Order's mutants?"
- passage: '[Civilians] Siegfried leads the Order\'s mutants. He has decided to exterminate the civilian population,
  and I cannot allow that to happen. "I will not let the mutants slaughter civilians."
- colber 11(1000, 11)
- tf idf: over 1000




## 3.	Find out what neither ColBERT nor TF-IDF can do. Compare queries that both failed to answered. Maybe you can find a pattern?
      
## 4.	ColBERT is just context-unaware, synonym-robust embedding?. Compare ColBERT embedding vs just its embedding matrix embedding
      
## 5.	Analyze the embedding space.
- maybe some dimensionality reduction for a visualization
- embedding space evenly used (anisotropy):
Recent work identifies an anisotropy problem in language representations (Ethayarajh, 2019; Li et al., 2020), i.e., the learned embeddings occupy a narrow cone in the vector space, which severely limits their expressiveness. Gao et al. (2019) demonstrate that language models trained with tied input/output embeddings lead to anisotropic word embeddings, and this is further observed by Ethayarajh (2019) in pre-trained contextual representations. Wang et al. (2020) show that singular values of the word embedding matrix in a language model decay drastically: except for a few dominating singular values, all others are close to zero.



---

## **A**

abc **abc**

## **B:**

see [TASKS.md](TASKS.md)