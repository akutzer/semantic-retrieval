# Model understanding

## 0. How does TF-IDF work?
Term Frequency Inverse Document Frequency
- Term Frequency: um so öfter ein Token oder Wort in einer passage vorkommt desto wichtiger ist es um zu bestimmen wovon
der Text handelt
- Inverse Dokument Frequency: Wörter welche in sehr vielen Dokumenten bzw. Passagen häufig vorkommen sollen jedoch weniger
berücksichtigt werden
- Semantik wird nicht betrachtet
- gefundene Passagen haben nicht den größten Informationsgehalt sondern das beste verhältnis an keywords
- daher leicht manipulierbar

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

## 2.	What is ColBERT capable of that TF-IDF can't do?
Beispiel 1: What do people of the city consider blasphemy? cb: 0 tf_idf: 2040
- TF IDF findet fälschlicherweise 2039 besser geeignete Passagen
- die Passage welche TF IDF als beste markiert enhält 4 mal das Wort city und auch das Wort considerr, aber nützliche 
Informationen sind offensichtlich nicht enhalten
- COLBERT hingegen versteht eine gewisse Menge an Semantik. So versteht COLBERT, dass mit "its citicens" und "everyone"
"people of the city" gemeint sind. Ebenfalls die Wörter "eternal", "symbol" und "holy" waren COLBERT wichtig obwohl diese
nur entfernt mit dem Wort Blasphemie in Verbindung zu bringen sind.  

Beispiel 2: What are some of the names used to refer to a witcher in different languages? cb: 7 tf_idf: 2045
- Although COLBERT did not find the query we were looking for it found another Passage that answers the question.
- Again COLBERT understands synonyms and correaltions between similar words without problems e.g. 
  "englisch translations" for "different languages".
- Interesting Words for Colbert are again: Words from the query and in this case words like "in" and "as", which refer to 
the language and to the name being used.
- COLBERT is even more interested in the quotation marks than the name that we are searching for itself. For COLBERT it
isonly import to know "Here stand a name!", but which name it is is not import to know whether the question is answered 
in this paragraph.
- Note that TF IDF favors short passages containing some words of the query.  

Synomnyms

## 3. What is ColBERT not capable of?
Who left the world to go to find Ciri? cb: 1306 tf_idf: 611
- long passage with alot of other content and only the last short sentence is actually abou the question.

Why was Ciri put into a trance? cb: 729 tf_idf: 135
- ColBERT find another correct passage but the original apssage is on position 729
- answer is a again in the last sentence after a long passage

What are the options to deal with the cook? cb: 299 tf_idf: 65
- Here ColBERT has the same problems human have when trying to understand the question. Does "deal" mean "negotiate" or
"kill"?

What can be done to save and heal during the battle? cb: 298 tf_idf: 204 (FEHLER: CB und TF IDF haben selbe passage preticted
  aber 298 != 204)
- Can ColBERT understand the type of question ("what", "when")?
- Does not seem so in this example.

Who respected witchers? cb: 264 tf_idf: 57
- in this case ColBERT encounters the same problem as TF IDF. The predicted passage contains the word "witcher" to often

## 4. Betrachtet ColBERT die Art des Fragewortes überhaupt?
"Who is Geralt?", "When is Geralt?", "Where is Geralt?", "What is Geralt?", "Is Geralt?", "Why is Geralt?", "How is Geralt?"
are not treated much differently according to visualization

## 3.	Find out what neither ColBERT nor TF-IDF can do. Compare queries that both failed to answered.Maybe you can find a pattern?
      
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