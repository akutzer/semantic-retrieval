from retrieval.indexing.colbert_retriever import ColBERTRetriever
from retrieval.models.colbert.inference import ColBERTInference

colbert = ColBERTInference.from_pretrained("../../data/colbertv2.0/")
query, mask = colbert.tokenizer.tensorize("Hallooooo", mode="doc")
# mask = mask.bool()
print(query.shape, mask.shape)
print(colbert.colbert.backbone.embeddings)
# print(help(colbert.colbert.backbone.embeddings))

# Token embedding, positional embedding, and normalization
normal_emb = colbert.colbert.backbone.embeddings(query)
print(normal_emb.shape)

# maybe it is better to remove the padding tokens (when using batches of strings as input)
normal_emb = normal_emb[mask.bool()]
print(normal_emb, normal_emb.shape)

# The raw token embedding:
word_emb = colbert.colbert.backbone.embeddings.word_embeddings(query)
word_emb = word_emb[mask.bool()]
print(word_emb, word_emb.shape)