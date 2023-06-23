import torch

from retrieval.models import ColBERTInference, load_colbert_and_tokenizer, inference_to_embedding




CHECKPOINT = "../../data/colbertv2.0"
colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT)
inference = ColBERTInference(colbert, tokenizer)
inference_norm = inference_to_embedding(inference, just_word_emb=False, layer_norm=True)
inference_no_norm = inference_to_embedding(inference, just_word_emb=False, layer_norm=False)

inference_norm.colbert.eval()
inference_no_norm.colbert.eval()
# inference_no_norm.colbert.backbone.embeddings.LayerNorm = inference_norm.colbert.backbone.embeddings.LayerNorm

# inference_norm.colbert.backbone.embeddings.dropout.p = 0.0
# inference_no_norm.colbert.backbone.embeddings.dropout.p = 0.0

print(inference_norm.colbert, inference_no_norm.colbert)
# data = torch.randint(0, 255, (16, 220))


DOC = "This is an example input!"

eps = inference.colbert.backbone.embeddings.LayerNorm.eps
weight = inference.colbert.backbone.embeddings.LayerNorm.weight
bias = inference.colbert.backbone.embeddings.LayerNorm.bias


with torch.inference_mode():
    out = inference_norm.doc_from_text([DOC])[0]
    out_no_norm = inference_no_norm.doc_from_text([DOC])[0]

    print(out.shape, out_no_norm.shape)
    mean = out_no_norm.mean(-1).unsqueeze(-1)
    var = out_no_norm.var(-1).unsqueeze(-1)
    out_norm = (out_no_norm - mean) / torch.sqrt(var + eps)
    out_norm_affine = out_norm * weight + bias
    # out_norm = torch.layer_norm(out_no_norm, (768,), weight=inference_norm.colbert.backbone.embeddings.LayerNorm.weight, bias=inference_norm.colbert.backbone.embeddings.LayerNorm.bias, eps=inference_norm.colbert.backbone.embeddings.LayerNorm.eps)#torch.layer_norm(out_no_norm, (768,), )
    
    # out_norm = inference_norm.colbert.backbone.embeddings.LayerNorm(out_no_norm)
    # out_norm_affine = out_norm



    print(out, out_no_norm, out_norm_affine)
    print(out.mean(-1), out.std(-1))
    print(out_norm.mean(-1), out_norm.std(-1))
    print(out_norm_affine.mean(-1), out_norm_affine.std(-1))

    out_denorm = (out - inference_norm.colbert.backbone.embeddings.LayerNorm.bias) / inference_norm.colbert.backbone.embeddings.LayerNorm.weight
    print(out_denorm.mean(-1), out_denorm.std(-1))

    print(out / out_norm_affine)
    print(out - out_norm_affine)

    print(weight[:10])
    print(bias[:10])

    # out_norm_affine_denorm = (out_norm_affine - inference_norm.colbert.backbone.embeddings.LayerNorm.bias) / inference_norm.colbert.backbone.embeddings.LayerNorm.weight
    # print(out_norm_affine_denorm.mean(-1), out_norm_affine_denorm.std(-1))
    # print(inference_no_norm.colbert.backbone.embeddings.dropout(torch.ones(1000)).sum())
    # print(out_no_norm * weight + bias)