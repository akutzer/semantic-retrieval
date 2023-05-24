#!/usr/bin/env python3
import math
from typing import Iterable, Tuple
import torch


def _split_into_batches(ids: torch.Tensor, mask: torch.Tensor, bsize: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    num_batches = math.ceil(ids.shape[0] / bsize)
    for b_ids, b_mask in zip(torch.chunk(ids, num_batches), torch.chunk(mask, num_batches)):
        b_maxlen = b_mask.sum(dim=-1).max()
        # remove unnecessary padding
        b_ids = b_ids[:, :b_maxlen]
        b_mask = b_mask[:, :b_maxlen]
        yield b_ids, b_mask


def _split_into_batches_sorted(ids: torch.Tensor, mask: torch.Tensor, bsize: int):
    total = ids.shape[0]

    # sort by paragraph length
    indices = mask.sum(dim=-1).sort(descending=True).indices
    reverse_indices = indices.sort().indices
    ids, mask = ids[indices], mask[indices]

    # split into sub-batches, while also removing unnecessary padding
    batch_maxlen = mask[::bsize].sum(dim=-1)

    for i in range(0, total, bsize):
        b_ids = ids[i:i+bsize, :batch_maxlen[i//bsize]]
        b_mask = mask[i:i+bsize, :batch_maxlen[i//bsize]]
        b_indices = indices[i:i+bsize]
        yield b_ids, b_mask, b_indices
