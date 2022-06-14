from typing import List

import torch
from dataclasses import dataclass
from torch import Tensor, tensor

from colbert.modeling.kd_emb_tokenization import QueryTokenizer
from colbert.modeling.kd_emb_tokenization.cross_tokenization import CrossTokenizer


@dataclass
class Tokens:
    Q_ids: Tensor = None
    Q_mask: Tensor = None
    D_ids: Tensor = None
    D_mask: Tensor = None


@dataclass
class DocTokens:
    pos_ids: Tensor = None
    pos_mask: Tensor = None
    neg_ids: Tensor = None
    neg_mask: Tensor = None


def push_sep_to_end(input_ids: Tensor, special_tokens: dict, is_cross=False):
    batch_size = input_ids.shape[0]
    if is_cross:
        input_ids[input_ids == special_tokens['[SEP]']] = special_tokens['[PAD]']
        input_ids[input_ids == special_tokens['[PAD]']] = special_tokens['[MASK]']
        return torch.cat((
            input_ids[:, :-1],
            tensor([[special_tokens['[SEP]']]]).repeat(batch_size, 1)
        ), dim=1)
    else:
        input_ids[input_ids == special_tokens['[SEP]']] = special_tokens['[MASK]']
        return torch.cat((
            input_ids[:, :-1],
            tensor([[special_tokens['[SEP]']]]).repeat(batch_size, 1)
        ), dim=1)


def tensorize_triples(query_tokenizer: QueryTokenizer,
                      doc_tokenizer: QueryTokenizer,
                      cross_tokenizer: CrossTokenizer,
                      queries: List[str],
                      positives: List[str],
                      negatives: List[str],
                      scores: List[float],
                      bsize: int):
    assert len(queries) == len(positives) == len(negatives) == len(scores)
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    # ======== COLBERT ======== #
    col_toks = Tokens(**{
        **dict(zip(
            ['Q_ids', 'Q_mask'],
            query_tokenizer.tensorize(queries)
        )),
        **dict(zip(
            ['D_ids', 'D_mask'],
            list(map(
                lambda i: i.view(2, N, -1),
                doc_tokenizer.tensorize(positives + negatives)
            ))
        ))
    })
    col_toks.Q_ids = push_sep_to_end(col_toks.Q_ids,
                                     special_tokens=dict(zip(
                                         query_tokenizer.tok.all_special_tokens,
                                         query_tokenizer.tok.all_special_ids
                                     )),
                                     is_cross=False)
    col_toks.Q_mask = torch.ones_like(col_toks.Q_ids)

    # ======== CROSS ======== #
    cross_toks = Tokens(**{
        **dict(zip(
            ['Q_ids', 'Q_mask', 'D_ids', 'D_mask'],
            cross_tokenizer.tensorize(queries=queries,
                                      docs=positives + negatives,
                                      push_sep_to_end=push_sep_to_end)
        ))
    })
    cross_toks.D_ids = cross_toks.D_ids.view(2, N, -1)
    cross_toks.D_mask = cross_toks.D_mask.view(2, N, -1)

    # =========== Sort by maxlens =========== #
    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = col_toks.D_mask.sum(-1).max(0).values
    indices = maxlens.sort().indices
    # colbert
    col_toks.Q_ids, col_toks.Q_mask = col_toks.Q_ids[indices], col_toks.Q_mask[indices]
    col_toks.D_ids, col_toks.D_mask = col_toks.D_ids[:, indices], col_toks.D_mask[:, indices]
    # cross
    cross_toks.Q_ids, cross_toks.Q_mask = cross_toks.Q_ids[indices], cross_toks.Q_mask[indices]
    cross_toks.D_ids, cross_toks.D_mask = cross_toks.D_ids[:, indices], cross_toks.D_mask[:, indices]

    col_doc_toks = DocTokens(**{
        **dict(zip(['pos_ids', 'neg_ids'], col_toks.D_ids)),
        **dict(zip(['pos_mask', 'neg_mask'], col_toks.D_mask))
    })
    cross_doc_toks = DocTokens(**{
        **dict(zip(['pos_ids', 'neg_ids'], cross_toks.D_ids)),
        **dict(zip(['pos_mask', 'neg_mask'], cross_toks.D_mask))
    })

    # =========== split to batches =========== #
    col_query_batches = _split_into_batches(col_toks.Q_ids, col_toks.Q_mask, bsize)
    col_positive_batches = _split_into_batches(col_doc_toks.pos_ids, col_doc_toks.pos_mask, bsize)
    col_negative_batches = _split_into_batches(col_doc_toks.neg_ids, col_doc_toks.neg_mask, bsize)

    cross_query_batches = _split_into_batches(cross_toks.Q_ids, cross_toks.Q_mask, bsize)
    cross_positive_batches = _split_into_batches(cross_doc_toks.pos_ids, cross_doc_toks.pos_mask, bsize)
    cross_negative_batches = _split_into_batches(cross_doc_toks.neg_ids, cross_doc_toks.neg_mask, bsize)
    # cross_positive_batches = _split_into_batches(cross_pos_pair_ids, cross_pos_pair_mask, bsize)
    # cross_negative_batches = _split_into_batches(cross_neg_pair_ids, cross_neg_pair_mask, bsize)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask), \
        (cq_ids, cq_mask), (cp_ids, cp_mask), (cn_ids, cn_mask) \
            in zip(col_query_batches, col_positive_batches, col_negative_batches,
                   cross_query_batches, cross_positive_batches, cross_negative_batches):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))

        C_QDp_ids, C_QDp_mask = cross_tokenizer.concat_qd(cq_ids, cq_mask, cp_ids, cp_mask)
        C_QDn_ids, C_QDn_mask = cross_tokenizer.concat_qd(cq_ids, cq_mask, cn_ids, cn_mask)
        CQD = (torch.cat((C_QDp_ids, C_QDn_ids)), torch.cat((C_QDp_mask, C_QDn_mask)))

        # view port
        col_q_range = list(range(2, q_ids.shape[1] - 1))
        cross_q_range = list(map(lambda i: i - 1, col_q_range))
        col_d_range = list(range(2, p_ids.shape[1] - 1))
        cross_d_range = list(map(lambda i: i + cross_q_range[-1], col_d_range))
        col_range = {
            "Q": col_q_range,
            "D": col_d_range
        }
        cross_range = {
            "Q": cross_q_range,
            "D": cross_d_range
        }
        batches.append((Q, D, CQD, col_range, cross_range))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids: Tensor, mask: Tensor, bsize: int):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset + bsize], mask[offset:offset + bsize]))

    return batches
