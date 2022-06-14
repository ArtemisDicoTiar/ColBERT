from typing import Tuple, Any

import torch
from torch import Tensor

from transformers import AutoTokenizer

from colbert.parameters import DEVICE


class CrossTokenizer():
    def __init__(self, query_maxlen, doc_maxlen):
        self.tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-electra-base")
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.special_tokens = dict(zip(self.tok.all_special_tokens, self.tok.all_special_ids))

    def tensorize(self,
                  queries: list, docs: list,
                  push_sep_to_end  # this is function
                  ) -> Tuple[Any, Any, Any, Any]:
        q_tokens = self.tok(queries, padding='max_length', truncation=True,
                            return_tensors='pt', max_length=self.query_maxlen - 1)
        d_tokens = self.tok(docs, padding='longest', truncation='longest_first',
                            return_tensors='pt', max_length=self.doc_maxlen - 1)

        q_ids = push_sep_to_end(q_tokens.input_ids, self.special_tokens, is_cross=True)
        q_mask = q_tokens.attention_mask
        d_ids = d_tokens.input_ids[:, 1:]
        d_maks = d_tokens.attention_mask[:, 1:]

        return q_ids, q_mask, d_ids, d_maks

    @staticmethod
    def concat_qd(q_ids, q_mask, d_ids, d_mask):
        qd_ids = torch.cat((q_ids, d_ids), dim=1)
        qd_mask = torch.cat((torch.ones_like(q_mask), d_mask), dim=1)

        return qd_ids, qd_mask
