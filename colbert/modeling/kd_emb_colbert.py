import string
import torch
import torch.nn as nn
from torch import Tensor, einsum
from torch.nn import LayerNorm
from torch.nn.functional import normalize

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE


class KDEmbedColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, pe_sampling_size, dim=128,
                 similarity_metric='cosine'):
        super(KDEmbedColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q, D):
        hQ, lQ = self.query(*Q)
        hD, lD = self.doc(*D)
        return (hQ, hD), self.student_score(lQ, lD)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        org_Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        red_Q = self.linear(org_Q)

        org_Q = torch.nn.functional.normalize(org_Q, p=2, dim=2)
        red_Q = torch.nn.functional.normalize(red_Q, p=2, dim=2)
        return org_Q, red_Q

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()

        # D: batch_size, Document_token_size
        org_D = self.bert(input_ids, attention_mask=attention_mask)[0]
        # D: batch_size, Document_token_size, 768(vocab size)
        red_D = self.linear(org_D)
        # D: batch_size, Document_token_size, 128(hidden size set on linear)
        org_D *= mask
        red_D *= mask

        org_D = torch.nn.functional.normalize(org_D, p=2, dim=2)
        red_D = torch.nn.functional.normalize(red_D, p=2, dim=2)

        if not keep_dims:
            org_D, mask = org_D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            org_D = [d[mask[idx]] for idx, d in enumerate(org_D)]

            red_D, mask = red_D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            red_D = [d[mask[idx]] for idx, d in enumerate(red_D)]

        return org_D, red_D

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

    def distill_score(self, Q: Tensor, D: Tensor):
        ...

    def student_score(self, Q: Tensor, D: Tensor):
        if self.similarity_metric == 'cosine':
            # this metric is used for test
            # Q = (1, 32, 128)
            # D.permute = (1000, 180, 128) -> (1000, 128, 180)
            # Q @ D.permute = (1000, 32, 180)
            # Qs = [1, 5, 32, 128]
            # Ds = [1000, 5, 180, 128] -> permute(0, 1, 3, 2) = [1000, 5, 128, 180]
            #
            # .amax((-1, -3)).sum(-1)

            # return einsum("ibqv,jbdv->jbqd", Qs, Ds)\
            #     .reshape(Ds.shape[0], Qs.shape[2], -1)\
            #     .amax(-1) \
            #     .sum(-1)

            return einsum("iqv,jdv->jqd", Q, D).amax(-1).sum(-1)

            # .reshape(Qs.shape[0], Qs.shape[1] * Qs.shape[2]) \

        # Training
        assert self.similarity_metric == 'l2'
        # MaxSim?
        # Objective: max( similarity_fn( Q, D ) )
        #   similarity_fns = {cosine, l2}
        # Inputs
        #   Q: ( batchSize, {query_tokens, vocab_size[본래 768이지만 128 리니어로 축소되어 있음.]} )
        #   D: ( batchSize, {document_tokens, vocab_size[본래 768이지만 128 리니어로 축소되어 있음.]} )
        # How?
        # * step 1:
        #   * 우선 시작에 앞서 배치 사이즈는 빼고
        #   * Q -> Q.unsqueeze(2) => {q_toks, vocab} -> {q_toks, 1, vocab}
        #   * D -> D.unsqueeze(1) => {d_toks, vocab} -> {1, d_toks, vocab}
        #   * 이렇게 만든 이유는 (q_toks - d_toks)를 해서 두 벡터간의 차를 구하기 위함.
        #   * 결과: {q_toks, d_toks, vocab}
        # * step 2: sum & max
        #   * squared-L2
        #       * sum(-1) 구하고 => {q_toks, d_toks} (q - d 토큰들에 대해 여러 vocab간의 관계중 가장 큰걸 추출)
        #   * max(-1).values 구하고 => {q_toks, } (q 에 대해 여러 d_toks 중 가장 큰 값을 추출)
        #   * 다시 sum(-1) => {1, } (q - d 의 relevance 점수)

        # update: amax 를 이용해서 PE 샘플링한 걸 반영해서 계산하게 함.
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).amax(-1).sum(-1)

        # # ============ PE ============ #
        # D_res, D_attn = self.attention(D)
        # # D_attn = self.attention(D)
        # mu = self.get_doc_mu(D, D_attn)
        # logsigma = self.get_doc_logsigma(D, D_attn)
        # Qs = Q.unsqueeze(1).repeat(1, self.n_samples, 1, 1)
        # Ds = self.sample_gaussian_tensors(mu, logsigma)
        # # ============================ #
        # return (-1.0 * ((Qs.unsqueeze(3) - Ds.unsqueeze(2)) ** 2).sum(-1)) \
        #     .reshape(Qs.shape[0], Qs.shape[2], -1) \
        #     .amax(-1) \
        #     .sum(-1)

        # .amax((-1, -3)) \
        # .reshape(Qs.shape[0], Qs.shape[1] * Qs.shape[2])\
