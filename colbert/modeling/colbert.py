import string
import torch
import torch.nn as nn
from torch import Tensor, einsum
from torch.nn import LayerNorm
from torch.nn.functional import normalize

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE
from colbert.probemb.pie_module import MultiHeadSelfAttention
from colbert.utils.utils import print_message


class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, pe_sampling_size, dim=128,
                 similarity_metric='cosine'):

        super(ColBERT, self).__init__(config)

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

        # ============ PE ============ #
        self.n_samples = pe_sampling_size
        print_message(f">> Probabilistic Embedding Sampling Size: {self.n_samples}")
        # https://github.com/naver-ai/pcme/blob/bc14001e9d67b28e3ab989ee367be3f98b999a1f/models/uncertainty_module.py#L45
        self.attention = MultiHeadSelfAttention(dim, dim, dim)
        # self.attention = MultiheadAttention(num_heads=1, embed_dim=dim)
        self.mu_linear = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.mu_ln = LayerNorm(dim)
        self.mu_l2 = lambda target: normalize(target, p=2, dim=-1)
        # ============================ #
        self.init_weights()

    def forward(self, Q, D):
        return self.score(self.query(*Q), self.doc(*D))

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        # D: batch_size, Document_token_size
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        # D: batch_size, Document_token_size, 768(vocab size)
        D = self.linear(D)
        # D: batch_size, Document_token_size, 128(hidden size set on linear)

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def get_doc_mu(self, D: Tensor, attn_D: Tensor):
        # https://github.com/naver-ai/pcme/blob/587aa0de710f3f40eac42cd97657c1a9dcfc6ebc/models/pie_model.py#L62
        soft_attn = self.sigmoid(self.mu_linear(attn_D))

        residual_merged = D + soft_attn
        ln = self.mu_ln(residual_merged)
        l2 = self.mu_l2(ln)

        return l2

    @staticmethod
    def get_doc_logsigma(D: Tensor, attn_D: Tensor):
        residual_merged = D + attn_D
        return residual_merged

    def sample_gaussian_tensors(self, mu: Tensor, logsigma: Tensor):
        eps = torch.randn(mu.size(0), self.n_samples, mu.size(1), mu.size(2), dtype=mu.dtype, device=mu.device)
        samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(mu.unsqueeze(1))
        # l2 정규화... (아마 여태 성능 안 나왔던건 샘플의 정규화가 안되어 있었어서...?)
        samples = normalize(samples, p=2, dim=-1)
        return samples

    def score(self, Q, D):
        # ============ PE ============ #
        D_res, D_attn = self.attention(D)
        # D_attn = self.attention(D)
        mu = self.get_doc_mu(D, D_attn)
        logsigma = self.get_doc_logsigma(D, D_attn)
        Ds = self.sample_gaussian_tensors(mu, logsigma)
        Qs = Q.unsqueeze(1).repeat(1, self.n_samples, 1, 1)
        # ============================ #
        if self.similarity_metric == 'cosine':
            # this metric is used for test
            # Q = (1, 32, 128)
            # D.permute = (1000, 180, 128) -> (1000, 128, 180)
            # Q @ D.permute = (1000, 32, 180)
            # Qs = [1, 5, 32, 128]
            # Ds = [1000, 5, 180, 128] -> permute(0, 1, 3, 2) = [1000, 5, 128, 180]
            #
            # .amax((-1, -3)).sum(-1)
            return einsum("ibqv,jbdv->jbqd", Qs, Ds)\
                .reshape(Ds.shape[0], Qs.shape[2], -1)\
                .amax(-1) \
                .sum(-1)
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

        return (-1.0 * ((Qs.unsqueeze(3) - Ds.unsqueeze(2)) ** 2).sum(-1)) \
            .reshape(Qs.shape[0], Qs.shape[2], -1) \
            .amax(-1) \
            .sum(-1)

        # .amax((-1, -3)) \
        # .reshape(Qs.shape[0], Qs.shape[1] * Qs.shape[2])\

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
