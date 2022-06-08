import os

def slow_rerank(args, query, pids, passages):
    colbert = args.colbert
    inference = args.inference

    Q = inference.queryFromText([query])

    D_ = inference.docFromText(passages, bsize=args.bsize)

    # if args.pe_sampling_size >= 1:
    #     # ============ PE ============ #
    #     D_res, D_attn = colbert.attention(D_)
    #     # D_attn = self.attention(D)
    #     mu = colbert.get_doc_mu(D_, D_attn)
    #     logsigma = colbert.get_doc_logsigma(D_, D_attn)
    #     D_ = colbert.sample_gaussian_tensors(mu, logsigma)
    #     Q = Q.unsqueeze(1).repeat(1, colbert.n_samples, 1, 1)
    #     # ============================ #

    scores = colbert.score(Q, D_).cpu()

    scores = scores.sort(descending=True)
    ranked = scores.indices.tolist()

    ranked_scores = scores.values.tolist()
    ranked_pids = [pids[position] for position in ranked]
    ranked_passages = [passages[position] for position in ranked]

    assert len(ranked_pids) == len(set(ranked_pids))

    return list(zip(ranked_scores, ranked_pids, ranked_passages))
