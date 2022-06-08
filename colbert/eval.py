"""
    Evaluate MS MARCO Passages ranking.
"""
import deepspeed
import ujson

import pandas as pd
import numpy as np

from argparse import ArgumentParser

import pyterrier as pt
from pathlib import Path
from pyterrier.measures import RR, nDCG, AP, NumRet, R, P

import os

import random
import torch
from colbert.utils.utils import print_message

from colbert.modeling.inference import ModelInference

from colbert.evaluation.slow import slow_rerank

from colbert.evaluation.loaders import load_colbert, load_qrels, load_queries, load_collection, load_topK_pids, \
    load_topK


def get_ranking_scores(args):
    args.colbert, args.checkpoint = load_colbert(args)
    args.qrels, args.qprel_scores = load_qrels(args.qrels)

    if args.collection or args.queries:
        assert args.collection and args.queries

        args.queries = load_queries(args.queries)
        args.collection = load_collection(args.collection)
        args.topK_pids, args.qrels = load_topK_pids(args.topK, args.qrels)

    else:
        args.queries, args.topK_docs, args.topK_pids = load_topK(args.topK)
    args.inference = ModelInference(args.colbert, amp=args.amp)

    qrels, queries, topK_pids, qprel_scores = args.qrels, args.queries, args.topK_pids, args.qprel_scores

    depth = args.depth
    collection = args.collection
    if collection is None:
        topK_docs = args.topK_docs

    def qid2passages(qid):
        if collection is not None:
            return [collection[pid] for pid in topK_pids[qid][:depth]]
        else:
            return topK_docs[qid][:depth]

    with torch.no_grad():
        keys = sorted(list(queries.keys()))
        random.shuffle(keys)

        ranking_output_path = "rank_score.tsv"
        rank_output_path = Path(ranking_output_path)
        if rank_output_path.is_file():
            rank_output_path.unlink()

        for query_idx, qid in enumerate(keys):
            query = queries[qid]

            print_message(query_idx, qid, query, '\n')

            if qrels and len(set.intersection(set(qrels[qid]), set(topK_pids[qid]))) == 0:
                continue

            ranking = slow_rerank(args, query, topK_pids[qid], qid2passages(qid))
            ranking = sorted(ranking, key=lambda row: row[0], reverse=True)
            score_pid_df = pd.DataFrame(
                list(enumerate(map(lambda row: row[1], ranking))), columns=["ranking", "pid"]
            )
            score_pid_df['qid'] = qid
            score_pid_df = score_pid_df[['qid', 'pid', 'ranking']]

            # output_path = 'my_csv.csv'
            score_pid_df.to_csv(ranking_output_path,
                                sep="\t", mode='a', index=False,
                                header=False)


def load_qrels_eval(qrels_path):
    if qrels_path is None:
        return None

    print("#> Loading qrels from", qrels_path, "...")

    qid_list = []
    pid_list = []
    rel_list = []
    with open(qrels_path, mode='r', encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            qid, _, pid, rel = line.strip().split()
            qid, pid, rel = map(int, (qid, pid, rel))
            qid_list.append(str(qid))
            pid_list.append(str(pid))
            rel_list.append(rel)
    qrels = pd.DataFrame({
        'qid': qid_list,
        'docno': pid_list,
        'label': np.array(rel_list, dtype=np.int64),
    })
    return qrels


def load_ranking_eval(path, ):
    print("#> Loading ranking from", path, "...")

    qid_list = []
    pid_list = []
    rank_list = []
    score_list = []

    with open(path, mode='r', encoding="utf-8") as f:

        if path.endswith('.jsonl'):
            for line_idx, line in enumerate(f):
                qid, pids = line.strip().split('\t')
                pids = ujson.loads(pids)
                for rank, pid in enumerate(pids):
                    pid = str(pid)
                    qid_list.append(qid)
                    pid_list.append(pid)
                    rank_list.append(rank + 1)
                    score_list.append(1000 - float(rank + 1))

        elif path.endswith('.tsv') or path.endswith('.txt'):
            for line_idx, line in enumerate(f):
                qid, pid, rank, *_ = line.strip().split('\t')

                qid_list.append(qid)
                pid_list.append(pid)
                rank_list.append(int(rank))
                score_list.append(1000 - float(rank))

    ranking = pd.DataFrame({
        'qid': qid_list,
        'docno': pid_list,
        'rank': np.array(rank_list, dtype=np.int64),
        'score': np.array(score_list, dtype=np.float64),
    })
    return ranking


def main(args):
    qrels_trec = load_qrels_eval(args.qrels)
    ranking_trec = load_ranking_eval(args.ranking)

    if not pt.started():
        pt.init()

    thr = args.binarization_point
    eval = pt.Utils.evaluate(
        res=ranking_trec,
        qrels=qrels_trec,
        metrics=[
            RR(rel=thr) @ 10, RR(rel=thr) @ 50, RR(rel=thr) @ 100, RR(rel=thr) @ 200,
            nDCG @ 10, nDCG @ 50, nDCG @ 100, nDCG @ 200,
            R(rel=thr) @ 1000, AP(rel=thr) @ 1000,
            NumRet, "num_q",
        ],
        # These measures are from "https://github.com/terrierteam/ir_measures/tree/f6b5dc62fd80f9e4ca5678e7fc82f6e8173a800d/ir_measures/measures"
    )

    print(f'#> Binarization point: {thr}')
    if thr == 1:
        print(f"#> MRR@10 = {eval['RR@10']}")
        print(f"#> MRR@50 = {eval['RR@50']}")
        print(f"#> MRR@100 = {eval['RR@100']}")
        print(f"#> MRR@200 = {eval['RR@200']}")
    else:
        print(f"#> MRR@10 = {eval['RR(rel=2)@10']}")
        print(f"#> MRR@50 = {eval['RR(rel=2)@50']}")
        print(f"#> MRR@100 = {eval['RR(rel=2)@100']}")
        print(f"#> MRR@200 = {eval['RR(rel=2)@200']}")
    print(f"#> NDCG@10 = {eval['nDCG@10']}")
    print(f"#> NDCG@50 = {eval['nDCG@50']}")
    print(f"#> NDCG@100 = {eval['nDCG@100']}")
    print(f"#> NDCG@200 = {eval['nDCG@200']}")
    if thr == 1:
        print(f"#> Recall@1000 = {eval['R@1000']}")
        print(f"#> MAP@1000 = {eval['AP@1000']}")
    else:
        print(f"#> Recall@1000 = {eval['R(rel=2)@1000']}")
        print(f"#> MAP@1000 = {eval['AP(rel=2)@1000']}")

    if args.annotate:
        args.output = f'{args.ranking}.metrics'
        with open(args.output, 'w', encoding='utf-8') as outfile:
            ujson.dump(eval, outfile, indent=4)
            outfile.write('\n')
            print(f'\n\noutput: \t\t{args.output}')

    if args.per_query_annotate:
        # ranking_trec = load_ranking(args.ranking)

        args.perquery_output = f'{args.ranking}.per_query.metrics'
        perquery_eval = pt.Utils.evaluate(
            res=ranking_trec,
            qrels=qrels_trec,
            metrics=[
                RR(rel=1) @ 10, RR(rel=2) @ 50, RR(rel=2) @ 100, RR(rel=2) @ 200,
                nDCG @ 10, nDCG @ 50, nDCG @ 100, nDCG @ 200,
                R(rel=1) @ 1000, AP(rel=1) @ 1000,
                NumRet,
                # "num_q",
            ],
            perquery=True,
        )

        ordered_qid = []
        with open(args.queries, 'r', encoding='utf-8') as infile:
            for line_idx, line in enumerate(infile):
                qid = line.strip().split('\t')[0]
                ordered_qid.append(qid)

        with open(args.perquery_output, 'w', encoding='utf-8') as outfile:
            for qid in ordered_qid:
                outfile.write(f'{qid}\t{ujson.dumps(perquery_eval[qid])}\n')
        print(f'\n\nperquery_output: \t\t{args.perquery_output}')

    print('\n\n')


if __name__ == "__main__":
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--qrels', dest='qrels', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', default="rank_score.tsv", required=True, type=str)
    parser.add_argument('--binarization_point', dest='binarization_point', type=int, required=True,
                        help="1 for MSMARCO and 2 for TREC-DL 2019/2020.")

    parser.add_argument('--annotate', dest='annotate', default=False, action='store_true')
    parser.add_argument('--per_query_annotate', dest='per_query_annotate', default=False, action='store_true')
    parser.add_argument('--queries', dest='queries', help="/path/to/queries.[].tsv")

    parser.add_argument('--ckpt', dest='checkpoint')
    parser.add_argument('--amp', dest='amp', default=False, action='store_true')
    parser.add_argument('--mask-punctuation', dest='mask_punctuation', default=False, action='store_true')
    parser.add_argument('--bsize', dest='bsize', default=128, type=int)

    parser.add_argument('--collection', dest='collection', default=None)
    parser.add_argument('--topk', dest='topK', required=True)

    parser.add_argument('--pe_sampling_size', dest='pe_sampling_size', default=5, type=int)

    parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
    parser.add_argument('--dim', dest='dim', default=128, type=int)
    parser.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
    parser.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)

    parser.add_argument('--rank', dest='rank', default=0, type=int)

    parser.add_argument('--depth', dest='depth', required=False, default=None, type=int)

    args = parser.parse_args()

    if args.per_query_annotate:
        assert os.path.exists(args.queries)
    # get_ranking_scores(args)
    main(args)
