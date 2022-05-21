import ujson

from collections import defaultdict

from math import log

from colbert.utils.runs import Run


class Metrics:
    def __init__(self, mrr_depths: set, recall_depths: set, success_depths: set, ndcg_depths: set, total_queries=None):
        self.results = {}
        self.mrr_sums = {depth: 0.0 for depth in mrr_depths}
        self.recall_sums = {depth: 0.0 for depth in recall_depths}
        self.success_sums = {depth: 0.0 for depth in success_depths}
        self.ndcg_sums = {depth: 0.0 for depth in ndcg_depths}
        self.total_queries = total_queries

        self.max_query_idx = -1
        self.num_queries_added = 0

    def add(self, query_idx, query_key, ranking, gold_positives, qprel_score):
        self.num_queries_added += 1

        assert query_key not in self.results
        assert len(self.results) <= query_idx
        assert len(set(gold_positives)) == len(gold_positives)
        assert len(set([pid for _, pid, _ in ranking])) == len(ranking)

        self.results[query_key] = ranking

        positives = [i for i, (_, pid, _) in enumerate(ranking) if pid in gold_positives]

        pids_ranked = list(map(lambda r: r[1], ranking))
        pids_ranked_score = list(map(lambda pid: qprel_score[pid] if pid in gold_positives else 0, pids_ranked))

        if len(positives) == 0:
            return

        for depth in self.mrr_sums:
            first_positive = positives[0]
            self.mrr_sums[depth] += (1.0 / (first_positive + 1.0)) if first_positive < depth else 0.0

        for depth in self.success_sums:
            first_positive = positives[0]
            self.success_sums[depth] += 1.0 if first_positive < depth else 0.0

        for depth in self.recall_sums:
            num_positives_up_to_depth = len([pos for pos in positives if pos < depth])
            self.recall_sums[depth] += num_positives_up_to_depth / len(gold_positives)

        for depth in self.ndcg_sums:
            scores = pids_ranked_score[:depth]
            idx_scores = list(enumerate(scores))
            dcg = sum(map(lambda item: item[1] / log(item[0] + 2, 2), idx_scores))
            rel_sort_idx_scores = sorted(idx_scores, key=lambda idx_score: idx_score[1], reverse=True)
            idcg = sum(map(lambda item: item[1] / log(item[0] + 2, 2), rel_sort_idx_scores))

            self.ndcg_sums[depth] += dcg / idcg if idcg != 0 else 0

    def print_metrics(self, query_idx):
        for depth in sorted(self.mrr_sums):
            print("MRR@" + str(depth), "=", self.mrr_sums[depth] / (query_idx + 1.0))

        for depth in sorted(self.success_sums):
            print("Success@" + str(depth), "=", self.success_sums[depth] / (query_idx + 1.0))

        for depth in sorted(self.recall_sums):
            print("Recall@" + str(depth), "=", self.recall_sums[depth] / (query_idx + 1.0))

        for depth in sorted(self.ndcg_sums):
            print("nDCG@" + str(depth), "=", self.ndcg_sums[depth] / (query_idx + 1.0))

    def log(self, query_idx):
        assert query_idx >= self.max_query_idx
        self.max_query_idx = query_idx

        Run.log_metric("ranking/max_query_idx", query_idx, query_idx)
        Run.log_metric("ranking/num_queries_added", self.num_queries_added, query_idx)

        for depth in sorted(self.mrr_sums):
            score = self.mrr_sums[depth] / (query_idx + 1.0)
            Run.log_metric("ranking/MRR." + str(depth), score, query_idx)

        for depth in sorted(self.success_sums):
            score = self.success_sums[depth] / (query_idx + 1.0)
            Run.log_metric("ranking/Success." + str(depth), score, query_idx)

        for depth in sorted(self.recall_sums):
            score = self.recall_sums[depth] / (query_idx + 1.0)
            Run.log_metric("ranking/Recall." + str(depth), score, query_idx)

        for depth in sorted(self.ndcg_sums):
            score = self.ndcg_sums[depth] / (query_idx + 1.0)
            Run.log_metric("ranking/nDCG." + str(depth), score, query_idx)

    def output_final_metrics(self, path, query_idx, num_queries):
        assert query_idx + 1 == num_queries
        assert num_queries == self.total_queries

        if self.max_query_idx < query_idx:
            self.log(query_idx)

        self.print_metrics(query_idx)

        output = defaultdict(dict)

        for depth in sorted(self.mrr_sums):
            score = self.mrr_sums[depth] / (query_idx + 1.0)
            output['mrr'][depth] = score

        for depth in sorted(self.success_sums):
            score = self.success_sums[depth] / (query_idx + 1.0)
            output['success'][depth] = score

        for depth in sorted(self.recall_sums):
            score = self.recall_sums[depth] / (query_idx + 1.0)
            output['recall'][depth] = score

        for depth in sorted(self.ndcg_sums):
            score = self.ndcg_sums[depth] / (query_idx + 1.0)
            output['ndcg'][depth] = score

        with open(path, 'w') as f:
            ujson.dump(output, f, indent=4)
            f.write('\n')


def evaluate_recall(qrels, queries, topK_pids):
    if qrels is None:
        return

    assert set(qrels.keys()) == set(queries.keys())
    recall_at_k = [len(set.intersection(set(qrels[qid]), set(topK_pids[qid]))) / max(1.0, len(qrels[qid]))
                   for qid in qrels]
    recall_at_k = sum(recall_at_k) / len(qrels)
    recall_at_k = round(recall_at_k, 3)
    print("Recall @ maximum depth =", recall_at_k)

# TODO: If implicit qrels are used (for re-ranking), warn if a recall metric is requested + add an asterisk to output.
