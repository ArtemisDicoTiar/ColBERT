import json
from typing import List, Tuple, Dict
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer

from colbert.modeling.kd_emb_tokenization import DocTokenizer, QueryTokenizer, tensorize_triples
from colbert.modeling.kd_emb_tokenization.cross_tokenization import CrossTokenizer
from colbert.utils.utils import print_message
from colbert.utils.runs import Run


class LazyBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_col_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_col_tokenizer = DocTokenizer(args.doc_maxlen)
        self.cross_tokenizer = CrossTokenizer(args.query_maxlen, args.doc_maxlen)
        self.tensorize_triples = partial(tensorize_triples,
                                         self.query_col_tokenizer, self.doc_col_tokenizer,
                                         self.cross_tokenizer)
        self.position = 0

        self.triples = self._load_triples(args.triples, rank, nranks)
        self.queries = self._load_queries(args.queries)
        self.collection = self._load_collection(args.collection)

    @staticmethod
    def _load_triples(path: str, rank, nranks) -> List[Tuple[str, str, str, float]]:
        print_message("#> Loading triples...")

        triples = []

        with open(path) as f:
            total = len(f.readlines())
            f.seek(0)
            for line_idx, line in tqdm(enumerate(f), total=total, desc='Loading Triples'):
                if line_idx % nranks == rank:
                    qid, pos, neg, diff_score = line.strip().split('\t')
                    triples.append((str(qid), str(pos), str(neg), float(diff_score)))

        return triples

    @staticmethod
    def _load_queries(path: str) -> Dict[str, str]:
        print_message("#> Loading queries...")

        queries = {}

        with open(path) as f:
            total = len(f.readlines())
            f.seek(0)
            for line in tqdm(f, total=total, desc='Loading Queries'):
                result = json.loads(line)
                qid, query = result["_id"], result['text']
                queries[str(qid)] = query

        return queries

    @staticmethod
    def _load_collection(path: str) -> Dict[str, str]:
        print_message("#> Loading collection...")

        collections = {}

        with open(path, 'r', encoding='utf-8') as f:
            total = len(f.readlines())
            f.seek(0)
            for line in tqdm(f, total=total, desc='Loading Collections'):
                result = json.loads(line)
                pid, passage = result["_id"], f"{result['title']} {result['text']}"
                collections[str(pid)] = passage

        return collections

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        queries, positives, negatives, scores = [], [], [], []

        for position in range(offset, endpos):
            query, pos, neg, diff_score = self.triples[position]
            query, pos, neg = self.queries[query], self.collection[pos], self.collection[neg]

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)
            scores.append(diff_score)

        return self.collate(queries, positives, negatives, scores)

    def collate(self,
                queries: List[str],
                positives: List[str], negatives: List[str],
                scores: List[float]):
        assert len(queries) == len(positives) == len(negatives) == len(scores) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, scores, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
        self.position = intended_batch_size * batch_idx
