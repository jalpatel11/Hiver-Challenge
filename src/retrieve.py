"""
Plain TF-IDF cosine retriever over history.jsonl. Given a new email, returns
the closest past emails plus whatever replies were actually sent to them -
those become the generator's few-shot examples.

Went with lexical TF-IDF over embeddings mainly because the corpus is small
(200 emails) and same-topic threads tend to share enough vocabulary that
overlap finds a decent neighbor without any extra dependency or network
call. Would probably switch to embeddings if this corpus grew a lot or the
emails got more paraphrase-heavy; topk() is the seam where that swap happens.
"""
import math
import re
from collections import Counter


def _tok(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())


class TfidfRetriever:
    def __init__(self, docs: list):
        # docs: list of ticket dicts with id / incoming_email / reference_reply
        self.docs = docs
        toks = [_tok(d["incoming_email"]) for d in docs]
        df = Counter()
        for t in toks:
            for w in set(t):
                df[w] += 1
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self._default_idf = math.log((n + 1) / 1) + 1
        self.vecs = [self._vec(t) for t in toks]

    def _vec(self, toks):
        if not toks:
            return {}
        tf = Counter(toks)
        return {w: (c / len(toks)) * self.idf.get(w, self._default_idf)
                for w, c in tf.items()}

    @staticmethod
    def _cos(a, b):
        if not a or not b:
            return 0.0
        num = sum(a[w] * b[w] for w in (a.keys() & b.keys()))
        na = math.sqrt(sum(x * x for x in a.values()))
        nb = math.sqrt(sum(x * x for x in b.values()))
        return num / (na * nb) if na and nb else 0.0

    def topk(self, query_email: str, k: int = 3, exclude_id: str = None):
        qv = self._vec(_tok(query_email))
        scored = []
        for d, v in zip(self.docs, self.vecs):
            if exclude_id and d["id"] == exclude_id:
                continue
            scored.append((round(self._cos(qv, v), 4), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
