"""BM25 and TF-IDF as array expressions, top-k via ``np.argpartition``.

The Lab 3 loop in ``ranked_search_python`` is the reference. This module
must return the same document order when scores are not tied. Ties break
by document id; the Python heap breaks ties by insertion order, which only
matters when two scores compare equal.
"""

from __future__ import annotations

import math

import numpy as np

from findex.index import Index
from findex.query import match_doc_ids
from findex.rank import BM25, SearchResult, TfIdf, _pack_results, _positive_terms


def argpartition_top(
    scores: np.ndarray,
    candidates: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """Highest ``k`` among ``candidates``. ``O(n)`` select, then a sort of ``k``."""
    if k <= 0 or candidates.size == 0:
        return []
    k_eff = min(k, int(candidates.size))
    vals = scores[candidates]
    chosen = candidates
    if k_eff < candidates.size:
        pick = np.argpartition(vals, -k_eff)[-k_eff:]
        chosen = candidates[pick]
        vals = vals[pick]
    order = np.lexsort((chosen, -vals))
    return [(int(chosen[i]), float(vals[i])) for i in order]


def _accumulate(
    scores: np.ndarray,
    touched: np.ndarray,
    index: Index,
    term: str,
    allowed: np.ndarray,
    scorer: BM25 | TfIdf,
) -> None:
    pair = index.posting_arrays(term)
    if pair is None:
        return
    doc_ids, tfs = pair
    mask = allowed[doc_ids]
    if not bool(mask.any()):
        return
    doc_ids = doc_ids[mask]
    tfs = tfs[mask]
    df = index.df(term)
    n_docs = index.num_docs
    if df <= 0 or n_docs <= 0 or doc_ids.size == 0:
        return
    # Mark even a zero contribution so a term present in every document
    # (IDF 0) still occupies a rank, the way the Lab 3 dict does.
    touched[doc_ids] = True
    tf = tfs.astype(np.float64, copy=False)
    if isinstance(scorer, TfIdf):
        idf = math.log(n_docs / df)
        scores[doc_ids] += tf * idf
        return
    avgdl = index.avg_doc_length
    if avgdl <= 0:
        return
    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
    lengths = index.doc_len_array[doc_ids].astype(np.float64, copy=False)
    k1 = float(scorer.k1)
    b = float(scorer.b)
    denom = tf + k1 * (1.0 - b + b * (lengths / avgdl))
    scores[doc_ids] += idf * (tf * (k1 + 1.0)) / denom


def ranked_search_numpy(
    index: Index,
    query: str,
    *,
    scorer: BM25 | TfIdf | None = None,
    k: int = 10,
    snippets: bool = True,
) -> list[SearchResult]:
    """Score every matching document with ufuncs, then ``np.argpartition``."""
    if scorer is None:
        scorer = BM25()
    if not query.strip():
        return []
    doc_ids = match_doc_ids(index, query)
    if not doc_ids:
        return []
    terms = _positive_terms(query)
    if not terms:
        top = [(doc_id, 0.0) for doc_id in doc_ids[:k]]
        return _pack_results(index, top, terms, snippets=snippets)

    width = int(index.doc_len_array.shape[0])
    scores = np.zeros(width, dtype=np.float64)
    touched = np.zeros(width, dtype=np.bool_)
    allowed = np.zeros(width, dtype=np.bool_)
    allowed[np.asarray(doc_ids, dtype=np.int32)] = True
    for term in terms:
        _accumulate(scores, touched, index, term, allowed, scorer)

    top = argpartition_top(scores, np.flatnonzero(touched), k)
    return _pack_results(index, top, terms, snippets=snippets)
