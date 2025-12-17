# src/retriever.py
"""
Robust Retriever: BM25 + Dense (SBERT) + hybrid merge.

Improvements vs simple script:
- Safe startup checks (index + ids length consistency)
- Graceful fallback tokenizer if NLTK punkt not available
- Avoid downloading NLTK data at import time
- Defensive handling for missing ids or index -1 results
- Configurable options (top_k, bm25_k, dense_k, remove_stopwords, stem)
- Clear return schema: list[{"id","score","source","text","method"}]
"""
from pathlib import Path
import json
import logging
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np
import faiss

# Sentence-transformers and BM25
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# NLTK tokenization (optional). We do NOT download punkt here to avoid side effects.
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    HAVE_NLTK = True
except Exception:
    HAVE_NLTK = False

# Simple fallback tokenizer
def simple_tokenize(text: str):
    return text.lower().split()

# Lightweight stemmer option (Porter)
try:
    from nltk.stem.porter import PorterStemmer
    HAVE_STEMMER = True
except Exception:
    HAVE_STEMMER = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(
        self,
        chunks_path: str = "outputs/chunks.json",
        index_dir: str = "outputs/index",
        model_name: str = MODEL_NAME,
        remove_stopwords: bool = False,
        use_stemming: bool = False,
    ):
        # load chunks
        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            raise FileNotFoundError(f"chunks.json not found at {chunks_path.resolve()}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.id2chunk = {c["id"]: c for c in self.chunks}

        # tokenization setup (defer nltk download; fallback to simple tokenizer)
        self.remove_stopwords = remove_stopwords and HAVE_NLTK
        if self.remove_stopwords:
            try:
                self.stop_words = set(stopwords.words("english"))
            except Exception:
                logger.warning("NLTK stopwords not available — continuing without stopword removal.")
                self.remove_stopwords = False

        self.use_stemming = use_stemming and HAVE_STEMMER
        if use_stemming and not HAVE_STEMMER:
            logger.warning("PorterStemmer not available; proceeding without stemming.")
            self.use_stemming = False
        self.stemmer = PorterStemmer() if self.use_stemming else None

        # build BM25 corpus tokens (tokenize once)
        token_func = word_tokenize if HAVE_NLTK else simple_tokenize
        tokenized = []
        for c in self.chunks:
            toks = token_func(c["text"].lower())
            if self.remove_stopwords:
                toks = [t for t in toks if t not in self.stop_words]
            if self.use_stemming:
                toks = [self.stemmer.stem(t) for t in toks]
            tokenized.append(toks)
        self.bm25 = BM25Okapi(tokenized)

        # load dense (SBERT) model + FAISS index + ids
        self.model = SentenceTransformer(model_name)
        index_path = Path(index_dir) / "faiss.index"
        ids_path = Path(index_dir) / "emb_ids.npy"
        if not index_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"FAISS index or emb_ids.npy missing in {index_dir}")
        self.index = faiss.read_index(str(index_path))
        self.ids = list(np.load(str(ids_path), allow_pickle=True))

        # sanity check: index length vs ids length
        if self.index.ntotal != len(self.ids):
            logger.warning(
                f"Index.ntotal ({self.index.ntotal}) != len(ids) ({len(self.ids)}). "
                "This may indicate stale index or ids file."
            )

    # --------- Tokenizer helper used for BM25 query processing ----------
    def _tokenize_query(self, query: str):
        if HAVE_NLTK:
            toks = word_tokenize(query.lower())
        else:
            toks = simple_tokenize(query)
        if self.remove_stopwords:
            toks = [t for t in toks if t not in self.stop_words]
        if self.use_stemming:
            toks = [self.stemmer.stem(t) for t in toks]
        return toks

    # --------- BM25 ----------
    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        q_tok = self._tokenize_query(query)
        scores = self.bm25.get_scores(q_tok)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idxs:
            cid = self.chunks[i]["id"]
            results.append({
                "id": cid,
                "score": float(scores[i]),
                "source": self.chunks[i].get("source"),
                "text": self.chunks[i].get("text"),
                "method": "bm25"
            })
        return results

    # --------- Dense (SBERT + FAISS) ----------
    def dense_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # encode query
        q_emb = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        # guard index shape
        try:
            D, I = self.index.search(np.array([q_emb]).astype("float32"), top_k)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

        results = []
        for score, idxpos in zip(D[0], I[0]):
            if int(idxpos) < 0 or int(idxpos) >= len(self.ids):
                # faiss can return -1 for empty slots; skip
                continue
            try:
                chunk_id = self.ids[int(idxpos)]
                chunk = self.id2chunk.get(chunk_id)
                if chunk is None:
                    logger.debug(f"chunk_id {chunk_id} not found in chunks.json — skipping")
                    continue
                results.append({
                    "id": chunk_id,
                    "score": float(score),
                    "source": chunk.get("source"),
                    "text": chunk.get("text"),
                    "method": "dense"
                })
            except Exception as e:
                logger.debug(f"Error mapping idxpos {idxpos} -> id: {e}")
        return results

    # --------- Hybrid (union of BM25 + dense) ----------
    def hybrid(
        self,
        query: str,
        top_k: int = 5,
        bm25_k: int = 30,
        dense_k: int = 30,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        bm25_results = self.bm25_search(query, top_k=bm25_k)
        dense_results = self.dense_search(query, top_k=dense_k)

        rrf_scores = defaultdict(float)
        doc_store = {}

        # BM25 ranks
        for rank, item in enumerate(bm25_results):
            doc_id = item["id"]
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
            doc_store[doc_id] = item

        # Dense ranks
        for rank, item in enumerate(dense_results):
            doc_id = item["id"]
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
            doc_store[doc_id] = item

        # Sort by RRF score
        ranked = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for doc_id, score in ranked[:top_k]:
            doc = doc_store[doc_id].copy()
            doc["rrf_score"] = score
            results.append(doc)

        return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="outputs/chunks.json")
    p.add_argument("--index_dir", default="outputs/index")
    p.add_argument("--query", default="what is the history of information retrieval?")
    
    # NEW: Add flags to enable stemming and stopwords
    p.add_argument("--stopwords", action="store_true", help="Enable stopword removal")
    p.add_argument("--stemming", action="store_true", help="Enable Porter stemming")
    
    args = p.parse_args()

    # Inform user if they try to use features without NLTK installed
    if (args.stopwords or args.stemming) and not HAVE_NLTK:
        logger.warning("You requested NLTK features (--stopwords or --stemming) but NLTK is not available/working.")
        logger.warning("Run: pip install nltk (and download data) to fix.")

    # Pass the flags to the Retriever
    r = Retriever(
        chunks_path=args.chunks, 
        index_dir=args.index_dir,
        remove_stopwords=args.stopwords, 
        use_stemming=args.stemming      
    )

    print(f"DEBUG: Stopwords={r.remove_stopwords}, Stemming={r.use_stemming}")
    print("-" * 30)
    print("BM25 sample:", r.bm25_search(args.query, top_k=3))
    print("Dense sample:", r.dense_search(args.query, top_k=3))
    print("Hybrid sample:", r.hybrid(args.query, top_k=5))
