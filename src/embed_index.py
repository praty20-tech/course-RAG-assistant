# src/embed_index.py
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MODEL_NAME = "all-MiniLM-L6-v2"  # SBERT

def build_index(chunks_path="outputs/chunks.json", index_dir="outputs/index", model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    d = embeddings.shape[1]

    # create FAISS index (inner product because we normalized embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # save index and metadata
    faiss.write_index(index, f"{index_dir}/faiss.index")
    np.save(f"{index_dir}/emb_ids.npy", np.array(ids))
    print(f"Saved FAISS index -> {index_dir}")
    return index_dir

if __name__ == "__main__":
    build_index()
