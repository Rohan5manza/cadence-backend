"""
Stage 8: Generate embeddings for all 2.28M papers.
Saves incrementally in chunks to avoid RAM overflow.
"""
import json
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR    = "./specter2-finetuned"
PAPERS_FILE  = "papers_merged.jsonl"
OUTPUT_EMB   = "embeddings.npy"
OUTPUT_IDS   = "paper_ids.json"
BATCH_SIZE   = 256
CHUNK_SIZE   = 100_000   # save to disk every 100k papers
MAX_SEQ_LEN  = 256
# ─────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
if device == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model from {MODEL_DIR}...")
model = SentenceTransformer(MODEL_DIR)
model.max_seq_length = MAX_SEQ_LEN
model = model.to(device)
print("Model loaded.")

# ── Load papers ───────────────────────────────────────────────────────────────
print(f"\nLoading papers from {PAPERS_FILE}...")
paper_ids = []
texts     = []

with open(PAPERS_FILE) as f:
    for line in tqdm(f, desc="Loading"):
        p        = json.loads(line)
        title    = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        paper_ids.append(str(p["paper_id"]))
        texts.append(title + " " + abstract)

total = len(texts)
print(f"Loaded {total:,} papers")

# ── Generate and save in chunks ───────────────────────────────────────────────
print(f"\nGenerating embeddings in chunks of {CHUNK_SIZE:,}...")

# Use memory-mapped file to avoid holding everything in RAM
emb_dim  = 768
emb_file = np.lib.format.open_memmap(
    OUTPUT_EMB,
    mode="w+",
    dtype=np.float32,
    shape=(total, emb_dim)
)

for chunk_start in range(0, total, CHUNK_SIZE):
    chunk_end   = min(chunk_start + CHUNK_SIZE, total)
    chunk_texts = texts[chunk_start:chunk_end]

    print(f"\nChunk {chunk_start//CHUNK_SIZE + 1}: papers {chunk_start:,}–{chunk_end:,}")

    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=BATCH_SIZE,
        device=device,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Write directly to memory-mapped file
    emb_file[chunk_start:chunk_end] = chunk_embeddings.astype(np.float32)
    emb_file.flush()  # force write to disk

    print(f"  Saved chunk {chunk_start//CHUNK_SIZE + 1} to disk")

    # Free RAM
    del chunk_embeddings

# ── Save paper IDs ────────────────────────────────────────────────────────────
print(f"\nSaving paper IDs to {OUTPUT_IDS}...")
with open(OUTPUT_IDS, "w") as f:
    json.dump(paper_ids, f)

print(f"\nDone.")
print(f"Embeddings : {OUTPUT_EMB}  ({os.path.getsize(OUTPUT_EMB)/1e9:.2f}GB)")
print(f"Paper IDs  : {OUTPUT_IDS}")
print(f"Shape      : ({total}, {emb_dim})")