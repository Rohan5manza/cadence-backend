"""
Stage 9: Build persistent usearch index from embeddings.npy

Saves checkpoints every 500k papers so a crash doesn't lose all progress.
Resume by just re-running — it picks up from the last checkpoint.
"""
import json
import os
import numpy as np
from usearch.index import Index
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDINGS       = "embeddings.npy"
PAPER_IDS        = "paper_ids.json"
OUTPUT           = "cadence.usearch"
CHECKPOINT       = "cadence_checkpoint.usearch"
CHECKPOINT_META  = "cadence_checkpoint.json"
CHUNK_SIZE       = 100_000
SAVE_EVERY       = 500_000   # save checkpoint every 500k papers
# ─────────────────────────────────────────────────────────────────────────────

print("Loading paper IDs...")
with open(PAPER_IDS) as f:
    paper_id_list = json.load(f)
total = len(paper_id_list)
print(f"Total papers: {total:,}")

print("Loading embeddings (memory-mapped)...")
emb = np.load(EMBEDDINGS, mmap_mode="r")
print(f"Embedding shape: {emb.shape}")

# ── Resume support ────────────────────────────────────────────────────────────
start_from = 0

if os.path.exists(CHECKPOINT) and os.path.exists(CHECKPOINT_META):
    with open(CHECKPOINT_META) as f:
        meta = json.load(f)
    start_from = meta.get("indexed_up_to", 0)
    print(f"Resuming from checkpoint at {start_from:,} papers")
    index = Index.restore(CHECKPOINT)
    print(f"Checkpoint loaded ✓")
else:
    print("No checkpoint found — starting fresh")
    index = Index(ndim=768, metric="cos")

# ── Build ─────────────────────────────────────────────────────────────────────
print(f"\nIndexing papers {start_from:,} → {total:,}...")

for start in tqdm(range(start_from, total, CHUNK_SIZE), desc="Indexing"):
    end = min(start + CHUNK_SIZE, total)

    index.add(
        np.arange(start, end, dtype=np.int64),
        emb[start:end].astype(np.float32),
    )

    # Save checkpoint every SAVE_EVERY papers
    if end % SAVE_EVERY < CHUNK_SIZE or end == total:
        print(f"\n  Saving checkpoint at {end:,}/{total:,}...")
        index.save(CHECKPOINT)
        with open(CHECKPOINT_META, "w") as f:
            json.dump({"indexed_up_to": end}, f)
        print(f"  Checkpoint saved ✓")

# ── Final save ────────────────────────────────────────────────────────────────
print(f"\nSaving final index to {OUTPUT}...")
index.save(OUTPUT)

# Clean up checkpoint files
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
if os.path.exists(CHECKPOINT_META):
    os.remove(CHECKPOINT_META)

print(f"\nDone.")
print(f"Index saved   : {OUTPUT}")
print(f"Papers indexed: {len(index):,}")

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nRunning sanity check...")
test = Index.restore(OUTPUT)
results = test.search(emb[0].astype(np.float32), 5)
print(f"Top 5 neighbors of paper 0 ({paper_id_list[0]}):")
for match in results:
    pid = paper_id_list[int(match.key)]
    print(f"  [{match.distance:.4f}] paper_id: {pid}")
print("\nSanity check passed ✓")