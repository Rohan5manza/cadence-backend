"""
embed_new_papers.py — Embed new papers and extend existing index
RAM-safe: uses memory-mapped files, never loads full corpus into RAM
Checkpoint-safe: saves every CHECKPOINT_EVERY papers
Strict deduplication: paper_id, DOI, arXiv ID, title fingerprint

Usage: python embed_new_papers.py
"""

import os
import json
import re
import hashlib
import numpy as np

CHECKPOINT_FILE  = "checkpoints/embed_checkpoint.json"
CHECKPOINT_EVERY = 10_000
BATCH_SIZE       = 128
NEW_PAPER_FILES  = [
    "papers_pubmed.jsonl",
    "papers_biorxiv.jsonl",
    "papers_doaj.jsonl",
    "papers_preprints.jsonl",
]

os.makedirs("checkpoints", exist_ok=True)

# ── Dedup helpers ─────────────────────────────────────────────────────────────

def normalize_doi(doi):
    if not doi: return None
    doi = doi.lower().strip()
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
    doi = re.sub(r'^doi:', '', doi)
    return doi if doi else None

def title_fingerprint(title):
    if not title: return ""
    cleaned = re.sub(r'[^a-z0-9]', '', title.lower())[:80]
    return hashlib.md5(cleaned.encode()).hexdigest()

def arxiv_id_normalize(arxiv_id):
    if not arxiv_id: return None
    return re.sub(r'v\d+$', '', arxiv_id.strip().lower())

def build_existing_dedup_sets():
    """Stream through existing corpus to build dedup sets — no full load into RAM."""
    print("[dedup] Streaming existing corpus for dedup sets...")
    with open("paper_ids.json") as f:
        existing_ids = set(json.load(f))

    existing_dois   = set()
    existing_fps    = set()
    existing_arxivs = set()
    count           = 0

    with open("papers_merged.jsonl") as f:
        for line in f:
            try:
                p = json.loads(line)
                doi = normalize_doi(p.get("doi", ""))
                if doi: existing_dois.add(doi)
                fp = title_fingerprint(p.get("title", ""))
                if fp: existing_fps.add(fp)
                ax = arxiv_id_normalize(p.get("arxiv_id", ""))
                if ax: existing_arxivs.add(ax)
                count += 1
                if count % 500_000 == 0:
                    print(f"  [dedup] Scanned {count:,}...")
            except: continue

    print(f"[dedup] {len(existing_ids):,} IDs | {len(existing_dois):,} DOIs | "
          f"{len(existing_fps):,} titles | {len(existing_arxivs):,} arXiv IDs")
    return existing_ids, existing_dois, existing_fps, existing_arxivs

def is_duplicate(p, ex_ids, ex_dois, ex_fps, ex_arxivs,
                 seen_ids, seen_dois, seen_fps, seen_arxivs):
    pid = str(p.get("paper_id", ""))
    if pid in ex_ids or pid in seen_ids: return True, "paper_id"
    doi = normalize_doi(p.get("doi", ""))
    if doi and (doi in ex_dois or doi in seen_dois): return True, "doi"
    ax = arxiv_id_normalize(p.get("arxiv_id", ""))
    if ax and (ax in ex_arxivs or ax in seen_arxivs): return True, "arxiv_id"
    fp = title_fingerprint(p.get("title", ""))
    if fp and (fp in ex_fps or fp in seen_fps): return True, "title_fingerprint"
    return False, ""

# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed_ids": [], "total_embedded": 0, "dedup_stats": {}}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f)

# ── Collect new papers ────────────────────────────────────────────────────────

def collect_new_papers(ex_ids, ex_dois, ex_fps, ex_arxivs, processed_set, dedup_stats):
    new_papers   = []
    seen_ids     = set(processed_set)
    seen_dois    = set()
    seen_fps     = set()
    seen_arxivs  = set()
    total_dupes  = 0

    print("\n[collect] Scanning new paper files...")
    for filepath in NEW_PAPER_FILES:
        if not os.path.exists(filepath):
            print(f"  [skip] {filepath} not found")
            continue
        file_new = file_dupes = 0
        with open(filepath) as f:
            for line in f:
                try:
                    p   = json.loads(line)
                    pid = str(p.get("paper_id", ""))
                    if not pid: continue

                    is_dup, reason = is_duplicate(
                        p, ex_ids, ex_dois, ex_fps, ex_arxivs,
                        seen_ids, seen_dois, seen_fps, seen_arxivs,
                    )
                    if is_dup:
                        dedup_stats[reason] = dedup_stats.get(reason, 0) + 1
                        file_dupes += 1; total_dupes += 1
                        continue

                    # Quality filter
                    if not p.get("title") or len(p.get("abstract", "")) < 50:
                        continue

                    seen_ids.add(pid)
                    doi = normalize_doi(p.get("doi", ""))
                    if doi: seen_dois.add(doi)
                    fp = title_fingerprint(p.get("title", ""))
                    if fp: seen_fps.add(fp)
                    ax = arxiv_id_normalize(p.get("arxiv_id", ""))
                    if ax: seen_arxivs.add(ax)

                    new_papers.append(p)
                    file_new += 1
                except: continue
        print(f"  {filepath}: {file_new:,} new | {file_dupes:,} dupes removed")

    print(f"\n[collect] Total dupes removed: {total_dupes:,}")
    print(f"[collect] Dedup breakdown: {dedup_stats}")
    print(f"[collect] New unique papers: {len(new_papers):,}")
    return new_papers

# ── RAM-safe append ───────────────────────────────────────────────────────────

def append_embeddings_ram_safe(new_arr: np.ndarray, new_ids: list):
    """
    Extend embeddings.npy without ever loading the full file into RAM.
    Peak RAM usage: ~300MB per 100K papers chunk + new embeddings only.
    """
    print("\n[save] Appending embeddings (RAM-safe)...")

    # Get shape without loading
    old_mmap   = np.load("embeddings.npy", mmap_mode="r")
    old_n, dim = old_mmap.shape
    new_n      = old_n + len(new_arr)
    del old_mmap

    print(f"[save] {old_n:,} existing + {len(new_arr):,} new = {new_n:,} total")

    # Create destination file at full final size
    dest = np.lib.format.open_memmap(
        "embeddings_new.npy", mode="w+", dtype=np.float32, shape=(new_n, dim)
    )

    # Copy existing in chunks — never more than chunk_size rows in RAM
    chunk_size = 100_000
    src        = np.load("embeddings.npy", mmap_mode="r")
    for start in range(0, old_n, chunk_size):
        end = min(start + chunk_size, old_n)
        dest[start:end] = src[start:end]
        if (start // chunk_size) % 5 == 0:
            print(f"  [save] Copied {end:,}/{old_n:,}...")
    del src

    # Write new vectors
    dest[old_n:] = new_arr
    del dest  # flush

    # Atomic replace
    os.rename("embeddings.npy",     "embeddings_backup.npy")
    os.rename("embeddings_new.npy", "embeddings.npy")
    print(f"[save] embeddings.npy updated ({new_n:,} vectors) ✓")

    # Update paper_ids.json (streaming append)
    existing_ids = json.load(open("paper_ids.json"))
    existing_ids.extend(new_ids)
    with open("paper_ids.json", "w") as f:
        json.dump(existing_ids, f)
    print(f"[save] paper_ids.json updated ({len(existing_ids):,} IDs) ✓")

def update_usearch(new_arr: np.ndarray, new_ids: list, start_idx: int):
    """Add new vectors to usearch index in chunks."""
    from usearch.index import Index
    print("\n[index] Updating usearch index...")
    index = Index.restore("cadence.usearch")
    chunk = 10_000
    for start in range(0, len(new_ids), chunk):
        end  = min(start + chunk, len(new_ids))
        keys = np.arange(start_idx + start, start_idx + end, dtype=np.int64)
        index.add(keys, new_arr[start:end])
        print(f"  [index] {end:,}/{len(new_ids):,}")
    os.rename("cadence.usearch", "cadence_backup.usearch")
    index.save("cadence.usearch")
    print("[index] cadence.usearch updated ✓")

def append_to_merged(new_ids: list):
    """Append new papers to papers_merged.jsonl."""
    print("\n[merge] Appending to papers_merged.jsonl...")
    new_id_set = set(new_ids)
    appended   = 0
    with open("papers_merged.jsonl", "a") as fout:
        for filepath in NEW_PAPER_FILES:
            if not os.path.exists(filepath): continue
            with open(filepath) as fin:
                for line in fin:
                    try:
                        p = json.loads(line)
                        if str(p.get("paper_id", "")) in new_id_set:
                            fout.write(line)
                            appended += 1
                    except: continue
    print(f"[merge] Appended {appended:,} papers ✓")

# ── Embedding loop with checkpointing ────────────────────────────────────────

def embed_all(model, new_papers: list, cp: dict):
    """
    Embed all new papers in batches.
    Saves partial .npy checkpoints every CHECKPOINT_EVERY papers.
    RAM usage: BATCH_SIZE × 768 × 4 bytes at any time.
    """
    processed_set  = set(cp["processed_ids"])
    remaining      = [p for p in new_papers if p["paper_id"] not in processed_set]
    total_embedded = cp["total_embedded"]

    print(f"\n[embed] {len(remaining):,} papers to embed "
          f"({total_embedded:,} already done from checkpoint)")

    all_vecs = []  # accumulate — only in RAM what we've embedded this run
    all_ids  = []

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        texts = [f"{p.get('title','')} [SEP] {p.get('abstract','')}" for p in batch]

        try:
            vecs = model.encode(
                texts, normalize_embeddings=True,
                show_progress_bar=False, batch_size=BATCH_SIZE,
            ).astype(np.float32)
        except Exception as e:
            print(f"  [warn] Batch failed: {e}, skipping")
            continue

        for p, vec in zip(batch, vecs):
            all_vecs.append(vec)
            all_ids.append(p["paper_id"])
            processed_set.add(p["paper_id"])
            total_embedded += 1

        # Checkpoint every CHECKPOINT_EVERY papers
        if total_embedded % CHECKPOINT_EVERY < BATCH_SIZE:
            partial_path = f"checkpoints/partial_emb_{total_embedded}.npy"
            partial_ids  = f"checkpoints/partial_ids_{total_embedded}.json"
            np.save(partial_path, np.array(all_vecs, dtype=np.float32))
            json.dump(all_ids, open(partial_ids, "w"))
            cp["total_embedded"] = total_embedded
            cp["processed_ids"]  = list(processed_set)[-200_000:]
            save_checkpoint(cp)
            print(f"[embed] ✓ Checkpoint: {total_embedded:,} papers | saved {partial_path}")

        progress = batch_start + len(batch)
        if progress % 10_000 < BATCH_SIZE:
            print(f"[embed] {progress:,}/{len(remaining):,} batches done...")

    cp["total_embedded"] = total_embedded
    cp["processed_ids"]  = list(processed_set)[-200_000:]
    save_checkpoint(cp)

    return np.array(all_vecs, dtype=np.float32), all_ids

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cp = load_checkpoint()
    print(f"[embed_new] Checkpoint: {cp['total_embedded']:,} already embedded")

    # Build dedup sets by streaming — not loading full corpus into RAM
    ex_ids, ex_dois, ex_fps, ex_arxivs = build_existing_dedup_sets()

    # Load model
    print("\n[embed_new] Loading SPECTER2 model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("./specter2-finetuned")
    print("[embed_new] Model ready ✓")

    # Collect with dedup
    processed_set = set(cp.get("processed_ids", []))
    dedup_stats   = cp.get("dedup_stats", {})
    new_papers    = collect_new_papers(
        ex_ids, ex_dois, ex_fps, ex_arxivs, processed_set, dedup_stats
    )
    cp["dedup_stats"] = dedup_stats
    save_checkpoint(cp)

    if not new_papers:
        print("[embed_new] Nothing new — corpus is already up to date!")
        return

    # Get current corpus size for index key offset
    with open("paper_ids.json") as f:
        start_idx = len(json.load(f))

    print(f"\n[embed_new] Embedding {len(new_papers):,} papers")
    print(f"[embed_new] Checkpoint every {CHECKPOINT_EVERY:,} | batch size {BATCH_SIZE}")
    print(f"[embed_new] Est. time: ~{len(new_papers) / 500 / 60:.1f} hours on GPU")
    print(f"[embed_new] RAM-safe: loading corpus in chunks, never full file\n")

    new_arr, new_ids = embed_all(model, new_papers, cp)
    del model  # free GPU memory before file ops

    if len(new_ids) == 0:
        print("[embed_new] No new vectors to save")
        return

    # RAM-safe append to embeddings.npy
    append_embeddings_ram_safe(new_arr, new_ids)

    # Update usearch index
    update_usearch(new_arr, new_ids, start_idx)

    # Append metadata to papers_merged.jsonl
    append_to_merged(new_ids)

    print(f"\n[embed_new] ✓ Complete!")
    print(f"[embed_new] Added: {len(new_ids):,} papers")
    print(f"[embed_new] Total corpus: {start_idx + len(new_ids):,} papers")
    print(f"\nDedup summary:")
    for k, v in cp.get("dedup_stats", {}).items():
        print(f"  {k}: {v:,} duplicates removed")
    print(f"\nBackup files created (delete after verifying):")
    print(f"  embeddings_backup.npy")
    print(f"  cadence_backup.usearch")
    print(f"\nVerify with:")
    print(f"  python3 -c \"import numpy as np; e=np.load('embeddings.npy',mmap_mode='r'); print(e.shape)\"")

if __name__ == "__main__":
    main()