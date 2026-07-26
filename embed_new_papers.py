"""
embed_new_papers.py — Embed new papers and add to existing index
RAM-safe: streams papers one batch at a time, never loads full corpus into RAM
Checkpoint-safe: saves every CHECKPOINT_EVERY papers

Usage: python embed_new_papers.py
"""

import os
import json
import re
import hashlib
import numpy as np

CHECKPOINT_FILE  = "checkpoints/embed_checkpoint.json"
CHECKPOINT_EVERY = 10_000
BATCH_SIZE       = 64   # smaller batch = less RAM
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
    """Stream through existing corpus — never loads full corpus into RAM."""
    print("[dedup] Streaming existing corpus for dedup sets...")
    with open("paper_ids.json") as f:
        existing_ids = set(json.load(f))
    existing_dois   = set()
    existing_fps    = set()
    existing_arxivs = set()
    count = 0
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
    if pid in ex_ids or pid in seen_ids: return True
    doi = normalize_doi(p.get("doi", ""))
    if doi and (doi in ex_dois or doi in seen_dois): return True
    ax = arxiv_id_normalize(p.get("arxiv_id", ""))
    if ax and (ax in ex_arxivs or ax in seen_arxivs): return True
    fp = title_fingerprint(p.get("title", ""))
    if fp and (fp in ex_fps or fp in seen_fps): return True
    return False

# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed_ids": [], "total_embedded": 0, "dedup_stats": {},
            "file_index": 0, "file_line": 0}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f)

# ── RAM-safe append ───────────────────────────────────────────────────────────

def append_embeddings_ram_safe(new_arr: np.ndarray, new_ids: list):
    """Extend embeddings.npy in chunks — never loads full file into RAM."""
    print(f"\n[save] Appending {len(new_ids):,} embeddings (RAM-safe)...")
    old_mmap   = np.load("embeddings.npy", mmap_mode="r")
    old_n, dim = old_mmap.shape
    new_n      = old_n + len(new_arr)
    del old_mmap

    dest = np.lib.format.open_memmap(
        "embeddings_new.npy", mode="w+", dtype=np.float32, shape=(new_n, dim)
    )
    src        = np.load("embeddings.npy", mmap_mode="r")
    chunk_size = 100_000
    for start in range(0, old_n, chunk_size):
        end = min(start + chunk_size, old_n)
        dest[start:end] = src[start:end]
        if start % 500_000 == 0 and start > 0:
            print(f"  [save] Copied {end:,}/{old_n:,}...")
    del src
    dest[old_n:] = new_arr
    del dest

    os.rename("embeddings.npy",     "embeddings_backup.npy")
    os.rename("embeddings_new.npy", "embeddings.npy")

    existing_ids = json.load(open("paper_ids.json"))
    existing_ids.extend(new_ids)
    json.dump(existing_ids, open("paper_ids.json", "w"))
    print(f"[save] embeddings.npy updated: {new_n:,} total vectors ✓")
    return old_n  # return start_idx for usearch

def update_usearch(new_arr: np.ndarray, new_ids: list, start_idx: int):
    from usearch.index import Index
    print("[index] Updating usearch index...")
    index = Index.restore("cadence.usearch")
    chunk = 10_000
    for start in range(0, len(new_ids), chunk):
        end  = min(start + chunk, len(new_ids))
        keys = np.arange(start_idx + start, start_idx + end, dtype=np.int64)
        index.add(keys, new_arr[start:end])
        if start % 100_000 == 0 and start > 0:
            print(f"  [index] {end:,}/{len(new_ids):,}")
    os.rename("cadence.usearch", "cadence_backup.usearch")
    index.save("cadence.usearch")
    print("[index] cadence.usearch updated ✓")

# ── Streaming paper iterator ──────────────────────────────────────────────────

def stream_new_papers(ex_ids, ex_dois, ex_fps, ex_arxivs,
                      processed_set, dedup_stats,
                      start_file_idx=0, start_line=0):
    """
    Generator that yields new papers one at a time.
    Never holds more than one paper in memory.
    """
    seen_ids = set(processed_set)
    seen_dois = seen_fps = seen_arxivs = set()

    for file_idx, filepath in enumerate(NEW_PAPER_FILES):
        if file_idx < start_file_idx:
            continue
        if not os.path.exists(filepath):
            print(f"  [skip] {filepath} not found")
            continue

        print(f"[stream] Processing {filepath}...")
        with open(filepath) as f:
            for line_num, line in enumerate(f):
                if file_idx == start_file_idx and line_num < start_line:
                    continue
                try:
                    p   = json.loads(line)
                    pid = str(p.get("paper_id", ""))
                    if not pid: continue

                    if is_duplicate(p, ex_ids, ex_dois, ex_fps, ex_arxivs,
                                    seen_ids, seen_dois, seen_fps, seen_arxivs):
                        reason = "dup"
                        dedup_stats[reason] = dedup_stats.get(reason, 0) + 1
                        continue

                    if not p.get("title") or len(p.get("abstract", "")) < 50:
                        continue

                    seen_ids.add(pid)
                    doi = normalize_doi(p.get("doi", ""))
                    if doi: seen_dois.add(doi)
                    fp = title_fingerprint(p.get("title", ""))
                    if fp: seen_fps.add(fp)
                    ax = arxiv_id_normalize(p.get("arxiv_id", ""))
                    if ax: seen_arxivs.add(ax)

                    yield p, file_idx, line_num
                except: continue

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cp = load_checkpoint()
    print(f"[embed_new] Checkpoint: {cp['total_embedded']:,} already embedded")

    # Build dedup sets
    ex_ids, ex_dois, ex_fps, ex_arxivs = build_existing_dedup_sets()

    # Load model
    print("\n[embed_new] Loading SPECTER2 model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("./specter2-finetuned")
    print("[embed_new] Model ready ✓")

    processed_set = set(cp.get("processed_ids", []))
    dedup_stats   = cp.get("dedup_stats", {})
    start_file    = cp.get("file_index", 0)
    start_line    = cp.get("file_line", 0)
    total_embedded = cp["total_embedded"]

    # Get current corpus size for index offset
    with open("paper_ids.json") as f:
        start_idx = len(json.load(f))

    print(f"\n[embed_new] Streaming and embedding papers...")
    print(f"[embed_new] Batch size: {BATCH_SIZE} | Checkpoint every: {CHECKPOINT_EVERY:,}")
    print(f"[embed_new] RAM-safe: no full corpus loaded into memory\n")

    # Accumulate embeddings in chunks of CHECKPOINT_EVERY
    batch_papers  = []
    batch_vecs    = []
    all_new_ids   = []
    all_new_vecs  = []  # only holds current checkpoint chunk

    paper_stream = stream_new_papers(
        ex_ids, ex_dois, ex_fps, ex_arxivs,
        processed_set, dedup_stats,
        start_file, start_line
    )

    current_batch = []
    last_file_idx = start_file
    last_line_num = start_line

    for p, file_idx, line_num in paper_stream:
        current_batch.append(p)
        last_file_idx = file_idx
        last_line_num = line_num

        if len(current_batch) < BATCH_SIZE:
            continue

        # Encode batch
        texts = [f"{p.get('title','')} [SEP] {p.get('abstract','')}" for p in current_batch]
        try:
            vecs = model.encode(
                texts, normalize_embeddings=True,
                show_progress_bar=False, batch_size=BATCH_SIZE,
            ).astype(np.float32)
        except Exception as e:
            print(f"  [warn] Batch failed: {e}")
            current_batch = []
            continue

        for p, vec in zip(current_batch, vecs):
            all_new_ids.append(p["paper_id"])
            all_new_vecs.append(vec)
            processed_set.add(p["paper_id"])
            total_embedded += 1

        current_batch = []

        # Checkpoint every CHECKPOINT_EVERY papers
        if total_embedded % CHECKPOINT_EVERY < BATCH_SIZE:
            print(f"[embed] ✓ {total_embedded:,} embedded | saving checkpoint...")
            # Save partial embeddings
            partial_arr  = np.array(all_new_vecs, dtype=np.float32)
            partial_path = f"checkpoints/partial_emb_{total_embedded}.npy"
            partial_ids  = f"checkpoints/partial_ids_{total_embedded}.json"
            np.save(partial_path, partial_arr)
            json.dump(all_new_ids, open(partial_ids, "w"))
            all_new_vecs = []   # clear to free RAM
            all_new_ids  = [] 
            cp["total_embedded"] = total_embedded
            cp["processed_ids"]  = list(processed_set)[-200_000:]
            cp["file_index"]     = last_file_idx
            cp["file_line"]      = last_line_num
            cp["dedup_stats"]    = dedup_stats
            save_checkpoint(cp)
            print(f"  Saved {partial_path}")

    # Process remaining batch
    if current_batch:
        texts = [f"{p.get('title','')} [SEP] {p.get('abstract','')}" for p in current_batch]
        try:
            vecs = model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False).astype(np.float32)
            for p, vec in zip(current_batch, vecs):
                all_new_ids.append(p["paper_id"])
                all_new_vecs.append(vec)
                total_embedded += 1
        except Exception as e:
            print(f"  [warn] Final batch failed: {e}")

    if not all_new_ids:
        print("[embed_new] Nothing new to embed!")
        return

    print(f"\n[embed_new] Embedding complete: {total_embedded:,} papers")
    print(f"[embed_new] Saving to disk...")

    del model  # free GPU memory before file ops

    new_arr = np.array(all_new_vecs, dtype=np.float32)
    del all_new_vecs  # free RAM

    # RAM-safe append
    append_embeddings_ram_safe(new_arr, all_new_ids)

    # Update usearch
    update_usearch(new_arr, all_new_ids, start_idx)

    # Append to papers_merged.jsonl
    print("[merge] Appending to papers_merged.jsonl...")
    new_id_set = set(all_new_ids)
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

    cp["total_embedded"] = total_embedded
    save_checkpoint(cp)

    print(f"\n[embed_new] ✓ Complete!")
    print(f"[embed_new] Added: {len(all_new_ids):,} papers")
    print(f"[embed_new] Total corpus: {start_idx + len(all_new_ids):,} papers")
    print(f"\nDedup stats: {dedup_stats}")
    print(f"\nVerify: python3 -c \"import numpy as np; e=np.load('embeddings.npy',mmap_mode='r'); print(e.shape)\"")
    print(f"Then restart API: sudo systemctl start cadence-api")

if __name__ == "__main__":
    main()