"""
nightly_ingest.py — Nightly job to fetch papers published yesterday
RAM-safe: never loads full embeddings into RAM, uses memory-mapped files
Checkpoint-safe: can be killed and restarted

Run via cron: 0 2 * * * cd /home/rohan/cadence && python nightly_ingest.py >> nightly.log 2>&1
"""

import os
import json
import re
import time
import hashlib
import requests
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

BATCH_SIZE = 64
yesterday  = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
today      = datetime.today().strftime("%Y-%m-%d")

os.makedirs("checkpoints", exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

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

def load_existing_dedup_sets():
    """Load dedup sets from existing corpus — streaming, no full load into RAM."""
    log("Building dedup sets from existing corpus (streaming)...")
    existing_ids   = set(json.load(open("paper_ids.json")))
    existing_dois  = set()
    existing_fps   = set()
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
            except: continue

    log(f"Dedup sets: {len(existing_ids):,} IDs | {len(existing_dois):,} DOIs | "
        f"{len(existing_fps):,} titles | {len(existing_arxivs):,} arXiv IDs")
    return existing_ids, existing_dois, existing_fps, existing_arxivs

def is_duplicate(p, ex_ids, ex_dois, ex_fps, ex_arxivs, seen_ids, seen_dois, seen_fps, seen_arxivs):
    pid = str(p.get("paper_id", ""))
    if pid in ex_ids or pid in seen_ids: return True
    doi = normalize_doi(p.get("doi", ""))
    if doi and (doi in ex_dois or doi in seen_dois): return True
    ax = arxiv_id_normalize(p.get("arxiv_id", ""))
    if ax and (ax in ex_arxivs or ax in seen_arxivs): return True
    fp = title_fingerprint(p.get("title", ""))
    if fp and (fp in ex_fps or fp in seen_fps): return True
    return False

# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_arxiv_new():
    log("Fetching new arXiv papers...")
    papers = []
    url    = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"submittedDate:[{yesterday.replace('-','')}0000+TO+{today.replace('-','')}2359]",
        "start":        0,
        "max_results":  500,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }
    try:
        r    = requests.get(url, params=params, timeout=30)
        root = ET.fromstring(r.content)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            try:
                title    = entry.findtext("atom:title", "", ns).replace("\n", " ").strip()
                abstract = entry.findtext("atom:summary", "", ns).replace("\n", " ").strip()
                if not title or len(abstract) < 50: continue

                arxiv_id = entry.findtext("atom:id", "", ns).split("/abs/")[-1]
                arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
                authors  = [a.findtext("atom:name", "", ns)
                            for a in entry.findall("atom:author", ns)][:10]
                year     = int(entry.findtext("atom:published", "2024", ns)[:4])
                cats     = [c.get("term", "")
                            for c in entry.findall("atom:category", ns)][:5]

                papers.append({
                    "paper_id":        f"arxiv_{arxiv_id.replace('/', '_')}",
                    "title":           title,
                    "abstract":        abstract,
                    "authors":         authors,
                    "year":            year,
                    "venue":           "arXiv",
                    "doi":             None,
                    "arxiv_id":        arxiv_id,
                    "categories":      cats,
                    "source":          "arxiv",
                    "citation_count":  None,
                    "open_access_url": f"https://arxiv.org/pdf/{arxiv_id}",
                })
            except: continue

        log(f"arXiv: {len(papers)} new papers")
    except Exception as e:
        log(f"arXiv fetch failed: {e}")
    return papers

def fetch_biorxiv_new():
    log("Fetching new bioRxiv/medRxiv papers...")
    papers = []
    for server in ["biorxiv", "medrxiv"]:
        url = f"https://api.biorxiv.org/details/{server}/{yesterday}/{today}/0/json"
        try:
            r    = requests.get(url, timeout=30)
            data = r.json()
            for item in data.get("collection", []):
                abstract = item.get("abstract", "").strip()
                if len(abstract) < 50: continue
                doi = item.get("doi", "")
                papers.append({
                    "paper_id":        f"{server}_{doi.replace('/', '_')}",
                    "title":           item.get("title", "").strip(),
                    "abstract":        abstract,
                    "authors":         [a.strip() for a in item.get("authors", "").split(";") if a.strip()][:10],
                    "year":            int(item.get("date", "2024")[:4]),
                    "venue":           server,
                    "doi":             doi or None,
                    "arxiv_id":        None,
                    "categories":      [item.get("category", "")] if item.get("category") else [],
                    "source":          server,
                    "citation_count":  None,
                    "open_access_url": f"https://www.{server}.org/content/{doi}v1.full.pdf",
                })
        except Exception as e:
            log(f"{server} failed: {e}")
    log(f"bioRxiv/medRxiv: {len(papers)} papers")
    return papers

# ── RAM-safe append ───────────────────────────────────────────────────────────

def append_embeddings_ram_safe(new_vecs: np.ndarray, new_ids: list):
    """
    Append new embeddings to embeddings.npy without loading full array into RAM.
    Uses memory-mapped files — RAM usage stays ~constant regardless of corpus size.
    """
    log("Appending embeddings (RAM-safe mmap method)...")

    # Get existing shape without loading data
    existing_mmap = np.load("embeddings.npy", mmap_mode="r")
    old_n, dims   = existing_mmap.shape
    new_n         = old_n + len(new_vecs)
    del existing_mmap  # release mmap immediately

    log(f"Existing: {old_n:,} vectors | Adding: {len(new_vecs):,} | New total: {new_n:,}")

    # Create new file with correct final size
    log("Creating new embeddings file...")
    new_emb = np.lib.format.open_memmap(
        "embeddings_new.npy",
        mode="w+",
        dtype=np.float32,
        shape=(new_n, dims),
    )

    # Copy existing data in RAM-safe chunks
    log("Copying existing embeddings in chunks...")
    old_emb    = np.load("embeddings.npy", mmap_mode="r")
    chunk_size = 100_000  # ~300MB per chunk at 768 dims
    for start in range(0, old_n, chunk_size):
        end = min(start + chunk_size, old_n)
        new_emb[start:end] = old_emb[start:end]
        if start % 500_000 == 0:
            log(f"  Copied {end:,}/{old_n:,}...")
    del old_emb

    # Append new vectors at the end
    new_emb[old_n:] = new_vecs
    del new_emb  # flush to disk
    log("New embeddings file written ✓")

    # Atomic swap
    os.rename("embeddings.npy",     "embeddings_prenightly.npy")
    os.rename("embeddings_new.npy", "embeddings.npy")

    # Update paper_ids.json
    existing_ids = json.load(open("paper_ids.json"))
    existing_ids.extend(new_ids)
    json.dump(existing_ids, open("paper_ids.json", "w"))

    # Clean prenightly backup
    os.remove("embeddings_prenightly.npy")
    log(f"embeddings.npy updated: {new_n:,} total vectors ✓")

def update_usearch_index(new_vecs: np.ndarray, new_ids: list, start_idx: int):
    """Add new vectors to usearch index."""
    from usearch.index import Index
    log("Updating usearch index...")
    index    = Index.restore("cadence.usearch")
    chunk    = 10_000
    for start in range(0, len(new_ids), chunk):
        end  = min(start + chunk, len(new_ids))
        keys = np.arange(start_idx + start, start_idx + end, dtype=np.int64)
        index.add(keys, new_vecs[start:end])
    os.rename("cadence.usearch", "cadence_prenightly.usearch")
    index.save("cadence.usearch")
    os.remove("cadence_prenightly.usearch")
    log(f"usearch index updated ✓")

# ── Embed ─────────────────────────────────────────────────────────────────────

def embed_papers(papers: list) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    log("Loading SPECTER2 model...")
    model = SentenceTransformer("./specter2-finetuned")
    texts = [f"{p['title']} [SEP] {p['abstract']}" for p in papers]
    log(f"Embedding {len(papers)} papers...")
    vecs  = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    ).astype(np.float32)
    del model  # free GPU memory immediately
    return vecs

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log(f"Nightly ingest for {yesterday}")
    log("=" * 60)

    ex_ids, ex_dois, ex_fps, ex_arxivs = load_existing_dedup_sets()
    log(f"Current corpus: {len(ex_ids):,} papers")

    # Fetch from all sources
    all_new = []
    all_new.extend(fetch_arxiv_new())
    all_new.extend(fetch_biorxiv_new())

    # Deduplicate
    seen_ids = set(); seen_dois = set()
    seen_fps = set(); seen_arxivs = set()
    filtered = []
    for p in all_new:
        if is_duplicate(p, ex_ids, ex_dois, ex_fps, ex_arxivs,
                        seen_ids, seen_dois, seen_fps, seen_arxivs):
            continue
        pid = p["paper_id"]
        seen_ids.add(pid)
        doi = normalize_doi(p.get("doi", ""))
        if doi: seen_dois.add(doi)
        fp = title_fingerprint(p.get("title", ""))
        if fp: seen_fps.add(fp)
        ax = arxiv_id_normalize(p.get("arxiv_id", ""))
        if ax: seen_arxivs.add(ax)
        filtered.append(p)

    log(f"New unique papers after dedup: {len(filtered)} "
        f"(removed {len(all_new) - len(filtered)} duplicates)")

    if not filtered:
        log("Nothing new today — done")
        return

    # Embed
    new_vecs = embed_papers(filtered)

    # Get current corpus size for index keys
    start_idx = len(ex_ids)

    # RAM-safe append to embeddings.npy
    new_ids = [p["paper_id"] for p in filtered]
    append_embeddings_ram_safe(new_vecs, new_ids)

    # Update usearch index
    update_usearch_index(new_vecs, new_ids, start_idx)

    # Append to papers_merged.jsonl
    with open("papers_merged.jsonl", "a") as f:
        for p in filtered:
            f.write(json.dumps(p) + "\n")

    log(f"✓ Added {len(filtered)} papers to corpus")
    log(f"✓ Total corpus: {start_idx + len(filtered):,} papers")

    # Restart API to pick up new index
    log("Restarting cadence-api...")
    import requests
    try:
        r = requests.post(
            "http://localhost:8000/admin/reload",
            params={"secret": "cadence-reload-2024"},
            timeout=300  # 5 min timeout for reload
        )
        log(f"Hot reload: {r.json()}")
    except Exception as e:
        log(f"Hot reload failed: {e}")
        log("Done ✓")

if __name__ == "__main__":
    main()