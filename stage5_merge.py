"""
Stage 5: Merge S2ORC + OpenAlex + ArXiv into one unified corpus.

Deduplication strategy:
1. DOI match — if two papers share a DOI, keep the one with the longer abstract
2. Title match — normalized title comparison for papers without DOIs
3. ArXiv ID match — papers with same arxiv_id across sources

Output: papers_merged.jsonl — all unique papers with source tag
"""
import json
import re
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SOURCES = [
    ("papers_s2orc.jsonl",     "s2orc"),
    ("papers_openalex.jsonl",  "openalex"),
    ("papers_arxiv.jsonl",     "arxiv"),
]
OUTPUT  = "papers_merged.jsonl"
# ─────────────────────────────────────────────────────────────────────────────


def normalize_title(title):
    """Lowercase, strip punctuation, collapse whitespace for fuzzy matching."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── Load all sources ──────────────────────────────────────────────────────────
all_papers = []

for fname, source in SOURCES:
    count = 0
    print(f"Loading {fname}...")
    try:
        with open(fname) as f:
            for line in tqdm(f, desc=f"  {source}"):
                try:
                    p = json.loads(line)
                    p["source"] = source
                    all_papers.append(p)
                    count += 1
                except Exception:
                    continue
        print(f"  Loaded {count:,} papers from {source}")
    except FileNotFoundError:
        print(f"  WARNING: {fname} not found, skipping")

print(f"\nTotal before dedup: {len(all_papers):,}")

# ── Deduplicate ───────────────────────────────────────────────────────────────
print("\nDeduplicating...")

seen_doi      = {}   # doi -> index in unique_papers
seen_arxiv    = {}   # arxiv_id -> index
seen_title    = {}   # normalized_title -> index
unique_papers = []

for paper in tqdm(all_papers):
    doi       = (paper.get("doi") or "").strip().lower()
    arxiv_id  = (paper.get("arxiv_id") or "").strip()
    norm_title = normalize_title(paper.get("title") or "")
    abstract_len = len(paper.get("abstract") or "")

    matched_idx = None

    # Check DOI match first (most reliable)
    if doi and doi in seen_doi:
        matched_idx = seen_doi[doi]
    # Then ArXiv ID
    elif arxiv_id and arxiv_id in seen_arxiv:
        matched_idx = seen_arxiv[arxiv_id]
    # Then normalized title (catches same paper in different sources)
    elif norm_title and len(norm_title) > 20 and norm_title in seen_title:
        matched_idx = seen_title[norm_title]

    if matched_idx is not None:
        # Keep the version with the longer abstract
        existing = unique_papers[matched_idx]
        if abstract_len > len(existing.get("abstract") or ""):
            # Update with better abstract, merge source tags
            paper["source"] = existing["source"] + "+" + paper["source"]
            unique_papers[matched_idx] = paper
        continue

    # New unique paper
    idx = len(unique_papers)
    unique_papers.append(paper)

    if doi:
        seen_doi[doi] = idx
    if arxiv_id:
        seen_arxiv[arxiv_id] = idx
    if norm_title and len(norm_title) > 20:
        seen_title[norm_title] = idx

# ── Write output ──────────────────────────────────────────────────────────────
print(f"\nWriting {OUTPUT}...")
with open(OUTPUT, "w") as f:
    for p in tqdm(unique_papers):
        f.write(json.dumps(p) + "\n")

# ── Stats ─────────────────────────────────────────────────────────────────────
from collections import Counter
source_counts = Counter(p["source"] for p in unique_papers)

print(f"\nDone.")
print(f"Total before dedup  : {len(all_papers):,}")
print(f"Total after dedup   : {len(unique_papers):,}")
print(f"Duplicates removed  : {len(all_papers) - len(unique_papers):,}")
print(f"\nSource breakdown:")
for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {source:30s} {count:>8,}")
print(f"\nSaved to {OUTPUT}")