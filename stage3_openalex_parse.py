"""
Stage 3: Parse OpenAlex shards into unified paper format.

OpenAlex stores abstracts as inverted indexes ({"word": [positions...]}). We
reconstruct them to plain text. We also extract referenced_works (citation graph)
which we'll use later for bibliographic coupling.
"""
import json
import os
import gzip
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
LOCAL_PATH = "./openalex_snapshot"
OUTPUT     = "papers_openalex.jsonl"
MIN_ABSTRACT_LEN = 100
# ─────────────────────────────────────────────────────────────────────────────


def reconstruct_abstract(inverted_index):
    """OpenAlex stores abstracts as {word: [positions]}. Rebuild to text."""
    if not inverted_index:
        return ""
    pos_word = {}
    for word, positions in inverted_index.items():
        for p in positions:
            pos_word[p] = word
    return " ".join(pos_word[i] for i in sorted(pos_word.keys()))


def clean_id(openalex_id):
    """Extract just the W-number from full URL like https://openalex.org/W123"""
    if not openalex_id:
        return None
    return openalex_id.rsplit("/", 1)[-1]


papers = []
total_lines = 0
total_kept  = 0

shards = sorted(f for f in os.listdir(LOCAL_PATH) if f.endswith(".gz"))
print(f"Parsing {len(shards)} OpenAlex shards...\n")

for fname in shards:
    print(f"  {fname}")
    with gzip.open(os.path.join(LOCAL_PATH, fname), "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=fname[:30]):
            total_lines += 1
            try:
                row = json.loads(line)

                # Reconstruct abstract from inverted index
                abstract = reconstruct_abstract(row.get("abstract_inverted_index"))
                if len(abstract) < MIN_ABSTRACT_LEN:
                    continue

                title = (row.get("title") or "").strip()
                if not title:
                    continue

                # References — list of OpenAlex work IDs this paper cites
                refs = [
                    clean_id(r)
                    for r in (row.get("referenced_works") or [])
                ]
                refs = [r for r in refs if r]

                # External IDs for cross-source deduplication later
                ids = row.get("ids") or {}

                papers.append({
                    "paper_id": clean_id(row.get("id")),
                    "source":   "openalex",
                    "title":    title,
                    "abstract": abstract,
                    "doi":      ids.get("doi"),
                    "arxiv_id": None,
                    "refs":     refs,           # citation graph for bib coupling
                })
                total_kept += 1

            except Exception:
                continue

print(f"\nWriting {OUTPUT}...")
with open(OUTPUT, "w") as f:
    for p in papers:
        f.write(json.dumps(p) + "\n")

print(f"\nTotal lines scanned : {total_lines:,}")
print(f"Papers kept         : {total_kept:,}")
print(f"Saved to            : {OUTPUT}")
if papers:
    print(f"\nSample paper:")
    print(f"  Title: {papers[0]['title'][:80]}")
    print(f"  Refs : {len(papers[0]['refs'])}")