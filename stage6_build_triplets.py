
"""
Stage 6: Build training triplets from all similarity signals.
 
Signals used:
1. Bibliographic coupling pairs (from pairs_biblio_coupling.jsonl)
2. ArXiv category pairs (same subcategory = related)
 
Output: triplets.jsonl — (anchor, positive, negative) for fine-tuning
"""

import json
from tqdm import tqdm
import random 
from collections import defaultdict 

MERGED_PAPERS="papers_merged.jsonl"
BIBLIO_PAIRS="pairs_biblio_coupling.jsonl"
OUTPUT="triplets.jsonl"
MAX_TRIPLETS=300_000
MAX_POS_PER_PAPER=5 

print(f"Loading merged papers from {MERGED_PAPERS}...")
papers = []
id_to_paper = {}

with open(MERGED_PAPERS) as f:
    for line in tqdm(f,desc="Loading papers"):
        p=json.loads(line)
        id_to_paper[str(p["paper_id"])] = p 
        papers.append(p)

all_ids=list(id_to_paper.keys())
print(f"loaded {len(papers):,}papers")

def make_triplet(anchor_id,positive_id):
    anchor=id_to_paper.get(str(anchor_id))
    positive=id_to_paper.get(str(positive_id))

    if not anchor or not positive:
        return None

    neg_id=random.choice(all_ids)
    attempts=0
    while(neg_id==str(anchor_id) or neg_id==str(positive_id)) and attempts<10:
        neg_id=random.choice(all_ids) 
        attempts+=1
    
    negative=id_to_paper.get(neg_id)
    if not negative:
        return None
    
    return {
        "anchor":   (anchor.get("title") or "")   + " " + (anchor.get("abstract") or ""),
        "positive": (positive.get("title") or "")  + " " + (positive.get("abstract") or ""),
        "negative": (negative.get("title") or "")  + " " + (negative.get("abstract") or ""),
    }

triplets=[]
source_counts=defaultdict(int)

# ── Signal 1: Bibliographic coupling ─────────────────────────────────────────
print(f"\nSignal 1: Bibliographic coupling pairs...")
try:
    biblio_added = 0
    with open(BIBLIO_PAIRS) as f:
        for line in tqdm(f, desc="Biblio coupling"):
            pair = json.loads(line)
            t    = make_triplet(pair["paper_a"], pair["paper_b"])
            if t:
                t["signal"] = "biblio_coupling"
                triplets.append(t)
                biblio_added += 1
    print(f"  Added {biblio_added:,} triplets from bibliographic coupling")
    source_counts["biblio_coupling"] = biblio_added
except FileNotFoundError:
    print(f"  {BIBLIO_PAIRS} not found, skipping")
 
# ── Signal 2: ArXiv category pairs ───────────────────────────────────────────
print(f"\nSignal 2: ArXiv category pairs...")
 
# Group ArXiv papers by their primary subcategory
category_groups = defaultdict(list)
for p in papers:
    if p.get("source") in ("arxiv", "s2orc+arxiv", "openalex+arxiv") and p.get("categories"):
        # Primary category e.g. "cs.LG math.ST" → use "cs.LG"
        primary = p["categories"].split()[0]
        category_groups[primary].append(str(p["paper_id"]))
 
arxiv_added = 0
for category, ids in tqdm(category_groups.items(), desc="Category pairs"):
    if len(ids) < 2:
        continue
    # Sample up to MAX_POS_PER_PAPER pairs per category
    random.shuffle(ids)
    for i in range(min(len(ids) - 1, MAX_POS_PER_PAPER * 10)):
        t = make_triplet(ids[i], ids[i + 1])
        if t:
            t["signal"] = "arxiv_category"
            triplets.append(t)
            arxiv_added += 1
        if arxiv_added >= 100_000:
            break
    if arxiv_added >= 100_000:
        break
 
print(f"  Added {arxiv_added:,} triplets from ArXiv categories")
source_counts["arxiv_category"] = arxiv_added
 

# ── Shuffle + cap ─────────────────────────────────────────────────────────────
print(f"\nTotal triplets before cap : {len(triplets):,}")
random.shuffle(triplets)
triplets = triplets[:MAX_TRIPLETS]
 
# ── Save ──────────────────────────────────────────────────────────────────────
print(f"Writing {OUTPUT}...")
with open(OUTPUT, "w") as f:
    for t in tqdm(triplets):
        f.write(json.dumps(t) + "\n")
 
print(f"\nDone.")
print(f"Final triplet count : {len(triplets):,}")
print(f"\nSignal breakdown:")
for signal, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {signal:25s} {count:>8,}")
print(f"\nSaved to {OUTPUT}")

