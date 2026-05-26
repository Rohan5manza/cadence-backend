

'''Citation triplets: A citation triplet is a set of three papers used to teach the model what "similar" means in academic research.

It has three parts:

- **Anchor** — a paper you're starting from
- **Positive** — a paper the anchor cites (by definition, related)
- **Negative** — a random paper (likely unrelated)

So concretely it looks like this:

```
Anchor:   "Attention is all you need. We propose a transformer architecture..."
Positive: "BERT: Pre-training deep bidirectional transformers..."
Negative: "Novel approaches to breast cancer fatty acid metabolism..."
```

During fine-tuning, the model sees these three and gets trained to answer one question: **"make the anchor embedding closer to the positive, and farther from the negative."**

The core insight is that **citation = similarity signal**. If a paper cites another paper, the authors considered it relevant enough to reference. That's a very strong human-generated label saying "these two papers are related" — and you have millions of them for free, without any manual annotation.

After enough triplets, the model learns that transformer papers cluster together, cancer biology papers cluster together, and so on — not because you told it the topics, but because it learned the pattern from citation relationships.

This is called **contrastive learning**, and it's the same fundamental idea behind how Spotify learns that two songs are similar — not from their audio, but from the fact that the same people listen to both.
'''


import json
import random
from tqdm import tqdm

INPUT  = "papers_enriched.jsonl"
OUTPUT = "triplets.jsonl"
MAX_POSITIVES_PER_PAPER = 5

# ── Load papers ───────────────────────────────────────────────────────────────
print(f"Loading papers from {INPUT}...")
with open(INPUT) as f:
    papers = [json.loads(l) for l in f]

id_to_paper = {str(p["paper_id"]): p for p in papers}
all_ids     = list(id_to_paper.keys())
print(f"Loaded {len(papers)} papers")

# ── Build triplets ─────────────────────────────────────────────────────────────
print("Building triplets...")
triplets = []
skipped  = 0

for paper in tqdm(papers):
    # Use similar_to if available, fall back to cited_by
    similar = paper.get("similar_to") or paper.get("cited_by") or []
    similar = [s for s in similar if str(s) in id_to_paper]

    if not similar:
        skipped += 1
        continue

    for sim_id in similar[:MAX_POSITIVES_PER_PAPER]:
        neg_id = random.choice(all_ids)
        while neg_id == str(paper["paper_id"]) or neg_id == str(sim_id):
            neg_id = random.choice(all_ids)

        triplets.append({
            "anchor":   paper["title"] + " " + paper["abstract"],
            "positive": id_to_paper[str(sim_id)]["title"] + " " + id_to_paper[str(sim_id)]["abstract"],
            "negative": id_to_paper[neg_id]["title"] + " " + id_to_paper[neg_id]["abstract"],
        })

random.shuffle(triplets)

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT, "w") as f:
    for t in triplets:
        f.write(json.dumps(t) + "\n")

print(f"\nDone.")
print(f"Triplets generated        : {len(triplets)}")
print(f"Papers with no matches    : {skipped}")
if triplets:
    print(f"Sample anchor (100 chars) : {triplets[0]['anchor'][:100]}")