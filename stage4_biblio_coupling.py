"""
Stage 4: Bibliographic coupling — memory-safe version.

Processes papers in batches and writes pairs to disk incrementally
instead of holding all pairs in RAM at once.
"""
import json
import os
from collections import defaultdict
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
INPUT         = "papers_openalex.jsonl"
OUTPUT        = "pairs_biblio_coupling.jsonl"
MIN_SHARED    = 3       # minimum shared refs to count as a positive pair
MAX_CITERS    = 200     # skip refs cited by more than this (too generic)
MAX_PAIRS     = 300_000 # cap total output pairs
CHUNK_SIZE    = 200_000 # process this many papers at a time
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading {INPUT}...")
papers    = []
id_to_idx = {}

with open(INPUT) as f:
    for i, line in enumerate(f):
        p = json.loads(line)
        papers.append(p)
        id_to_idx[p["paper_id"]] = i

print(f"Loaded {len(papers)} papers")
print(f"Processing in chunks of {CHUNK_SIZE}...\n")

total_pairs_written = 0

with open(OUTPUT, "w") as out_f:
    # Process in chunks to limit memory usage
    for chunk_start in range(0, len(papers), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(papers))
        chunk     = papers[chunk_start:chunk_end]
        print(f"Chunk {chunk_start//CHUNK_SIZE + 1}: papers {chunk_start}–{chunk_end}")

        # Build inverted index for this chunk only
        ref_to_citers = defaultdict(list)
        for local_i, paper in enumerate(tqdm(chunk, desc="  Building index")):
            global_i = chunk_start + local_i
            for ref in (paper.get("refs") or []):
                ref_to_citers[ref].append(global_i)

        # Compute pairs within this chunk
        pair_strength = defaultdict(int)
        for ref, citers in tqdm(ref_to_citers.items(), desc="  Computing pairs"):
            if len(citers) < 2 or len(citers) > MAX_CITERS:
                continue
            for i in range(len(citers)):
                for j in range(i + 1, len(citers)):
                    a, b = citers[i], citers[j]
                    if a > b:
                        a, b = b, a
                    pair_strength[(a, b)] += 1

        # Write strong pairs to disk immediately, clear from memory
        chunk_pairs = [
            (a, b, s) for (a, b), s in pair_strength.items()
            if s >= MIN_SHARED
        ]
        chunk_pairs.sort(key=lambda x: -x[2])
        chunk_pairs = chunk_pairs[:MAX_PAIRS // (len(papers) // CHUNK_SIZE + 1)]

        for a, b, strength in chunk_pairs:
            out_f.write(json.dumps({
                "paper_a":  papers[a]["paper_id"],
                "paper_b":  papers[b]["paper_id"],
                "strength": strength,
            }) + "\n")
        out_f.flush()

        total_pairs_written += len(chunk_pairs)
        print(f"  Pairs written this chunk: {len(chunk_pairs):,}")
        print(f"  Total pairs so far: {total_pairs_written:,}\n")

        # Explicitly free memory
        del ref_to_citers
        del pair_strength

        if total_pairs_written >= MAX_PAIRS:
            print("Reached MAX_PAIRS cap, stopping early.")
            break

print(f"\nDone. Total pairs written: {total_pairs_written:,}")
print(f"Saved to: {OUTPUT}")

# Show sample
print("\nSample pairs:")
with open(OUTPUT) as f:
    for i, line in enumerate(f):
        p = json.loads(line)
        a_title = papers[id_to_idx.get(p['paper_a'], 0)]['title'][:50]
        b_title = papers[id_to_idx.get(p['paper_b'], 0)]['title'][:50]
        print(f"  [{p['strength']} shared] {a_title} ↔ {b_title}")
        if i >= 4:
            break