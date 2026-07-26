import numpy as np, json, glob, os

print("Starting partial merge...", flush=True)

partial_npy = sorted(glob.glob("checkpoints/partial_emb_*.npy"),
    key=lambda x: int(x.split("partial_emb_")[1].replace(".npy","")))
partial_ids_files = sorted(glob.glob("checkpoints/partial_ids_*.json"),
    key=lambda x: int(x.split("partial_ids_")[1].replace(".json","")))

print(f"Found {len(partial_npy)} files", flush=True)

# Count unique IDs one file at a time — never load all into RAM
print("Counting unique IDs (streaming)...", flush=True)
seen = set()
total_unique = 0
for f in partial_ids_files:
    ids = json.load(open(f))
    for pid in ids:
        if pid not in seen:
            seen.add(pid)
            total_unique += 1
del seen  # free RAM immediately
print(f"Unique new papers: {total_unique:,}", flush=True)

# Get existing shape
existing   = np.load("embeddings.npy", mmap_mode="r")
old_n, dim = existing.shape
new_n      = old_n + total_unique
del existing
print(f"Final shape: ({new_n:,}, {dim})", flush=True)

# Create destination file
print("Creating embeddings_merged.npy...", flush=True)
merged = np.lib.format.open_memmap(
    "embeddings_merged.npy", mode="w+", dtype=np.float32, shape=(new_n, dim)
)

# Copy existing in chunks
print("Copying existing embeddings...", flush=True)
src = np.load("embeddings.npy", mmap_mode="r")
for start in range(0, old_n, 100_000):
    end = min(start + 100_000, old_n)
    merged[start:end] = src[start:end]
    if start % 500_000 == 0 and start > 0:
        print(f"  {end:,}/{old_n:,}", flush=True)
del src
print("Existing copied ✓", flush=True)

# Append new vectors one file at a time
print("Appending new vectors...", flush=True)
write_pos = old_n
seen2 = set()
for i, (npy_f, ids_f) in enumerate(zip(partial_npy, partial_ids_files)):
    ids  = json.load(open(ids_f))
    arr  = np.load(npy_f, mmap_mode="r")
    for j, pid in enumerate(ids):
        if pid not in seen2:
            seen2.add(pid)
            merged[write_pos] = arr[j]
            write_pos += 1
    del arr
    if i % 50 == 0:
        print(f"  File {i+1}/{len(partial_npy)} | written: {write_pos-old_n:,}", flush=True)
del merged
print(f"New vectors appended: {write_pos-old_n:,} ✓", flush=True)

# Swap
os.rename("embeddings.npy", "embeddings_old.npy")
os.rename("embeddings_merged.npy", "embeddings.npy")

# Update IDs — stream one file at a time
print("Updating paper_ids.json...", flush=True)
existing_ids = json.load(open("paper_ids.json"))
seen3 = set(existing_ids)
with open("paper_ids_new.json", "w") as f:
    # Write existing
    json.dump(existing_ids, f)
del existing_ids

# Append new unique IDs
new_unique_ids = []
seen4 = set()
for ids_f in partial_ids_files:
    for pid in json.load(open(ids_f)):
        if pid not in seen3 and pid not in seen4:
            seen4.add(pid)
            new_unique_ids.append(pid)

all_ids = json.load(open("paper_ids_new.json")) + new_unique_ids
json.dump(all_ids, open("paper_ids.json", "w"))
print(f"paper_ids.json: {len(all_ids):,} total ✓", flush=True)
print("Done! Next: update usearch index", flush=True)
