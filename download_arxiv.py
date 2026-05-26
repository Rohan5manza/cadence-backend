"""
Download and parse Kaggle ArXiv snapshot into unified paper format.
Output: papers_arxiv.jsonl
"""
import json
import os
import subprocess
from tqdm import tqdm

# ── Kaggle auth ───────────────────────────────────────────────────────────────
os.environ["KAGGLE_KEY"]      = "KGAT_bc3a1af65c90a2b405bcecafded472eb"
os.environ["KAGGLE_USERNAME"] = "rohanmarar"

# Write kaggle.json from env vars
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    json.dump({
        "username": os.environ["KAGGLE_USERNAME"],
        "key":      os.environ["KAGGLE_KEY"]
    }, f)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

# ── Download ──────────────────────────────────────────────────────────────────
INPUT_DIR  = "./arxiv_raw"
INPUT_FILE = os.path.join(INPUT_DIR, "arxiv-metadata-oai-snapshot.json")
OUTPUT     = "papers_arxiv.jsonl"
MAX_PAPERS = 500_000

os.makedirs(INPUT_DIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    print("Downloading ArXiv dataset from Kaggle...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "Cornell-University/arxiv",
        "-p", INPUT_DIR,
        "--unzip"
    ], check=True)
else:
    print(f"Found existing {INPUT_FILE}, skipping download.")

# ── Parse ─────────────────────────────────────────────────────────────────────
print(f"\nParsing {INPUT_FILE}...")
papers  = 0
skipped = 0

with open(INPUT_FILE) as in_f, open(OUTPUT, "w") as out_f:
    for line in tqdm(in_f, desc="Parsing ArXiv"):
        try:
            row      = json.loads(line)
            abstract = (row.get("abstract") or "").replace("\n", " ").strip()
            title    = (row.get("title") or "").replace("\n", " ").strip()

            if len(abstract) < 100 or not title:
                skipped += 1
                continue

            paper = {
                "paper_id":   "arxiv_" + row["id"],
                "source":     "arxiv",
                "title":      title,
                "abstract":   abstract,
                "doi":        row.get("doi"),
                "arxiv_id":   row["id"],
                "categories": row.get("categories", ""),
                "refs":       [],
            }

            out_f.write(json.dumps(paper) + "\n")
            out_f.flush()
            papers += 1

        except Exception:
            skipped += 1
            continue

        if papers >= MAX_PAPERS:
            break

print(f"\nDone.")
print(f"Papers saved : {papers:,}")
print(f"Skipped      : {skipped:,}")