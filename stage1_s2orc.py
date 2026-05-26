"""
Stage 1: Download and parse S2ORC shards to build paper index.
Targets 200k papers for stronger foundation in Path B.
"""
import json
import os
import re
import gzip
import requests
import wget
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = "Ed4VXsSEcSaWFxMyS7SxJ9CkX1dUgFmP6OQ3VfO1"
NUM_SHARDS = 8
LOCAL_PATH = "./s2orc_shards"
OUTPUT     = "papers_s2orc.jsonl"   # renamed for clarity in multi-source setup
MAX_PAPERS = 200_000
MIN_ABSTRACT_LEN = 100
# ─────────────────────────────────────────────────────────────────────────────


def download_shards():
    os.makedirs(LOCAL_PATH, exist_ok=True)

    print("Fetching latest release ID...")
    release_id = requests.get(
        "https://api.semanticscholar.org/datasets/v1/release/latest"
    ).json()["release_id"]
    print(f"Latest release: {release_id}")

    print("Fetching shard URLs...")
    response = requests.get(
        f"https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/s2orc/",
        headers={"x-api-key": API_KEY}
    ).json()

    all_urls = response["files"]
    print(f"Total shards: {len(all_urls)} — downloading {NUM_SHARDS}")

    for url in tqdm(all_urls[:NUM_SHARDS], desc="Downloading"):
        match = re.match(
            r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*\.gz).*", url
        )
        shard_name = match.group(2)
        out_path   = os.path.join(LOCAL_PATH, shard_name)
        if os.path.exists(out_path):
            print(f"  {shard_name} already exists, skipping")
        else:
            wget.download(url, out=out_path)
            print()


def parse_papers():
    print(f"Parsing shards into {OUTPUT}...")
    papers = []

    for fname in sorted(os.listdir(LOCAL_PATH)):
        if not fname.endswith(".gz"):
            continue
        print(f"  Parsing {fname[:50]}...")

        with gzip.open(os.path.join(LOCAL_PATH, fname), "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)

                    text = row.get("content", {}).get("text", "")
                    ann  = row.get("content", {}).get("annotations", {})

                    # Extract abstract via offsets
                    abstract = ""
                    if ann.get("abstract"):
                        spans = json.loads(ann["abstract"]) if isinstance(ann["abstract"], str) else ann["abstract"]
                        abstract = " ".join(
                            text[s["start"]:s["end"]].replace("\n", " ").strip()
                            for s in spans
                        )

                    if len(abstract) < MIN_ABSTRACT_LEN:
                        continue

                    # Extract title via offsets
                    title = ""
                    if ann.get("title"):
                        spans = json.loads(ann["title"]) if isinstance(ann["title"], str) else ann["title"]
                        if spans:
                            title = text[spans[0]["start"]:spans[0]["end"]].replace("\n", " ").strip()

                    papers.append({
                        "paper_id": row.get("corpusid", ""),
                        "source":   "s2orc",
                        "title":    title,
                        "abstract": abstract,
                        "doi":      (row.get("externalids") or {}).get("doi"),
                        "arxiv_id": (row.get("externalids") or {}).get("arxiv"),
                    })

                except Exception:
                    continue

                if len(papers) >= MAX_PAPERS:
                    break

        if len(papers) >= MAX_PAPERS:
            break

    with open(OUTPUT, "w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")

    print(f"\nSaved {len(papers)} papers to {OUTPUT}")
    if papers:
        print(f"Sample: {papers[0]['title'][:80]}")


if __name__ == "__main__":
    download_shards()
    parse_papers()