"""
ingest_preprints.py — Download from OSF preprint servers
Fixed to match actual OSF API v2 response structure
Checkpoint-safe: saves every page

Usage: python ingest_preprints.py
Output: papers_preprints.jsonl
"""

import os
import json
import time
import requests

OUTPUT_FILE     = "papers_preprints.jsonl"
CHECKPOINT_FILE = "checkpoints/preprints_checkpoint.json"

SERVERS = [
    {"id": "psyarxiv",   "label": "PsyArXiv",   "field": "psychology"},
    {"id": "socarxiv",   "label": "SocArXiv",   "field": "social sciences"},
    {"id": "chemrxiv",   "label": "ChemRxiv",   "field": "chemistry"},
    {"id": "eartharxiv", "label": "EarthArXiv", "field": "earth sciences"},
    {"id": "engrxiv",    "label": "EngrXiv",    "field": "engineering"},
    {"id": "mindrxiv",   "label": "MindRxiv",   "field": "mind sciences"},
    {"id": "africarxiv", "label": "AfricArXiv", "field": "african research"},
    {"id": "lawarxiv",   "label": "LawArXiv",   "field": "law"},
]

os.makedirs("checkpoints", exist_ok=True)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"server_next_urls": {}, "total_papers": 0, "seen_ids": []}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f)

def fetch_page(url: str) -> dict:
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            print(f"  [warn] HTTP {r.status_code}, attempt {attempt+1}")
        except Exception as e:
            print(f"  [warn] Attempt {attempt+1}: {e}")
        time.sleep(5 * (attempt + 1))
    return None

def extract_paper(item: dict, server: dict) -> dict | None:
    try:
        attrs       = item.get("attributes", {})
        preprint_id = item.get("id", "")

        title    = attrs.get("title", "").strip()
        abstract = attrs.get("description", "").strip()  # OSF uses 'description' not 'abstract'

        if not title or not abstract or len(abstract) < 50:
            return None

        # Year from date_published
        year = None
        date = attrs.get("date_published") or attrs.get("date_created", "")
        if date:
            try: year = int(date[:4])
            except: pass

        # DOI — from attributes.doi or links.preprint_doi
        doi = attrs.get("doi")
        if not doi:
            links = item.get("links", {})
            preprint_doi = links.get("preprint_doi", "")
            if preprint_doi and "doi.org/" in preprint_doi:
                doi = preprint_doi.split("doi.org/")[-1]

        # Categories from subjects
        categories = [server["field"]]
        for subj_group in attrs.get("subjects", []):
            for subj in subj_group:
                text = subj.get("text", "").strip()
                if text and text not in categories:
                    categories.append(text)
        categories = categories[:5]

        # Tags
        tags = attrs.get("tags", [])[:3]

        # Open access URL
        oa_url = item.get("links", {}).get("html", "")

        return {
            "paper_id":        f"{server['id']}_{preprint_id}",
            "title":           title,
            "abstract":        abstract,
            "authors":         [],  # would need separate API call per paper
            "year":            year,
            "venue":           server["label"],
            "doi":             doi or None,
            "arxiv_id":        None,
            "categories":      categories,
            "source":          server["id"],
            "citation_count":  None,
            "open_access_url": oa_url or None,
        }
    except Exception:
        return None

def main():
    cp       = load_checkpoint()
    seen_ids = set(cp.get("seen_ids", []))
    print(f"[preprints] Checkpoint: {cp['total_papers']:,} papers done")

    output = open(OUTPUT_FILE, "a")

    for server in SERVERS:
        sid = server["id"]

        # Resume from saved next URL or start fresh
        next_url = cp["server_next_urls"].get(sid)
        if not next_url:
            next_url = (
                f"https://api.osf.io/v2/preprints/"
                f"?filter%5Bprovider%5D={sid}"
                f"&page%5Bsize%5D=100"
                f"&sort=-date_created"
            )

        print(f"\n[preprints] Starting {server['label']}...")
        page_count = 0
        server_total = 0

        while next_url:
            data = fetch_page(next_url)
            if not data:
                print(f"  [error] Failed after retries, stopping {sid}")
                break

            items    = data.get("data", [])
            links    = data.get("links", {})
            meta     = data.get("meta", {})
            total    = meta.get("total", "?")
            next_url = links.get("next")  # None when last page

            added = 0
            for item in items:
                item_id = item.get("id", "")
                if item_id in seen_ids:
                    continue
                paper = extract_paper(item, server)
                if paper:
                    output.write(json.dumps(paper) + "\n")
                    seen_ids.add(item_id)
                    cp["total_papers"] += 1
                    server_total += 1
                    added += 1

            page_count += 1
            cp["server_next_urls"][sid] = next_url
            cp["seen_ids"] = list(seen_ids)[-100000:]
            save_checkpoint(cp)

            print(f"  [{sid}] page {page_count} | +{added} | server: {server_total} | grand total: {cp['total_papers']:,} / {total}")

            if not next_url:
                print(f"  [done] Finished {server['label']}: {server_total} papers")
                break

            time.sleep(0.5)

    output.close()
    print(f"\n[preprints] Done! {cp['total_papers']:,} papers → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()