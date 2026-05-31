"""
ingest_biorxiv.py — Download bioRxiv and medRxiv preprints via API
Fixed pagination: uses cursor arithmetic, not collection length check
Checkpoint-safe: saves every page

Usage: python ingest_biorxiv.py
Output: papers_biorxiv.jsonl
"""

import os
import json
import time
import requests
from datetime import datetime

OUTPUT_FILE     = "papers_biorxiv.jsonl"
CHECKPOINT_FILE = "checkpoints/biorxiv_checkpoint.json"
SERVERS         = ["biorxiv", "medrxiv"]
START_DATE      = "2013-01-01"
END_DATE        = datetime.today().strftime("%Y-%m-%d")
PAGE_SIZE       = 100

os.makedirs("checkpoints", exist_ok=True)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"server_cursors": {}, "server_totals": {}, "total_papers": 0, "seen_dois": []}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f)

def fetch_page(server: str, cursor: int) -> dict | None:
    url = f"https://api.biorxiv.org/details/{server}/{START_DATE}/{END_DATE}/{cursor}/json"
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                data = r.json()
                # Validate response structure
                if "messages" in data and "collection" in data:
                    return data
                print(f"  [warn] Unexpected response structure: {list(data.keys())}")
                return None
            print(f"  [warn] HTTP {r.status_code}, attempt {attempt+1}")
        except Exception as e:
            print(f"  [warn] Attempt {attempt+1}: {e}")
        time.sleep(10 * (attempt + 1))
    return None

def main():
    cp        = load_checkpoint()
    seen_dois = set(cp.get("seen_dois", []))
    print(f"[biorxiv] Checkpoint: {cp['total_papers']:,} papers done")

    output = open(OUTPUT_FILE, "a")

    for server in SERVERS:
        cursor = cp["server_cursors"].get(server, 0)
        total  = cp["server_totals"].get(server, None)
        print(f"\n[biorxiv] Starting {server} from cursor {cursor} (total: {total})")

        while True:
            data = fetch_page(server, cursor)
            if not data:
                print(f"  [error] Failed after retries, stopping {server}")
                break

            # Parse messages
            messages   = data.get("messages", [])
            collection = data.get("collection", [])

            if messages:
                msg = messages[0]
                try:
                    total = int(msg.get("total", total or 0))
                except (ValueError, TypeError):
                    pass
                cp["server_totals"][server] = total

            if not collection:
                print(f"  [done] Empty collection at cursor {cursor}, finished {server}")
                break

            added = 0
            for item in collection:
                doi = item.get("doi", "")
                if not doi or doi in seen_dois:
                    continue

                abstract = item.get("abstract", "").strip()
                if not abstract or len(abstract) < 50:
                    continue

                paper = {
                    "paper_id":        f"{server}_{doi.replace('/', '_')}",
                    "title":           item.get("title", "").strip(),
                    "abstract":        abstract,
                    "authors":         [a.strip() for a in item.get("authors", "").split(";") if a.strip()][:10],
                    "year":            int(item.get("date", "2020")[:4]) if item.get("date") else None,
                    "venue":           server,
                    "doi":             doi,
                    "arxiv_id":        None,
                    "categories":      [item.get("category", "").strip()] if item.get("category") else [],
                    "source":          server,
                    "citation_count":  None,
                    "open_access_url": f"https://www.{server}.org/content/{doi}v1.full.pdf",
                }
                output.write(json.dumps(paper) + "\n")
                seen_dois.add(doi)
                cp["total_papers"] += 1
                added += 1

            cursor += len(collection)
            cp["server_cursors"][server] = cursor
            cp["seen_dois"] = list(seen_dois)[-100000:]
            save_checkpoint(cp)

            pct = f"{cursor/total*100:.1f}%" if total else "?"
            print(f"  [{server}] {cursor:,}/{total} ({pct}) | +{added} | total: {cp['total_papers']:,}")

            # Done when we've fetched all papers
            if total and cursor >= total:
                print(f"  [done] Finished {server}")
                break

            time.sleep(0.5)

    output.close()
    print(f"\n[biorxiv] Done! {cp['total_papers']:,} papers → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()