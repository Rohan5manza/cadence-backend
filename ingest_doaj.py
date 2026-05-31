"""
ingest_doaj.py — Download DOAJ articles via OAI-PMH protocol
OAI-PMH is designed for bulk harvesting with no result limits
Uses resumption tokens to page through all 12M+ records
Checkpoint-safe: saves resumption token after every page

Usage: python ingest_doaj.py
Output: papers_doaj.jsonl

Note: Also email dominic@doaj.org for direct bulk dump access (faster)
      Article metadata is CC0 (public domain) — free to use commercially
"""

import os
import json
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

OUTPUT_FILE     = "papers_doaj.jsonl"
CHECKPOINT_FILE = "checkpoints/doaj_checkpoint.json"

# DOAJ OAI-PMH endpoint
OAI_URL    = "https://doaj.org/oai.article"
OAI_NS     = {
    "oai":  "http://www.openarchives.org/OAI/2.0/",
    "dc":   "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}

os.makedirs("checkpoints", exist_ok=True)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"resumption_token": None, "total_papers": 0, "seen_ids": []}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f)

def fetch_oai_page(resumption_token: str = None) -> ET.Element:
    """Fetch one page of OAI-PMH records."""
    if resumption_token:
        params = {
            "verb":             "ListRecords",
            "resumptionToken":  resumption_token,
        }
    else:
        params = {
            "verb":             "ListRecords",
            "metadataPrefix":   "oai_dc",
        }

    for attempt in range(5):
        try:
            r = requests.get(OAI_URL, params=params, timeout=60)
            if r.status_code == 200:
                return ET.fromstring(r.content)
            print(f"  [warn] HTTP {r.status_code}, attempt {attempt+1}")
        except Exception as e:
            print(f"  [warn] Attempt {attempt+1}: {e}")
        time.sleep(10 * (attempt + 1))
    return None

def extract_paper(record: ET.Element) -> dict | None:
    """Extract paper fields from an OAI-PMH DC record."""
    try:
        # Check if record is deleted
        header = record.find("oai:header", OAI_NS)
        if header is not None and header.get("status") == "deleted":
            return None

        # Get OAI identifier
        identifier_el = record.find("oai:header/oai:identifier", OAI_NS)
        oai_id        = identifier_el.text.strip() if identifier_el is not None else ""
        if not oai_id:
            return None

        # Get DC metadata
        dc = record.find(".//oai_dc:dc", OAI_NS)
        if dc is None:
            # Try without namespace
            dc = record.find(".//{http://www.openarchives.org/OAI/2.0/oai_dc/}dc")
        if dc is None:
            return None

        def get_all(tag):
            results = dc.findall(f"dc:{tag}", OAI_NS)
            if not results:
                results = dc.findall(f"{{http://purl.org/dc/elements/1.1/}}{tag}")
            return [el.text.strip() for el in results if el.text and el.text.strip()]

        title     = get_all("title")
        abstracts = get_all("description")
        creators  = get_all("creator")
        dates     = get_all("date")
        subjects  = get_all("subject")
        publishers = get_all("publisher")
        identifiers = get_all("identifier")

        if not title or not abstracts:
            return None

        title_str    = title[0]
        abstract_str = " ".join(abstracts).strip()

        if len(abstract_str) < 50:
            return None

        # Extract year from date
        year = None
        for d in dates:
            try:
                year = int(d[:4])
                if 1900 <= year <= 2030:
                    break
            except: pass

        # Extract DOI from identifiers
        doi = None
        oa_url = None
        for ident in identifiers:
            if "doi.org/" in ident.lower() or ident.lower().startswith("10."):
                doi = ident.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
            elif ident.startswith("http"):
                oa_url = ident

        # Clean paper_id from OAI identifier
        # Format: oai:doaj.org/article:abc123def456
        paper_id = oai_id.replace("oai:doaj.org/article:", "").replace(":", "_")
        if not paper_id:
            return None

        return {
            "paper_id":        f"doaj_{paper_id}",
            "title":           title_str,
            "abstract":        abstract_str,
            "authors":         creators[:10],
            "year":            year,
            "venue":           publishers[0] if publishers else "",
            "doi":             doi,
            "arxiv_id":        None,
            "categories":      subjects[:5],
            "source":          "doaj",
            "citation_count":  None,
            "open_access_url": oa_url,
        }
    except Exception:
        return None

def main():
    cp              = load_checkpoint()
    resumption_token = cp.get("resumption_token")
    seen_ids        = set(cp.get("seen_ids", []))

    print(f"[doaj] OAI-PMH harvester starting")
    print(f"[doaj] Checkpoint: {cp['total_papers']:,} papers | "
          f"token: {'yes' if resumption_token else 'fresh start'}")

    output   = open(OUTPUT_FILE, "a")
    page_num = 0

    while True:
        print(f"[doaj] Fetching page {page_num + 1}...")
        root = fetch_oai_page(resumption_token)

        if root is None:
            print("[doaj] Failed after retries, stopping")
            break

        # Check for OAI errors
        error = root.find(".//oai:error", OAI_NS)
        if error is not None:
            code = error.get("code", "")
            print(f"[doaj] OAI error: {code} — {error.text}")
            if code == "noRecordsMatch":
                print("[doaj] No more records — done!")
            break

        # Extract records
        records = root.findall(".//oai:record", OAI_NS)
        added   = 0

        for record in records:
            paper = extract_paper(record)
            if not paper:
                continue
            pid = paper["paper_id"]
            if pid in seen_ids:
                continue
            output.write(json.dumps(paper) + "\n")
            seen_ids.add(pid)
            cp["total_papers"] += 1
            added += 1

        page_num += 1

        # Get resumption token for next page
        token_el         = root.find(".//oai:resumptionToken", OAI_NS)
        resumption_token = token_el.text.strip() if (token_el is not None and token_el.text) else None

        # Save checkpoint
        cp["resumption_token"] = resumption_token
        cp["seen_ids"]         = list(seen_ids)[-200_000:]
        save_checkpoint(cp)

        # Get cursor/complete list size if available
        cursor        = token_el.get("cursor", "?") if token_el is not None else "?"
        complete_size = token_el.get("completeListSize", "?") if token_el is not None else "?"

        print(f"  [doaj] Page {page_num} | +{added} | total: {cp['total_papers']:,} | "
              f"cursor: {cursor}/{complete_size}")

        if not resumption_token:
            print("[doaj] No resumption token — harvest complete!")
            break

        # Be polite to the server
        time.sleep(1)

    output.close()
    print(f"\n[doaj] Done! {cp['total_papers']:,} papers → {OUTPUT_FILE}")
    print(f"\nTip: For faster bulk access email dominic@doaj.org")
    print(f"     Article metadata is CC0 — free for any use")

if __name__ == "__main__":
    main()