"""
ingest_pubmed.py — Download PubMed Central OA papers via bulk XML tar.gz files
Uses 13 baseline files (each contains ~350K papers) instead of 5.2M individual files
RAM-safe: streams XML with iterparse
Checkpoint-safe: resumes from last completed bulk file

Usage: python ingest_pubmed.py
Output: papers_pubmed.jsonl
"""

import os
import json
import gzip
import hashlib
import tarfile
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

BULK_BASE_URL   = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_comm/xml/"
OUTPUT_FILE     = "papers_pubmed.jsonl"
CHECKPOINT_FILE = "checkpoints/pubmed_checkpoint.json"
DOWNLOAD_DIR    = "pubmed_raw"

# All 13 baseline bulk files
BULK_FILES = [
    "oa_comm_xml.PMC000xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC001xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC002xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC003xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC004xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC005xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC006xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC007xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC008xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC009xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC010xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC011xxxxxx.baseline.2026-01-23.tar.gz",
    "oa_comm_xml.PMC012xxxxxx.baseline.2026-01-23.tar.gz",
]

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed_files": [], "total_papers": 0}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f)

def download_file(url: str, dest: str) -> bool:
    """Download with progress reporting."""
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f"\r  Downloading... {downloaded/1024/1024/1024:.1f}GB / {total_size/1024/1024/1024:.1f}GB ({pct:.0f}%)", end="", flush=True)
            print()
            return True
        except Exception as e:
            print(f"\n  [warn] Attempt {attempt+1} failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)
    return False

def parse_nxml_streaming(xml_content: bytes) -> list:
    """Parse a single NXML file from bytes using iterparse."""
    papers = []
    try:
        import io
        context = ET.iterparse(io.BytesIO(xml_content), events=("end",))
        for event, elem in context:
            if not elem.tag.endswith("article"):
                continue
            try:
                title_el = elem.find(".//{*}article-title")
                title    = "".join(title_el.itertext()).strip() if title_el is not None else ""
                if not title:
                    elem.clear(); continue

                abstract_parts = [
                    "".join(p.itertext()).strip()
                    for p in elem.findall(".//{*}abstract//{*}p")
                ]
                abstract = " ".join(abstract_parts).strip()
                if len(abstract) < 50:
                    elem.clear(); continue

                authors = []
                for contrib in elem.findall(".//{*}contrib[@contrib-type='author']"):
                    sn = contrib.findtext(".//{*}surname", "")
                    gn = contrib.findtext(".//{*}given-names", "")
                    if sn: authors.append(f"{gn} {sn}".strip())

                year = None
                for tag in [".//{*}pub-date[@pub-type='epub']",
                            ".//{*}pub-date[@pub-type='ppub']",
                            ".//{*}pub-date"]:
                    pd = elem.find(tag)
                    if pd is not None:
                        try: year = int(pd.findtext(".//{*}year", "")); break
                        except: pass

                doi = pmc_id = None
                for aid in elem.findall(".//{*}article-id"):
                    t = aid.get("pub-id-type", "")
                    if t == "doi": doi    = aid.text
                    if t == "pmc": pmc_id = f"PMC{aid.text}"

                venue      = elem.findtext(".//{*}journal-title", "")
                categories = [
                    "".join(kw.itertext()).strip()[:50]
                    for kw in elem.findall(".//{*}kwd") if kw.text
                ][:5]

                paper_id = pmc_id or (
                    f"doi_{hashlib.md5(doi.encode()).hexdigest()[:12]}" if doi else None
                )
                if not paper_id:
                    elem.clear(); continue

                papers.append({
                    "paper_id":        paper_id,
                    "title":           title,
                    "abstract":        abstract,
                    "authors":         authors[:10],
                    "year":            year,
                    "venue":           venue,
                    "doi":             doi,
                    "arxiv_id":        None,
                    "categories":      categories,
                    "source":          "pubmed",
                    "citation_count":  None,
                    "open_access_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/" if pmc_id else None,
                })
            except Exception:
                pass
            finally:
                elem.clear()
    except Exception as e:
        print(f"  [warn] Parse error: {e}")
    return papers

def process_bulk_tar(tar_path: str, output, cp: dict) -> int:
    """
    Stream through a bulk tar.gz file, parsing each NXML inside.
    Never extracts all files at once — processes one at a time.
    """
    total_papers = 0
    file_count   = 0

    print(f"  Processing tar file (streaming)...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if not (member.name.endswith(".nxml") or member.name.endswith(".xml")):
                    continue

                try:
                    f       = tar.extractfile(member)
                    if f is None: continue
                    content = f.read()
                    papers  = parse_nxml_streaming(content)

                    for p in papers:
                        output.write(json.dumps(p) + "\n")

                    total_papers += len(papers)
                    file_count   += 1

                    if file_count % 10000 == 0:
                        print(f"  Processed {file_count:,} XML files, {total_papers:,} papers so far...")
                        cp["total_papers"] += total_papers
                        save_checkpoint(cp)
                        total_papers = 0  # reset counter after checkpoint

                except Exception as e:
                    continue

    except Exception as e:
        print(f"  [error] Failed to process tar: {e}")

    return total_papers

def main():
    cp = load_checkpoint()
    # Reset checkpoint from old individual-file approach
    if cp.get("processed_files") and any("PMC2" in f for f in cp.get("processed_files", [])):
        print("[pubmed] Detected old checkpoint format, resetting...")
        cp = {"processed_files": [], "total_papers": 0}
        save_checkpoint(cp)

    processed_set = set(cp["processed_files"])
    print(f"[pubmed] Checkpoint: {len(processed_set)}/{len(BULK_FILES)} bulk files done, "
          f"{cp['total_papers']:,} papers")

    output = open(OUTPUT_FILE, "a")

    for i, filename in enumerate(BULK_FILES):
        if filename in processed_set:
            print(f"[pubmed] [{i+1}/{len(BULK_FILES)}] Skipping {filename} (done)")
            continue

        tar_path = os.path.join(DOWNLOAD_DIR, filename)
        url      = BULK_BASE_URL + filename

        print(f"\n[pubmed] [{i+1}/{len(BULK_FILES)}] {filename}")

        # Download if not already present
        if not os.path.exists(tar_path):
            print(f"  Downloading from {url}...")
            if not download_file(url, tar_path):
                print(f"  [error] Download failed, skipping")
                continue
            print(f"  Downloaded ✓")

        # Process — stream through tar without full extraction
        print(f"  Processing {filename}...")
        before       = cp["total_papers"]
        new_papers   = process_bulk_tar(tar_path, output, cp)
        cp["total_papers"] += new_papers
        cp["processed_files"].append(filename)
        save_checkpoint(cp)

        added = cp["total_papers"] - before
        print(f"  ✓ +{added:,} papers | Running total: {cp['total_papers']:,}")

        # Delete tar after processing to save disk space
        os.remove(tar_path)
        print(f"  Deleted {filename} to save space")

    output.close()
    print(f"\n[pubmed] Done! {cp['total_papers']:,} papers → {OUTPUT_FILE}")
    print(f"[pubmed] Note: Also run incremental files for papers after Jan 2026")

if __name__ == "__main__":
    main()