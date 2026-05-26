"""
Stage 2: Download OpenAlex works snapshot — fixed version.

OpenAlex organizes works by updated_date partition. The small files
(< 50MB) are incremental updates. The bulk data files are large (500MB+).
We filter to only large files and use unique output names per partition.
"""
import os
import subprocess
import sys

# ── Config ────────────────────────────────────────────────────────────────────
LOCAL_PATH   = "./openalex_snapshot"
NUM_FILES    = 8        # number of large bulk files to download
MIN_SIZE_MB  = 100      # skip files smaller than this (incremental updates)
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(LOCAL_PATH, exist_ok=True)


def check_aws():
    try:
        r = subprocess.run(["aws", "--version"], capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


if not check_aws():
    print("AWS CLI not installed. Run: pip install awscli")
    sys.exit(1)

print("Listing OpenAlex works files (takes ~1 min)...")
result = subprocess.run([
    "aws", "s3", "ls",
    "s3://openalex/data/works/",
    "--no-sign-request",
    "--recursive"
], capture_output=True, text=True)

if result.returncode != 0:
    print("Failed:", result.stderr)
    sys.exit(1)

# Parse: DATE TIME SIZE PATH
all_files = []
for line in result.stdout.strip().split("\n"):
    parts = line.split()
    if len(parts) >= 4 and parts[-1].endswith(".gz"):
        try:
            size_bytes = int(parts[2])
            path       = parts[-1]
            all_files.append((size_bytes, path))
        except ValueError:
            continue

print(f"Total files found: {len(all_files)}")

# Filter to large files only (bulk data, not incremental updates)
large_files = [
    (size, path) for size, path in all_files
    if size >= MIN_SIZE_MB * 1_000_000
]
large_files.sort(key=lambda x: -x[0])  # largest first
print(f"Large files (>{MIN_SIZE_MB}MB): {len(large_files)}")
print(f"Targeting first {NUM_FILES}\n")

# ── Download with unique filenames ────────────────────────────────────────────
for i, (size_bytes, remote_path) in enumerate(large_files[:NUM_FILES]):
    # Use partition date + index as unique filename e.g. works_0001.gz
    out_name = f"works_{i:04d}.gz"
    out_path = os.path.join(LOCAL_PATH, out_name)
    size_gb  = size_bytes / 1e9

    if os.path.exists(out_path):
        local_size = os.path.getsize(out_path)
        if local_size == size_bytes:
            print(f"  ✓ {out_name} complete ({size_gb:.2f}GB)")
            continue
        else:
            print(f"  ✗ {out_name} partial — re-downloading")
            os.remove(out_path)

    print(f"Downloading {out_name} ({size_gb:.2f}GB) from {remote_path}...")
    r = subprocess.run([
        "aws", "s3", "cp",
        f"s3://openalex/{remote_path}",
        out_path,
        "--no-sign-request"
    ])
    if r.returncode == 0:
        print(f"  ✓ {out_name} done")
    else:
        print(f"  ✗ {out_name} failed")

print("\nDone.")
print(f"Files in {LOCAL_PATH}/:")
for f in sorted(os.listdir(LOCAL_PATH)):
    size = os.path.getsize(os.path.join(LOCAL_PATH, f))
    print(f"  {f}  {size/1e9:.2f}GB")