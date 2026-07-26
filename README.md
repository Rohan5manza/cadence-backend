# Cadence Backend

> AI-powered research paper discovery — "Spotify for research papers"

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)

Cadence is a personalized research paper discovery platform. This repository contains the full backend: the FastAPI server, the ML pipeline (corpus ingestion, embedding, fine-tuning), and the nightly ingestion cron job.

**Live demo:** [cadence.rohanmarar.com](https://cadence.rohanmarar.com)  
**Frontend repo:** [github.com/Rohan5manza/cadence-app](https://github.com/Rohan5manza/cadence-frontend)  
**Embedding model:** [huggingface.co/rohan5manza/cadence-specter2](https://huggingface.co/rohan5manza/cadence-specter2)

---

## Embedding Model (Fine-tuned by me)

Cadence uses a fine-tuned version of [SPECTER2](https://huggingface.co/allenai/specter2_base) — available at my HF profile [rohan5manza/cadence-specter2](https://huggingface.co/rohan5manza/cadence-specter2).

| Paper Pair | Base SPECTER2 | Cadence Fine-tuned |
|------------|--------------|-------------------|
| Attention is All You Need ↔ BERT | 0.833 | **0.871** |
| Attention is All You Need ↔ Cancer Research Paper | 0.852 | **-0.044** |
| **Similarity gap** | **-0.019** ❌ | **+0.914** ✅ |


**The base model was confused** — it rated a cancer paper as *more* 
similar to a transformer paper than BERT was (gap: -0.019, wrong direction).

**After fine-tuning** — the model has clear domain separation. 
The cancer paper scores negative similarity against transformer papers, 
meaning it's on the opposite side of the embedding space entirely. 
The gap flipped from -0.019 to +0.914.

This directly improves recommendation quality: when a user's taste vector 
points toward NLP/transformers, the nearest neighbors are precisely relevant 
papers — not papers from unrelated domains that happen to share academic language.

Trained on 267,841 triplets · 3 epochs · Final loss: 0.042 · RTX 4060 Ti


## Using the Embedding Model

The fine-tuned model is publicly available and can be used independently of Cadence in any scientific NLP workflow.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("rohan5manza/cadence-specter2")

embeddings = model.encode(
    ["Attention Is All You Need [SEP] The dominant sequence transduction models..."],
    normalize_embeddings=True
)
```

**Important:** Use `[SEP]` to separate title from abstract — this matches the training format. Always set `normalize_embeddings=True` for cosine similarity. Output is 768-dimensional float32 vectors.

### Use cases

- Scientific paper recommendation systems
- Semantic search over research literature
- Paper clustering by field or topic
- Finding similar papers given a query
- Building citation recommendation systems

### When it outperforms base SPECTER2

The fine-tuned model is stronger when you need clear domain separation — CS papers clustering away from biology papers, NLP papers clustering away from clinical medicine. The +0.914 similarity gap between related and unrelated domains (vs -0.019 for base SPECTER2) makes it significantly better for multi-domain corpora.

### When base SPECTER2 may be preferable

For highly specialized single-domain tasks (e.g. purely biomedical), or for citation prediction specifically, the base model may perform comparably. Our fine-tuning optimized for cross-domain separation across a broad multi-field corpus.


## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [The ML Pipeline](#the-ml-pipeline)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Contributing](#contributing)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cadence Backend                          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  FastAPI     │    │  usearch     │    │  paper_meta      │  │
│  │  (main.py)   │───▶│  (ANN index) │    │  (in-memory      │  │
│  │              │    │  cadence     │    │   dict, 2.28M    │  │
│  │  8000/tcp    │    │  .usearch    │    │   papers)        │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│          │                                                        │
│          │                                                        │
│  ┌───────▼────────────────────────────────────────────────────┐ │
│  │                    Recommendation Engine                    │ │
│  │                                                             │ │
│  │  1. Load user interactions from Supabase                   │ │
│  │  2. Compute taste vector (weighted avg of paper embeddings) │ │
│  │  3. Query usearch for 20 nearest neighbors                 │ │
│  │  4. Filter by user profile (field, role, reading goal)     │ │
│  │  5. Return personalized feed                               │ │
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Supabase    │    │  embeddings  │    │  SPECTER2        │  │
│  │  PostgreSQL  │    │  .npy        │    │  fine-tuned      │  │
│  │              │    │  (2.28M×768) │    │  model           │  │
│  │  users       │    │              │    │                  │  │
│  │  interactions│    │  paper_ids   │    │  generates 768-d │  │
│  │  saved_papers│    │  .json       │    │  embeddings      │  │
│  │  playlists   │    │              │    │                  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow for a personalized feed request:**
```
GET /feed/discover
    ↓
Fetch user interactions from Supabase (last 50)
    ↓
Compute taste vector:
  taste = Σ (interaction_weight × paper_embedding × recency_decay)
  weights: save=3.0, like=1.5, read=1.0, share=2.0, skip=-1.5
  decay: 0.95^days_since_interaction
    ↓
usearch.search(taste_vector, top_100)
    ↓
Filter by user profile (field, role, experience, reading_goal)
    ↓
Apply 10% diversity injection
    ↓
Return top 20 papers
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU with 8GB+ VRAM (for embedding, not required for API only)
- 16GB+ RAM (for serving 2.28M paper index)
- Ubuntu 22.04+ (or any Linux)

### 1. Clone and set up environment

```bash
git clone https://github.com/Rohan5manza/cadence-backend.git
cd cadence-backend

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required variables:
```
DB_URL=postgresql://postgres:PASSWORD@db.YOUR_PROJECT.supabase.co:5432/postgres
SECRET_KEY=your-secret-key-min-32-chars
```

### 3. Set up Supabase

Create a Supabase project at supabase.com, then run the schema:

```bash
# Copy the SQL from schema.sql and run in Supabase SQL editor
```

### 4. Build the corpus (or download pre-built)

**Option A — Download pre-built corpus (recommended for getting started):**
```bash
# Download from HuggingFace datasets (coming soon)
# For now, run the ingestion scripts (see ML Pipeline section)
```

**Option B — Build from scratch:**
```bash
# See ML Pipeline section below
```

### 5. Start the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Visit `http://localhost:8000/health` — should return:
```json
{"status": "ok", "papers_loaded": 2288250, "index_ready": true}
```

---

## Repository Structure

```
cadence-backend/
│
├── main.py                        # FastAPI application — all API endpoints
│
├── Corpus Ingestion:
│   ├── ingest_pubmed.py           # Download PubMed Central OA (~4.5M papers)
│   ├── ingest_biorxiv.py          # Download bioRxiv + medRxiv (~417K papers)
│   ├── ingest_doaj.py             # Download DOAJ via OAI-PMH (~277K papers)
│   ├── ingest_preprints.py        # Download OSF preprints (~86K papers)
│   └── download_arxiv.py          # Download arXiv papers
│
├── Original ML Pipeline (run in order for initial build):
│   ├── stage1_s2orc.py            # Parse S2ORC shards
│   ├── stage2_openalex_download.py # Download OpenAlex snapshot
│   ├── stage3_openalex_parse.py   # Parse OpenAlex into JSONL
│   ├── stage4_biblio_coupling.py  # Build bibliographic coupling pairs
│   ├── stage5_merge.py            # Merge all sources into papers_merged.jsonl
│   ├── stage6_build_triplets.py   # Build training triplets
│   ├── stage7_finetune.py         # Fine-tune SPECTER2 on triplets
│   ├── stage8_embed.py            # Embed all papers with fine-tuned model
│   └── stage9_build_index.py      # Build usearch HNSW index
│
├── Ongoing Pipeline:
│   ├── embed_new_papers.py        # Embed new papers, extend existing index
│   ├── merge_partials.py          # Merge partial embedding checkpoints
│   ├── nightly_ingest.py          # Nightly cron — fetch + embed yesterday's papers ( needs improvements, sometimes crashes in my current homelab setup)
│   └── build_triplets.py          # Build training triplets from corpus
│
├── Configuration:
│   ├── .gitignore
│   └── cadence-api.service        # systemd service file
│
├── Data files (not in git — too large):
│   ├── papers_merged.jsonl        # 2.28M paper metadata (main corpus)
│   ├── papers_merged_full.jsonl   # 7.15M extended corpus (backup)
│   ├── papers_arxiv.jsonl         # Raw arXiv papers
│   ├── papers_biorxiv.jsonl       # Raw bioRxiv/medRxiv papers
│   ├── papers_doaj.jsonl          # Raw DOAJ papers
│   ├── papers_openalex.jsonl      # Raw OpenAlex papers
│   ├── papers_preprints.jsonl     # Raw OSF preprints
│   ├── papers_pubmed.jsonl        # Raw PubMed papers
│   ├── papers_s2orc.jsonl         # Raw S2ORC papers
│   ├── embeddings.npy             # 2.28M × 768 float32 embeddings
│   ├── embeddings_old.npy         # Previous embeddings backup
│   ├── embeddings_backup.npy      # Pre-operation backup
│   ├── cadence.usearch            # usearch HNSW index (current)
│   ├── cadence_old.usearch        # Previous index backup
│   ├── paper_ids.json             # Ordered list of paper IDs
│   ├── paper_ids_new.json         # Updated paper IDs list
│   ├── pairs_biblio_coupling.jsonl # Bibliographic coupling pairs
│   ├── triplets.jsonl             # Fine-tuning training triplets
│   ├── specter2-finetuned/        # Fine-tuned model weights (→ HuggingFace)
│   ├── arxiv_raw/                 # Raw arXiv download files
│   ├── openalex_snapshot/         # Raw OpenAlex snapshot
│   ├── pubmed_raw/                # Raw PubMed download files
│   ├── s2orc_shards/              # Raw S2ORC shards
│   └── checkpoints/               # Embedding + ingestion checkpoints
│
└── Logs (not in git):
    ├── arxiv.log
    ├── biblio.log
    ├── biorxiv.log
    ├── doaj.log
    ├── embed.log
    ├── embed_new.log
    ├── finetune.log
    ├── merge_partials.log
    ├── nightly.log
    ├── openalex.log
    ├── parse_openalex.log
    ├── preprints.log
    └── pubmed.log
```

---

## The ML Pipeline

The ML pipeline converts raw paper sources into a searchable, embeddable corpus. Run these scripts **in order** once to build the initial corpus. After that, `nightly_ingest.py` keeps it current automatically.

### Step 1: Corpus Ingestion

Each script downloads papers from a different source and writes to a `.jsonl` file. All scripts are crash-safe — they checkpoint progress and resume from where they left off.

#### `ingest_pubmed.py` — PubMed Central Open Access

Downloads the full PubMed Central Open Access corpus via HTTPS bulk download.

**What it does:**
- Downloads 13 tar.gz archive files (8-12GB each) from NCBI's server
- Extracts and parses each XML file for title, abstract, authors, DOI, year, journal
- Writes to `papers_pubmed.jsonl`
- Deletes each archive after processing to save disk space
- Checkpoints after each archive — safe to kill and restart

**Run:**
```bash
nohup python ingest_pubmed.py > pubmed.log 2>&1 &
tail -f pubmed.log
```

**Expected output:** ~4.5M papers, takes 8-24 hours depending on network speed.

**Output file:** `papers_pubmed.jsonl`

---

#### `ingest_biorxiv.py` — bioRxiv and medRxiv

Downloads all preprints from bioRxiv (biology) and medRxiv (medicine) via their official REST API.

**What it does:**
- Fetches papers page by page (100 per page) from 2013 to today
- Handles pagination, rate limiting, and retries automatically
- Checkpoints every 100 pages

**Run:**
```bash
nohup python ingest_biorxiv.py > biorxiv.log 2>&1 &
tail -f biorxiv.log
```

**Expected output:** ~417K papers, takes 1-3 hours.

**Output file:** `papers_biorxiv.jsonl`

---

#### `ingest_doaj.py` — Directory of Open Access Journals

Downloads papers from DOAJ using OAI-PMH (Open Archives Initiative Protocol for Metadata Harvesting). Covers social sciences, humanities, law, economics, and everything arXiv doesn't.

**What it does:**
- Uses resumption tokens to paginate through DOAJ's full dataset
- Parses Dublin Core XML metadata
- Handles connection timeouts gracefully

**Run:**
```bash
nohup python ingest_doaj.py > doaj.log 2>&1 &
tail -f doaj.log
```

**Expected output:** ~277K papers, takes 3-6 hours.

**Output file:** `papers_doaj.jsonl`

---

#### `ingest_preprints.py` — OSF Preprints

Downloads preprints from OSF-hosted servers: PsyArXiv, SocArXiv, ChemRxiv, EarthArXiv, and others via the OSF REST API.

**Run:**
```bash
nohup python ingest_preprints.py > preprints.log 2>&1 &
tail -f preprints.log
```

**Expected output:** ~86K papers, takes 1-2 hours.

**Output file:** `papers_preprints.jsonl`

---

### Step 2: Embedding

After all ingestion scripts complete, run `embed_new_papers.py` to embed every new paper and extend the index.

#### `embed_new_papers.py` — Embed and Index

This is the most important script. It reads all four `.jsonl` files, deduplicates them against the existing corpus using four-layer deduplication (paper_id, DOI, arXiv ID, title fingerprint), runs each new paper through the fine-tuned SPECTER2 model to generate a 768-dimensional embedding, and extends the existing index.

**What it does:**
1. Streams through `papers_merged.jsonl` to build dedup sets (no full load into RAM)
2. Scans all four new paper files, removing duplicates
3. Loads SPECTER2 fine-tuned model onto GPU
4. Embeds papers in batches of 128
5. Saves checkpoint every 10,000 papers (safe to kill and restart)
6. Appends new embeddings to `embeddings.npy` using memory-mapped files
7. Updates `cadence.usearch` index with new vectors
8. Appends new paper metadata to `papers_merged.jsonl`

**RAM safety:** Never loads the full embedding matrix into RAM. Uses `numpy.memmap` to copy in 100K-row chunks. Peak RAM usage ~4GB regardless of corpus size.

**Run:**
```bash
# Stop the API first to free RAM for embedding
sudo systemctl stop cadence-api

nohup python embed_new_papers.py > embed_new.log 2>&1 &
tail -f embed_new.log

# After completion, restart the API
sudo systemctl start cadence-api
```

**Checkpoint recovery:** If the script crashes, simply re-run it. It reads `checkpoints/embed_checkpoint.json` and skips already-embedded papers.

**Expected time:** ~2 hours per 1M papers on an RTX 4060 Ti.

**Output:** Extended `embeddings.npy`, updated `cadence.usearch`, extended `papers_merged.jsonl`.

---

### Step 3: Nightly Ingestion

#### `nightly_ingest.py` — Daily Update

Runs every night via cron. Fetches papers published yesterday from arXiv and bioRxiv, embeds them, and adds them to the live index — all without restarting the API.

**What it does:**
1. Fetches yesterday's arXiv submissions via the arXiv Atom API
2. Fetches yesterday's bioRxiv/medRxiv papers via their REST API
3. Deduplicates against existing corpus (four-layer)
4. Embeds new papers on GPU
5. Appends to `embeddings.npy` using RAM-safe mmap
6. Updates `cadence.usearch` index
7. Appends metadata to `papers_merged.jsonl`
8. Calls `POST /admin/reload` to hot-reload the index in the live API (zero downtime)

**Set up cron:**
```bash
crontab -e
# Add:
0 3 * * * cd /home/your_path/cadence && /home/your_path/cadence/.venv/bin/python nightly_ingest.py >> nightly.log 2>&1
```

**Monitor:**
```bash
tail -f nightly.log
```

**Expected nightly additions:** 500-2000 new papers per night, taking 5-15 minutes.

---

### Step 4: Fine-tuning (optional)

#### `build_triplets.py` — Build Training Data

Generates training triplets (anchor, positive, negative) from the corpus using:
- Bibliographic coupling (papers citing the same references)
- Category co-occurrence (papers in the same arXiv category)
- User interaction signals (from Supabase interactions table)

**Run:**
```bash
python build_triplets.py
```

**Output:** `triplets.jsonl` — one triplet per line with `anchor`, `positive`, `negative`, `signal` fields.

---

#### `stage7_finetune.py` — Fine-tune SPECTER2

Fine-tunes the SPECTER2 base model on your triplets using triplet loss with hard negatives.

**Requirements:** NVIDIA GPU with 12GB+ VRAM, ~4 hours training time.

**Run:**
```bash
python stage7_finetune.py
```

**Output:** `specter2-finetuned/` — ready to use as a sentence-transformers model.

---

## API Reference

All endpoints require JWT authentication except `/auth/register` and `/auth/login`.

**Base URL:** `https://your-api-domain.com`

**Authentication:**
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" https://api.example.com/feed/discover
```

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register with email + password |
| POST | `/auth/login` | Login, returns 30-day JWT |

### Feed

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/feed/discover` | Personalized feed (taste vector search) |
| GET | `/feed/discover?sort=date` | Latest papers |
| GET | `/feed/discover?sort=popular` | Most cited papers |
| GET | `/feed/liked` | Papers you've liked |
| GET | `/feed/similar-to-saved` | Papers similar to your saved library |
| GET | `/feed/trending?category=cs.AI` | Trending by category |
| GET | `/feed/todays-pick` | One curated paper for today |
| GET | `/feed/hot?category=cs_ml` | Hot papers from HuggingFace + Semantic Scholar |
| POST | `/feed/interaction` | Log a paper interaction |

### Papers

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/papers/search?q=attention+transformers` | Semantic + keyword search |
| GET | `/papers/{id}` | Full paper metadata |
| GET | `/papers/{id}/similar` | Similar papers via usearch |
| GET | `/papers/{id}/unpaywall` | Find free PDF via Unpaywall |
| GET | `/papers/{id}/by-author` | Other papers by same author(s) |

### Library

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/library/saved` | All saved papers |
| POST | `/library/saved/{id}` | Save a paper |
| DELETE | `/library/saved/{id}` | Unsave a paper |
| GET | `/library/playlists` | All playlists |
| POST | `/library/playlists` | Create playlist |
| GET | `/library/playlists/{id}` | Get playlist with papers |
| PATCH | `/library/playlists/{id}` | Update playlist |
| DELETE | `/library/playlists/{id}` | Delete playlist |
| POST | `/library/playlists/{id}/papers/{paper_id}` | Add paper to playlist |
| DELETE | `/library/playlists/{id}/papers/{paper_id}` | Remove from playlist |

### User

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/user/profile` | Get user profile + preferences |
| PUT | `/user/profile` | Update profile |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/admin/reload` | Hot-reload index without restart |
| GET | `/health` | Server health + corpus size |

---

## Deployment

### systemd Service

```bash
sudo cp cadence-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cadence-api
sudo systemctl start cadence-api
sudo systemctl status cadence-api
```

`cadence-api.service`:
```ini
[Unit]
Description=Cadence FastAPI Backend
After=network.target

[Service] #change paths accordingly
User=rohan
WorkingDirectory=/home/rohan/cadence
Environment=PATH=/home/rohan/cadence/.venv/bin
ExecStart=/home/rohan/cadence/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Cloudflare Tunnel (expose without opening ports) ( you can deploy this on any machine running Linux, be it your local home PC, or some VM on a cloud service. To connect the backend and access it, you can make use of cloudflare tunnels which is a secure way to deploy such applications)

```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/

# Authenticate
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create cadence-api

# Configure (in ~/.cloudflared/config.yml):
# tunnel: YOUR_TUNNEL_ID
# credentials-file: /home/rohan/.cloudflared/YOUR_TUNNEL_ID.json
# ingress:
#   - hostname: cadence-api.yourdomain.com
#     service: http://localhost:8000
#   - service: http_status:404

# Run as service
sudo cloudflared service install
sudo systemctl start cloudflared
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB+ |
| GPU VRAM | 8GB (for embedding) | 16GB |
| Storage | 50GB | 200GB+ |
| CPU | 4 cores | 8 cores |

For 2.28M papers:
- `paper_meta` dict in RAM: ~4GB
- `cadence.usearch` index: ~3.6GB
- `embeddings.npy` (mmap): ~6.6GB (not all in RAM)
- SPECTER2 model: ~2GB
- Total RAM needed: ~12-14GB

---

## Configuration

All configuration via `.env` file:

```env
# Database (Supabase)
DB_URL=postgresql://postgres:PASSWORD@db.PROJECT.supabase.co:5432/postgres

# JWT
SECRET_KEY=your-secret-key-at-least-32-characters-long

# Optional: External APIs
GEMINI_API_KEY=your-gemini-key
```

File paths (hardcoded in `main.py`, change if needed):
```python
PAPERS_FILE    = "papers_merged.jsonl"
EMBEDDINGS     = "embeddings.npy"
PAPER_IDS      = "paper_ids.json"
EMBEDDINGS_INDEX = "cadence.usearch"
MODEL_PATH     = "./specter2-finetuned"
```

---

## Database Schema

```sql
-- Users
CREATE TABLE users (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email            TEXT UNIQUE NOT NULL,
    password_hash    TEXT NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    topics           TEXT[],
    difficulty       TEXT DEFAULT 'any',
    display_name     TEXT,
    role             TEXT,
    institution      TEXT,
    primary_field    TEXT,
    reading_goal     TEXT,
    experience_level TEXT,
    weekly_goal      INT DEFAULT 5
);

-- Interactions (drives recommendation engine)
CREATE TABLE interactions (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          UUID REFERENCES users(id) ON DELETE CASCADE,
    paper_id         TEXT NOT NULL,
    type             TEXT CHECK (type IN ('save','like','skip','read','share','download')),
    duration_seconds INT,
    swipe_velocity   FLOAT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Saved papers
CREATE TABLE saved_papers (
    user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
    paper_id   TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, paper_id)
);

-- Playlists
CREATE TABLE playlists (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE playlist_papers (
    playlist_id UUID REFERENCES playlists(id) ON DELETE CASCADE,
    paper_id    TEXT NOT NULL,
    added_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (playlist_id, paper_id)
);

-- Today's pick cache
CREATE TABLE todays_pick_cache (
    user_id  UUID REFERENCES users(id) ON DELETE CASCADE,
    date     DATE NOT NULL,
    paper_id TEXT NOT NULL,
    PRIMARY KEY (user_id, date)
);
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas where help is most needed:**
- Collaborative filtering implementation
- Audio summary feature
- Larger embedding model training
- Citation graph integration
- More corpus sources

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Citation

If you use Cadence or the cadence-specter2 model in your research:

```bibtex
@software{cadence2026,
  author  = {Marar, Rohan},
  title   = {Cadence: AI-Powered Research Paper Discovery},
  year    = {2026},
  url     = {https://github.com/Rohan5manza/cadence-backend}
}
```
