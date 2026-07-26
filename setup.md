# Cadence — Complete Setup Guide

This guide walks through setting up a complete Cadence instance from scratch on a fresh Ubuntu server.

---

## Prerequisites

- Ubuntu 22.04+ server
- 16GB+ RAM (32GB recommended)
- NVIDIA GPU with 8GB+ VRAM (for embedding — not required for API-only)
- 100GB+ disk space
- A domain name (optional but recommended)
- A Supabase account (free tier works)

---

## Part 1: Server Setup

### 1.1 Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y

# Python
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Build tools
sudo apt install -y build-essential git curl wget

# NVIDIA drivers (if using GPU)
sudo apt install -y nvidia-driver-545
nvidia-smi  # verify GPU is detected
```

### 1.2 Clone the backend

```bash
git clone https://github.com/Rohan5manza/cadence-backend.git
cd cadence-backend

python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 Create environment file

```bash
cp .env.example .env
nano .env
```

Fill in:
```env
DB_URL=postgresql://postgres:YOUR_PASSWORD@db.YOUR_PROJECT.supabase.co:5432/postgres
SECRET_KEY=generate-a-random-32-char-string-here
```

Generate a secret key:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## Part 2: Supabase Setup

### 2.1 Create project

1. Go to [supabase.com](https://supabase.com) → New Project
2. Save your database password
3. Copy the connection string from Project Settings → Database → URI

### 2.2 Create schema

In Supabase SQL Editor, paste and run the contents of `schema.sql`.

### 2.3 Enable Row Level Security (optional but recommended)

```sql
ALTER TABLE interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_papers  ENABLE ROW LEVEL SECURITY;
ALTER TABLE playlists     ENABLE ROW LEVEL SECURITY;
-- Add policies as needed
```

---

## Part 3: Build the Corpus

This is the most time-consuming part. Skip to Part 4 if you're using a pre-built corpus.

### 3.1 Run ingestion scripts (in parallel)

Open 4 terminal windows or tmux panes:

```bash
# Terminal 1 — PubMed Central (~8-24 hours)
nohup python ingest_pubmed.py > pubmed.log 2>&1 &
tail -f pubmed.log

# Terminal 2 — bioRxiv/medRxiv (~1-3 hours)
nohup python ingest_biorxiv.py > biorxiv.log 2>&1 &
tail -f biorxiv.log

# Terminal 3 — DOAJ (~3-6 hours)
nohup python ingest_doaj.py > doaj.log 2>&1 &
tail -f doaj.log

# Terminal 4 — OSF Preprints (~1-2 hours)
nohup python ingest_preprints.py > preprints.log 2>&1 &
tail -f preprints.log
```

Monitor progress:
```bash
wc -l papers_pubmed.jsonl papers_biorxiv.jsonl papers_doaj.jsonl papers_preprints.jsonl
```

### 3.2 Download the SPECTER2 fine-tuned model

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='rohan5manza/cadence-specter2',
    local_dir='./specter2-finetuned',
)
print('Model downloaded')
"
```

Or use the base SPECTER2 model (less accurate but works):
```bash
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('allenai/specter2_base')
model.save('./specter2-finetuned')
print('Base model saved')
"
```

### 3.3 Embed all papers

After all ingestion scripts complete:

```bash
# This takes ~2 hours per 1M papers on an RTX 4060 Ti
nohup python embed_new_papers.py > embed_new.log 2>&1 &
tail -f embed_new.log
```

If it crashes (common due to RAM), just re-run it — checkpoints every 10K papers.

Verify the final output:
```bash
python3 -c "
import numpy as np, json
emb = np.load('embeddings.npy', mmap_mode='r')
ids = json.load(open('paper_ids.json'))
print(f'Embeddings: {emb.shape}')
print(f'Paper IDs:  {len(ids):,}')
print(f'Match: {emb.shape[0] == len(ids)}')
"
```

---

## Part 4: Start the API

### 4.1 Test start

```bash
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Wait 1-3 minutes for startup (loading papers into RAM). Then:
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok","papers_loaded":XXXXXX,"index_ready":true}
```

### 4.2 Set up as systemd service

```bash
sudo cp cadence-api.service /etc/systemd/system/
# Edit the service file to match your paths
sudo nano /etc/systemd/system/cadence-api.service

sudo systemctl daemon-reload
sudo systemctl enable cadence-api
sudo systemctl start cadence-api
sudo systemctl status cadence-api
```

### 4.3 View logs

```bash
journalctl -u cadence-api -f
```

---

## Part 5: Expose to Internet

### Option A: Cloudflare Tunnel (recommended — no open ports needed)

```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Authenticate
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create cadence-api

# Create config
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml << EOF
tunnel: YOUR_TUNNEL_ID
credentials-file: /home/YOUR_USER/.cloudflared/YOUR_TUNNEL_ID.json
ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# Add DNS record in Cloudflare dashboard for api.yourdomain.com

# Run as service
sudo cloudflared service install
sudo systemctl start cloudflared
```

### Option B: Nginx reverse proxy

```bash
sudo apt install nginx -y

sudo cat > /etc/nginx/sites-available/cadence << EOF
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/cadence /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Add SSL with certbot
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d api.yourdomain.com
```

---

## Part 6: Set Up Nightly Ingestion

```bash
crontab -e
# Add this line (runs at 3 AM):
0 3 * * * cd /home/YOUR_USER/cadence-backend && /home/YOUR_USER/cadence-backend/.venv/bin/python nightly_ingest.py >> nightly.log 2>&1
```

Verify it works:
```bash
python nightly_ingest.py
# Should print: fetching arXiv... N new papers... Done
```

---

## Part 7: Deploy the Frontend

### 7.1 Clone and configure

```bash
# On your Mac/dev machine
git clone https://github.com/Rohan5manza/cadence-app.git
cd cadence-app
npm install
```

Update API URL in `services/api.ts`:
```typescript
const API_BASE = 'https://api.yourdomain.com'
```

### 7.2 Build PWA

```bash
npx expo export --platform web
./scripts/post-build.sh
```

### 7.3 Deploy to Netlify

Option A — Netlify CLI:
```bash
npm install -g netlify-cli
netlify deploy --dir dist --prod
```

Option B — Drag and drop:
Go to netlify.com → drag `dist/` folder into deploy box.

### 7.4 Custom domain (optional)

In Cloudflare DNS, add a CNAME record:
- Name: `cadence` (or your subdomain)
- Target: `your-site.netlify.app`
- Proxy: DNS only (grey cloud)

In Netlify: Domain settings → Add custom domain → verify → provision SSL.

---

## Verification Checklist

```bash
# Backend health
curl https://api.yourdomain.com/health

# Register a test user
curl -X POST https://api.yourdomain.com/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpassword123"}'

# Login
curl -X POST https://api.yourdomain.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpassword123"}'

# Get discover feed (use token from login response)
curl https://api.yourdomain.com/feed/discover \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## Troubleshooting

### API crashes on startup
```bash
# Check logs
journalctl -u cadence-api -n 50

# Common causes:
# 1. DB_URL wrong → "asyncpg: failed to connect"
# 2. Not enough RAM → OOM killed
# 3. Missing files → "FileNotFoundError: papers_merged.jsonl"
```

### Out of memory during embedding
```bash
# Free up RAM before embedding
sudo systemctl stop cadence-api
sudo swapoff -a && sudo swapon -a  # clear swap

# Use smaller batch size
# Edit embed_new_papers.py: BATCH_SIZE = 32
```

### Embedding script crashes and restarts from 0
```bash
# Check checkpoint exists
cat checkpoints/embed_checkpoint.json | python3 -m json.tool

# If checkpoint exists but script starts over,
# check the checkpoint file isn't corrupt:
python3 -c "import json; json.load(open('checkpoints/embed_checkpoint.json'))"
```

### usearch index out of sync with embeddings
```bash
python3 -c "
import numpy as np, json
from usearch.index import Index

emb = np.load('embeddings.npy', mmap_mode='r')
ids = json.load(open('paper_ids.json'))
idx = Index.restore('cadence.usearch')

print(f'Embeddings: {emb.shape[0]:,}')
print(f'Paper IDs:  {len(ids):,}')
print(f'Index size: {len(idx):,}')
# All three should match
"
```

If out of sync, rebuild the index:
```bash
python3 -c "
import numpy as np, json
from usearch.index import Index

emb = np.load('embeddings.npy', mmap_mode='r')
print(f'Rebuilding index for {emb.shape[0]:,} vectors...')
idx = Index(ndim=768, metric='cos')
import numpy as np2
keys = np2.arange(emb.shape[0], dtype=np.int64)
batch = 50000
for s in range(0, emb.shape[0], batch):
    e = min(s+batch, emb.shape[0])
    idx.add(keys[s:e], emb[s:e])
    print(f'  {e:,}/{emb.shape[0]:,}')
idx.save('cadence.usearch')
print('Done')
"
```
