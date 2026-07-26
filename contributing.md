# Contributing to Cadence

Thank you for your interest in contributing to Cadence. This document explains how to get started.

---

## Areas Where Help Is Most Needed

### High priority
- **Collaborative filtering** — user similarity matrix + SQL-based recommendations. Needs ~1000 users before it's meaningful, but the infrastructure can be built now. See `main.py` `_taste_vector()` for the existing content-based approach.
- **Audio summaries** — podcast-style paper summaries using Claude API + ElevenLabs TTS. The architecture is planned: paper abstract → Claude summarizes to 500-word script → TTS → cached in Supabase Storage.
- **Citation graph** — integrate S2ORC citation links into a PostgreSQL `references` table. Add `/papers/{id}/references` and `/papers/{id}/cited-by` endpoints.

### Medium priority
- **Larger embedding model** — train a bigger SPECTER2 variant on the full 2.28M corpus with harder negatives. See `stage7_finetune.py` for the existing fine-tuning setup.
- **More corpus sources** — Semantic Scholar Open Research Corpus, CrossRef, Europe PMC. Add a new `ingest_*.py` script following the pattern of existing ones.
- **iPad layout** — the PWA and native app both work on iPad but aren't optimized for it.
- **Offline mode** — cache the last 20 papers in AsyncStorage/localStorage for reading without internet.

### Low priority / good first issues
- Fix the `by-author` endpoint — authors field is currently empty in the corpus; needs to be populated during ingestion
- Add paper summary endpoint using Claude API
- Add dark/light mode toggle to the web PWA
- Improve search ranking (currently basic BM25 + semantic hybrid)
- Add more language support for paper metadata

---

## Development Setup

### Backend

```bash
git clone https://github.com/Rohan5manza/cadence-backend.git
cd cadence-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Supabase credentials
```

For development without the full corpus, you can start with a small subset:
```bash
# Download 10K papers for testing
python3 -c "
import json
with open('papers_merged.jsonl') as f:
    papers = [json.loads(l) for l, _ in zip(f, range(10000))]
with open('papers_dev.jsonl', 'w') as f:
    for p in papers: f.write(json.dumps(p) + '\n')
"
# Then set PAPERS_FILE = 'papers_dev.jsonl' in main.py for dev
```

### Frontend

```bash
git clone https://github.com/Rohan5manza/cadence-app.git
cd cadence-app
npm install
# Update API_BASE in services/api.ts to point to your local backend
npx expo start
```

---

## Pull Request Guidelines

1. **Fork** the repo and create a branch: `git checkout -b feature/your-feature-name`
2. **One feature per PR** — keep changes focused
3. **Test your changes** — for backend, test the affected endpoints manually
4. **Document new scripts** — if you add an ingestion script, follow the docstring format in existing ones
5. **No large files** — never commit `.npy`, `.usearch`, `.jsonl`, or model weights

### Commit message format
```
feat: add collaborative filtering endpoint
fix: trending endpoint timeout on large corpus
docs: add citation graph setup guide
refactor: extract dedup helpers to utils.py
```

---

## Code Style

### Python (backend)
- Follow existing style in `main.py`
- Type hints where practical
- Docstrings for all functions over 20 lines
- `# ── Section name ─────────` style separators for long files

### TypeScript (frontend)
- Follow existing patterns in `home.tsx` and `useStore.ts`
- Platform-specific code via `Platform.OS === 'web'` checks
- Web-safe storage via the `webStorage` wrapper in `home.tsx` or `storage` in `useStore.ts`
- Never import `react-native-webview` at the top level — use conditional require

---

## Architecture Decisions

**Why in-memory paper_meta instead of database?**  
Sub-300ms response times require everything in RAM. A DB query for 2.28M papers would be far too slow for real-time recommendation. The tradeoff is 12-14GB RAM usage on the server.

**Why usearch instead of Faiss or Pinecone?**  
usearch is faster than Faiss for HNSW, has a clean Python API, and doesn't require a separate service. Pinecone would work but adds $70+/month for this corpus size.

**Why not PostgreSQL pgvector?**  
pgvector is excellent but slower than in-memory usearch for our query pattern (single user taste vector → top 20 nearest neighbors in 2.28M vectors). At 2.28M vectors, usearch returns results in <5ms; pgvector would be ~50-200ms.

**Why sentence-transformers instead of raw transformers?**  
Sentence-transformers handles the pooling and normalization correctly for SPECTER2 and makes batch encoding straightforward. The `encode()` method with `normalize_embeddings=True` produces unit-norm vectors ready for cosine similarity via usearch.

**Why Expo + React Native instead of pure React Native or Flutter?**  
Expo's web export (`npx expo export --platform web`) gives a free PWA build from the same codebase. This lets us ship on web (no app store) before dealing with Apple review.

---

## Questions

Open a GitHub issue or email [cadence@rohanmarar.com](mailto:cadence@rohanmarar.com).
