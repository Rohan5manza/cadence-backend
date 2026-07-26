# Cadence — Architecture Deep Dive

This document explains the internal design decisions behind Cadence's recommendation engine, data pipeline, and API design.

---

## The Recommendation Engine

### Embedding Space

Every research paper is converted to a 768-dimensional vector by the fine-tuned SPECTER2 model. This vector encodes the semantic meaning of the paper's title and abstract.

Papers about similar topics end up close together in this 768-dimensional space. Papers about unrelated topics end up far apart. The distance metric is cosine similarity (inner product of unit-norm vectors).

```
Input:  "Attention Is All You Need [SEP] The dominant sequence transduction..."
Output: [0.021, -0.134, 0.089, 0.045, ..., -0.012]  ← 768 floats
```

SPECTER2 was specifically pre-trained on scientific text using citation relationships as a training signal — papers that cite each other are pulled closer in the embedding space. This gives it strong prior knowledge about scientific similarity that general-purpose models like BERT lack.

### Taste Vector

Each user has a taste vector — a single 768-dimensional point that represents their current reading preferences. It is computed on every feed request from the user's interaction history.

```python
def _taste_vector(uid, conn):
    rows = await conn.fetch(
        """SELECT i.paper_id, i.type, i.duration_seconds, i.created_at
           FROM interactions i
           WHERE i.user_id = $1
           ORDER BY i.created_at DESC
           LIMIT 50""",
        uid
    )
    
    vectors = []
    weights = []
    
    for row in rows:
        paper_emb = get_embedding(row['paper_id'])  # from embeddings.npy
        
        # Interaction type weight
        weight = {
            'save':  3.0,
            'share': 2.0,
            'like':  1.5,
            'read':  1.0,
            'skip': -1.5,
        }.get(row['type'], 0.0)
        
        # Boost long reads
        if row['type'] == 'read' and row['duration_seconds'] > 120:
            weight = 2.0
        
        # Recency decay: 0.95^days_since
        days = (now - row['created_at']).days
        decay = 0.95 ** days
        
        vectors.append(paper_emb)
        weights.append(weight * decay)
    
    # Weighted average, normalized to unit norm
    taste = np.average(vectors, weights=weights, axis=0)
    taste = taste / np.linalg.norm(taste)
    return taste
```

This taste vector moves continuously as the user interacts with papers. A single skip doesn't overwhelm many likes. Recent interactions matter more than old ones.

### Vector Search

usearch implements HNSW (Hierarchical Navigable Small World), a graph-based ANN algorithm. The index is built once, then queried on every feed request.

```python
# Query: find 100 papers nearest to user's taste vector
matches = paper_index.search(taste_vector, 100)
# Returns: [(key, distance), ...] sorted by distance ascending

# Map usearch keys back to paper IDs
paper_ids = [paper_id_list[int(m.key)] for m in matches]
```

HNSW works by building a multi-layer graph where each paper is connected to its nearest neighbors at different scales. Queries navigate this graph by following edges toward the query vector, achieving O(log n) search time instead of O(n) brute force.

At 2.28M papers, each query takes <5ms.

### Profile-Based Filtering

After vector search returns 100 candidates, they are filtered and re-ranked based on the user's profile:

```python
# 1. Primary field gets 3× weight in category matching
if primary_field:
    primary_cats = TOPIC_CATEGORY_MAP[primary_field]  # e.g. ['cs.LG', 'cs.AI']

# 2. Reading goal affects sort
if reading_goal == 'stay_current':
    papers.sort(key=lambda p: p['year'], reverse=True)
    # Also filter out pre-2022 papers (70% probability)
elif reading_goal == 'deep_dive':
    papers.sort(key=lambda p: p['citation_count'] or 0, reverse=True)

# 3. Experience level affects difficulty filter
if experience_level == 'beginner':
    # Filter out heavy math/physics categories
    papers = [p for p in papers if not any(
        cat in ['math.AP', 'math.PR', 'quant-ph', 'hep-th']
        for cat in p['categories']
    )]

# 4. 10% diversity injection
n_diverse = max(1, len(papers) // 10)
diverse = random.sample(paper_meta_keys, n_diverse)
papers[-n_diverse:] = diverse
```

### Cold Start

New users have no interactions, so there's no taste vector. Cold start is handled using the onboarding profile:

1. User selects topics, primary field, role, reading goal during onboarding
2. Discover endpoint falls back to category-based sampling if no taste vector
3. First interactions immediately start building the taste vector

After ~5-10 interactions, recommendations become meaningfully personalized.

---

## Data Pipeline

### Corpus Structure

All paper data is stored in three files that must stay in sync:

```
papers_merged.jsonl     ← paper metadata (title, abstract, authors, etc.)
embeddings.npy          ← paper embeddings, one row per paper
paper_ids.json          ← ordered list of paper IDs

Rule: papers_merged.jsonl[i] corresponds to embeddings.npy[i] corresponds to paper_ids.json[i]
      BUT the order in papers_merged.jsonl does NOT match the array indices.
      paper_ids.json is the canonical mapping: paper_ids.json[i] → embeddings.npy[i]
```

**Loading at startup:**
```python
# Load all paper metadata into RAM
paper_meta: dict[str, dict] = {}
with open('papers_merged.jsonl') as f:
    for line in f:
        p = json.loads(line)
        paper_meta[p['paper_id']] = p  # keyed by paper_id

# Load ordered ID list (maps array index → paper_id)
with open('paper_ids.json') as f:
    paper_id_list = json.load(f)

# Load usearch index
paper_index = Index.restore('cadence.usearch')
# usearch keys are array indices (0, 1, 2, ...) → paper_id_list[key] → paper_meta[paper_id]
```

**Lookup chain:**
```
usearch.search(taste) → [key=42, key=1337, ...]
                           ↓
paper_id_list[42] → "arxiv_2301.12345"
                           ↓
paper_meta["arxiv_2301.12345"] → {title, abstract, authors, ...}
```

### Deduplication

Papers are deduplicated across sources using four methods simultaneously:

```python
# 1. Exact paper_id match (fastest)
if pid in existing_ids: duplicate

# 2. DOI match (catches same paper from different sources)
doi = normalize_doi(p['doi'])  # lowercase, strip URL prefix
if doi in existing_dois: duplicate

# 3. arXiv ID match (catches version differences: 2301.12345v1 = 2301.12345v2)
arxiv_id = re.sub(r'v\d+$', '', p['arxiv_id'])
if arxiv_id in existing_arxivs: duplicate

# 4. Title fingerprint (catches same paper with slightly different metadata)
fp = md5(re.sub(r'[^a-z0-9]', '', title.lower())[:80])
if fp in existing_fps: duplicate
```

This four-layer approach is necessary because the same paper often appears in multiple sources with different IDs, DOIs with/without URL prefixes, and minor title variations.

### RAM-Safe Embedding Extension

Adding new embeddings to `embeddings.npy` without loading the full 6.6GB file into RAM:

```python
# 1. Get existing shape without loading data
mmap = np.load('embeddings.npy', mmap_mode='r')
old_n, dim = mmap.shape  # e.g. (2288250, 768)
del mmap  # release immediately

# 2. Create new file at full final size
new_n = old_n + len(new_vecs)
dest = np.lib.format.open_memmap(
    'embeddings_new.npy', mode='w+', dtype=np.float32, shape=(new_n, dim)
)

# 3. Copy existing data in 100K-row chunks (~300MB each)
src = np.load('embeddings.npy', mmap_mode='r')
for start in range(0, old_n, 100_000):
    end = min(start + 100_000, old_n)
    dest[start:end] = src[start:end]
del src

# 4. Append new vectors
dest[old_n:] = new_vecs
del dest  # flush to disk

# 5. Atomic replace
os.rename('embeddings.npy', 'embeddings_backup.npy')
os.rename('embeddings_new.npy', 'embeddings.npy')
```

Peak RAM: ~300MB per chunk + new vectors only. Safe on a 16GB machine.

---

## API Design

### Authentication

JWT tokens with 30-day expiry. Tokens are verified on every request via a FastAPI dependency:

```python
async def current_user(token: str = Depends(oauth2_scheme)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    return payload['sub']  # user UUID
```

No refresh tokens — users just log in again after 30 days.

### Database Connection Pooling

asyncpg connection pool with min 2, max 10 connections:

```python
db_pool = await asyncpg.create_pool(
    dsn=DB_URL, ssl='require', min_size=2, max_size=10
)
```

Each endpoint gets a connection from the pool via `Depends(get_db)`. Connections are returned to the pool after the response.

### Hot Reload

The `/admin/reload` endpoint updates the live index without restarting the server:

```python
@app.post("/admin/reload")
async def hot_reload(secret: str = Query(...)):
    global paper_meta, paper_id_list, paper_index
    
    # Add only NEW papers (don't reload existing ones)
    new_count = 0
    with open(PAPERS_FILE) as f:
        for line in f:
            p = json.loads(line)
            pid = p['paper_id']
            if pid not in paper_meta:
                paper_meta[pid] = {minimal metadata}
                new_count += 1
    
    # Reload IDs and index
    with open(PAPER_IDS) as f:
        paper_id_list = json.load(f)
    
    # Free old index memory before loading new one
    paper_index = None
    gc.collect()
    paper_index = Index.restore(EMBEDDINGS_INDEX)
    
    return {"new_papers": new_count, "total": len(paper_meta)}
```

This is called by `nightly_ingest.py` after adding new papers, ensuring the live API serves them immediately without downtime.

### Transient Papers

External papers (from HuggingFace daily papers, Semantic Scholar) are served from the `/feed/hot` endpoint but don't exist in the corpus. They're cached in a `transient_papers` dict:

```python
transient_papers: dict[str, dict] = {}

# In /feed/hot:
paper_id = f"hf_{arxiv_id}"
transient_papers[paper_id] = {full paper dict}

# In /papers/{paper_id}:
if paper_id in transient_papers:
    return transient_papers[paper_id]
# Otherwise look up in paper_meta
```

This allows users to tap into a hot paper and see its full detail page even though it's not in the permanent index.

---

## Frontend Architecture

### State Management

Zustand store with three layers:

1. **In-memory** — React state within components (card stack, loading states)
2. **Persistent** — localStorage/SecureStore (token, preferences, history, streak)
3. **Server** — Supabase (interactions, saved papers, playlists)

The store uses a platform-adaptive storage wrapper so the same code works on web (localStorage) and native (expo-secure-store).

### Web Safety

Three categories of native-only code require special handling on web:

**1. expo-notifications** — wrapped in try/catch, every function guards with `if (!Notifications) return`

**2. react-native-webview** — conditionally required:
```typescript
let WebView = null
if (Platform.OS !== 'web') {
  WebView = require('react-native-webview').WebView
}
// In render:
Platform.OS === 'web'
  ? <iframe src={url} />
  : <WebView source={{uri: url}} />
```

**3. expo-secure-store** — replaced with localStorage on web via the `storage` wrapper in `useStore.ts`

### Routing

Expo Router provides file-based routing. The route tree:
```
/               → index.tsx (auth check)
/auth           → auth.tsx
/onboarding     → onboarding.tsx
/(tabs)/home    → home.tsx
/(tabs)/feed    → feed.tsx
/(tabs)/search  → search.tsx
/(tabs)/library → library.tsx
/(tabs)/profile → profile.tsx
/paper/[id]     → [id].tsx
/paper/read     → read.tsx (requires ?id= param)
/paper/similar  → similar.tsx (requires ?id= param)
/playlist/[id]  → [id].tsx
```

### PWA Architecture

The web build is a single-page application with:
- **Service worker** (`sw.js`) — minimal fetch passthrough, enables PWA install
- **Web manifest** (`manifest.json`) — app name, icons, display mode
- **_redirects** — Netlify SPA fallback: `/* /index.html 200`
- **Cookies** — auth token and preferences stored in cookies (not just localStorage) for persistence across PWA and browser contexts

The `post-build.sh` script auto-generates the manifest with the correct hashed icon path after each `expo export`.

---

## Scaling Considerations

### Current bottlenecks

1. **RAM** — 2.28M papers × ~1.7KB metadata = ~4GB. Growing the corpus requires more RAM.
2. **Single process** — `--workers 1` because paper_meta is a global in-process dict. Multiple workers would each load their own 4GB copy.
3. **API restart time** — ~2 minutes to reload 2.28M papers from disk.

### Path to scale

**Short term (2-10M papers):**
- Move to Hetzner AX42 (64GB RAM, €39/month) — fits 10M papers with full abstracts
- Reduce `paper_meta` to store only essential fields (drop abstracts, load on-demand)

**Medium term (10M+ papers):**
- Split paper_meta into a fast key-value store (Redis or DuckDB in-memory)
- Use multiple uvicorn workers with shared memory for embeddings
- Pre-compute and cache daily recommendations in Supabase

**Long term (100M+ papers):**
- Dedicated vector database (Weaviate, Qdrant) instead of in-process usearch
- Distributed embedding service
- Collaborative filtering layer on top of content-based
