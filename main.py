"""
Cadence FastAPI Backend — clean rewrite
"""
import os
import json
import random
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager
import httpx

import asyncpg
import bcrypt
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from pydantic import BaseModel

import httpx
 

# ── Config ─────────────────────────────────────────────────────────────────────
DB_URL       = "postgresql://YOUR_CUSTOM_PATH@localhost:5432/cadence"
SECRET_KEY   = "replace_with_long_random_string_in_production"
ALGORITHM    = "HS256"
TOKEN_EXPIRE = 30  # days

PAPERS_FILE  = "papers_merged.jsonl"
EMBEDDINGS   = "embeddings.npy"
PAPER_IDS    = "paper_ids.json"
# ──────────────────────────────────────────────────────────────────────────────

# Replace your existing TOPIC_CATEGORY_MAP in main.py with this:

TOPIC_CATEGORY_MAP = {
    # Computer Science
    'cs_ai':         ['cs.ai', 'cs.lg', 'cs.ne'],
    'cs_ml':         ['cs.lg', 'stat.ml'],
    'cs_nlp':        ['cs.cl', 'cs.ir'],
    'cs_cv':         ['cs.cv'],
    'cs_ro':         ['cs.ro', 'cs.sy'],
    'cs_cr':         ['cs.cr'],
    'cs_hci':        ['cs.hc'],
    'cs_ds':         ['cs.ds', 'cs.dm'],

    # Biology
    'bio_genomics':  ['q-bio.gn', 'q-bio.qm', 'genomics', 'genetics'],
    'bio_neuro':     ['q-bio.nc', 'psych', 'neuro'],
    'bio_cell':      ['q-bio.cb', 'q-bio.bm', 'biochem', 'biology'],
    'bio_ecology':   ['q-bio.pe', 'q-bio.to', 'ecology'],

    # Medicine
    'med_clinical':  ['med', 'health', 'clinical', 'pubmed'],
    'med_imaging':   ['eess.iv', 'cs.cv', 'medical imaging'],
    'med_pharma':    ['q-bio.qm', 'pharmacol', 'drug'],

    # Physics
    'phys_quantum':   ['quant-ph'],
    'phys_condensed': ['cond-mat'],
    'phys_astro':     ['astro-ph'],
    'phys_hep':       ['hep-ph', 'hep-th', 'hep-ex'],

    # Mathematics
    'math_pure':     ['math.ag', 'math.nt', 'math.at', 'math.gr', 'math'],
    'math_stats':    ['stat.th', 'stat.me', 'math.pr', 'stat'],
    'math_applied':  ['math.na', 'math.oc', 'math.ap'],

    # Economics
    'econ_theory':   ['econ.th', 'econ.gn', 'econ'],
    'econ_finance':  ['q-fin', 'finance'],

    # Psychology
    'psych_cog':     ['psych', 'cog'],

    # Environment
    'env_climate':   ['atmos', 'environ', 'climate'],

    # Humanities
    'phil_ethics':   ['phil', 'ethics'],
    'hist_social':   ['history', 'humanities', 'social'],

    # Legacy keys (keep for backwards compatibility)
    'cs_ai_legacy':  ['cs.ai', 'cs.lg', 'cs.cl', 'cs.cv', 'cs.ne'],
    'biology':       ['q-bio', 'biology', 'biochem'],
    'medicine':      ['med', 'health', 'clinical', 'pubmed'],
    'physics':       ['physics', 'cond-mat', 'quant-ph', 'hep'],
    'mathematics':   ['math'],
    'economics':     ['econ', 'q-fin'],
    'psychology':    ['psych', 'neuro'],
    'climate':       ['atmos', 'environ', 'climate'],
    'history':       ['history', 'humanities'],
    'philosophy':    ['phil', 'ethics'],
}

# ── Global state ───────────────────────────────────────────────────────────────
db_pool:        asyncpg.Pool  = None
paper_index                   = None   # usearch Index
embeddings_arr: np.ndarray    = None
paper_meta:     dict          = {}     # paper_id -> dict
paper_id_list:  list          = []
index_ready:    bool          = False
transient_papers = {}
# ──────────────────────────────────────────────────────────────────────────────

query_model = None

class ProfileIn(BaseModel):
    topics:           list[str]
    difficulty:       str
    display_name:     str        = ""
    role:             str        = "curious"
    institution:      str        = ""
    primary_field:    str        = ""
    reading_goal:     str        = "broad"
    experience_level: str        = "intermediate"
    weekly_goal:      int        = 5
 

# Load model in background thread (add to lifespan)
def _load_query_model():
    global query_model
    try:
        from sentence_transformers import SentenceTransformer
        print("[model] Loading query model...")
        query_model = SentenceTransformer("./specter2-finetuned")
        print("[model] Query model ready ✓")
    except Exception as e:
        print(f"[model] WARNING: {e}")

# In lifespan, after starting index thread:
threading.Thread(target=_load_query_model, daemon=True).start()

def safe_array(val) -> list:
    """Safely convert any value to a list of strings."""
    if not val:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v]
    if isinstance(val, str):
        return [v for v in val.replace(',', ' ').split() if v]
    return []

def _build_index_thread():
    global paper_index, embeddings_arr, index_ready
    try:
        from usearch.index import Index
        if os.path.exists("cadence.usearch"):
            print("[index] Loading pre-built index...")
            paper_index    = Index.restore("cadence.usearch")
            embeddings_arr = np.load(EMBEDDINGS, mmap_mode="r")
            index_ready    = True
            print("[index] Index loaded ✓")
        else:
            print("[index] No pre-built index found — building from scratch...")
            emb   = np.load(EMBEDDINGS, mmap_mode="r")
            idx   = Index(ndim=768, metric="cos")
            chunk = 100_000
            total = len(paper_id_list)
            for start in range(0, total, chunk):
                end = min(start + chunk, total)
                idx.add(
                    np.arange(start, end, dtype=np.int64),
                    emb[start:end].astype(np.float32),
                )
            paper_index    = idx
            embeddings_arr = emb
            index_ready    = True
            print("[index] Index built ✓")
    except Exception as exc:
        print(f"[index] WARNING: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, paper_meta, paper_id_list

    # 1. Database
    print("[startup] Connecting to database...")
    db_pool = await asyncpg.create_pool(dsn=DB_URL, min_size=2, max_size=10)
    print("[startup] Database connected ✓")

    # 2. Paper metadata
    print(f"[startup] Loading paper metadata from {PAPERS_FILE}...")
    if os.path.exists(PAPERS_FILE):
        count = 0
        with open(PAPERS_FILE) as f:
            for line in f:
                try:
                    p   = json.loads(line)
                    pid = str(p["paper_id"])
                    paper_meta[pid] = {
                        "title":           p.get("title") or "",
                        "abstract":        p.get("abstract") or "",
                        "authors":         p.get("authors") or [],
                        "year":            p.get("year"),
                        "venue":           p.get("venue") or "",
                        "doi":             p.get("doi"),
                        "arxiv_id":        p.get("arxiv_id"),
                        "categories":      p.get("categories") or [],
                        "source":          p.get("source") or "",
                        "citation_count":  p.get("citation_count"),
                        "open_access_url": p.get("open_access_url"),
                    }
                    count += 1
                except Exception:
                    pass
        print(f"[startup] Loaded {count:,} papers ✓")

    # 3. Paper ID list
    if os.path.exists(PAPER_IDS):
        with open(PAPER_IDS) as f:
            paper_id_list = json.load(f)
        print(f"[startup] Loaded {len(paper_id_list):,} paper IDs ✓")

    # 4. Start index build in background
    if os.path.exists(EMBEDDINGS) and paper_id_list:
        t = threading.Thread(target=_build_index_thread, daemon=True)
        t.start()
        print("[startup] Index building in background — API ready immediately ✓")

    yield  # ← app runs here

    # Shutdown
    print("[shutdown] Closing database pool...")
    await db_pool.close()

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Cadence API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


# ── Auth helpers ───────────────────────────────────────────────────────────────
def _hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def _verify_pw(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def _make_token(user_id: str) -> str:
    exp = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE)
    return jwt.encode({"sub": user_id, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def _decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token")
        return uid
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> str:
    return _decode_token(creds.credentials)


# ── DB dependency ──────────────────────────────────────────────────────────────
class DBConn:
    """Dependency that yields a single connection from the pool."""
    async def __call__(self):
        async with db_pool.acquire() as conn:
            yield conn

get_db = DBConn()


# ── Pydantic models ────────────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    email:    str
    password: str

class InteractionIn(BaseModel):
    paper_id:         str
    type:             str   # save | skip | read | share
    duration_seconds: Optional[int]   = None
    swipe_velocity:   Optional[float] = None

class PlaylistIn(BaseModel):
    name:        str
    description: Optional[str] = None

class PreferencesIn(BaseModel):
    topics: list[str]
    difficulty: str

# ── Helpers ────────────────────────────────────────────────────────────────────
def _paper_dict(paper_id: str) -> dict:
    p = paper_meta.get(str(paper_id), {})
    return {
        "id":              str(paper_id),
        "title":           p.get("title") or "",
        "abstract":        p.get("abstract") or "",
        "authors":         p.get("authors") or [],
        "year":            p.get("year"),
        "venue":           p.get("venue") or "",
        "doi":             p.get("doi"),
        "arxiv_id":        p.get("arxiv_id"),
        "categories":      p.get("categories") or [],
        "source":          p.get("source") or "",
        "citation_count":  p.get("citation_count"),
        "open_access_url": p.get("open_access_url"),
    }

async def _taste_vector(user_id: str, conn: asyncpg.Connection) -> Optional[np.ndarray]:
    """Compute weighted average embedding for user taste."""
    if not index_ready or embeddings_arr is None:
        return None

    rows = await conn.fetch(
        "SELECT paper_id, type, duration_seconds, created_at "
        "FROM interactions WHERE user_id = $1 "
        "ORDER BY created_at DESC LIMIT 50",
        user_id,
    )
    if not rows:
        return None

    pid_to_idx = {pid: i for i, pid in enumerate(paper_id_list)}
    vecs, wts  = [], []

    for r in rows:
        idx = pid_to_idx.get(str(r["paper_id"]))
        if idx is None:
            continue
        strength = {"save": 3.0, "like": 1.5, "skip": -1.5, "read": 1.0, "share": 2.0}.get(r["type"], 0.5)
        if r["type"] == "read" and r["duration_seconds"] and r["duration_seconds"] > 120:
            strength = 2.0
        days = (datetime.utcnow() - r["created_at"].replace(tzinfo=None)).days
        vecs.append(embeddings_arr[idx].astype(np.float32))
        wts.append(strength * (0.95 ** days))

    if not vecs:
        return None

    taste = np.average(vecs, weights=wts, axis=0)
    norm  = np.linalg.norm(taste)
    return (taste / norm).astype(np.float32) if norm > 0 else taste


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "papers_loaded": len(paper_meta),
        "index_ready":   index_ready,
        "timestamp":     datetime.utcnow().isoformat(),
    }


# Auth
@app.post("/auth/register")
async def register(req: AuthRequest, conn: asyncpg.Connection = Depends(get_db)):
    if await conn.fetchrow("SELECT id FROM users WHERE email = $1", req.email):
        raise HTTPException(400, "Email already registered")
    uid = await conn.fetchval(
        "INSERT INTO users (email, password_hash) VALUES ($1, $2) RETURNING id",
        req.email, _hash_pw(req.password),
    )
    return {"access_token": _make_token(str(uid)), "token_type": "bearer"}

@app.post("/auth/login")
async def login(req: AuthRequest, conn: asyncpg.Connection = Depends(get_db)):
    user = await conn.fetchrow("SELECT id, password_hash FROM users WHERE email = $1", req.email)
    if not user or not _verify_pw(req.password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": _make_token(str(user["id"])), "token_type": "bearer"}


@app.get("/papers/search")
async def search_papers(q: str = Query(...), limit: int = Query(30)):
    q_stripped = q.strip()
    if not q_stripped:
        return []

    # ── Semantic search (preferred) ───────────────────────────────────────────
    if query_model is not None and paper_index is not None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()

            # Embed query in thread pool (CPU bound)
            query_vec = await loop.run_in_executor(
                None,
                lambda: query_model.encode(
                    q_stripped,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32)
            )

            matches = paper_index.search(query_vec, limit * 2)
            results = []
            for m in matches:
                pid = paper_id_list[int(m.key)]
                results.append(_paper_dict(pid))
                if len(results) >= limit:
                    break
            return results
        except Exception as e:
            print(f"[search] Semantic search failed, falling back: {e}")

    # ── Keyword fallback ──────────────────────────────────────────────────────
    q_lower  = q_stripped.lower()
    q_words  = q_lower.split()
    scored   = []

    for pid, p in paper_meta.items():
        title    = (p.get("title") or "").lower()
        abstract = (p.get("abstract") or "").lower()

        # Score: all words in title = 3, any word in title = 2, abstract = 1
        if all(w in title for w in q_words):
            scored.append((3, pid))
        elif any(w in title for w in q_words):
            scored.append((2, pid))
        elif all(w in abstract for w in q_words):
            scored.append((1, pid))

        if len(scored) >= limit * 3:
            break

    scored.sort(key=lambda x: x[0], reverse=True)
    return [_paper_dict(pid) for _, pid in scored[:limit]]

# ── User Profile & Preferences ────────────────────────────────────────────────

@app.get("/user/profile")
async def get_profile(
    uid:  str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    row = await conn.fetchrow(
        """SELECT topics, difficulty, display_name, role, institution,
                  primary_field, reading_goal, experience_level, weekly_goal
           FROM users WHERE id = $1""",
        uid
    )
    if not row:
        return {
            "topics": [], "difficulty": "any", "display_name": "",
            "role": "curious", "institution": "", "primary_field": "",
            "reading_goal": "broad", "experience_level": "intermediate",
            "weekly_goal": 5,
        }
    return {
        "topics":           row["topics"]           or [],
        "difficulty":       row["difficulty"]       or "any",
        "display_name":     row["display_name"]     or "",
        "role":             row["role"]             or "curious",
        "institution":      row["institution"]      or "",
        "primary_field":    row["primary_field"]    or "",
        "reading_goal":     row["reading_goal"]     or "broad",
        "experience_level": row["experience_level"] or "intermediate",
        "weekly_goal":      row["weekly_goal"]      or 5,
    }
 
 
@app.put("/user/profile")
async def update_profile(
    req:  ProfileIn,
    uid:  str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    await conn.execute(
        """UPDATE users SET
            topics           = $1,
            difficulty       = $2,
            display_name     = $3,
            role             = $4,
            institution      = $5,
            primary_field    = $6,
            reading_goal     = $7,
            experience_level = $8,
            weekly_goal      = $9
           WHERE id = $10""",
        req.topics, req.difficulty, req.display_name,
        req.role, req.institution, req.primary_field,
        req.reading_goal, req.experience_level, req.weekly_goal,
        uid,
    )
    return {"status": "updated"}
 
 
# Feed
@app.get("/feed/discover")
async def discover(
    limit: int = Query(20),
    uid:   str = Depends(current_user), sort: str = Query("relevance"),
    conn:  asyncpg.Connection = Depends(get_db),
):
    # 1. Fetch full user profile from DB
    user_row = await conn.fetchrow(
        """SELECT topics, difficulty, role, primary_field,
                  reading_goal, experience_level
           FROM users WHERE id = $1""",
        uid
    )

    active_topics    = (user_row["topics"]           or []) if user_row else []
    active_diff      = (user_row["difficulty"]       or "any") if user_row else "any"
    role             = (user_row["role"]             or "curious") if user_row else "curious"
    primary_field    = (user_row["primary_field"]    or "") if user_row else ""
    reading_goal     = (user_row["reading_goal"]     or "broad") if user_row else "broad"
    experience_level = (user_row["experience_level"] or "intermediate") if user_row else "intermediate"

    # 2. Build category keywords
    # Primary field gets 3x weight by repeating its keywords
    cat_keywords = []
    if primary_field:
        primary_cats = TOPIC_CATEGORY_MAP.get(primary_field, [primary_field])
        cat_keywords.extend(primary_cats * 3)
    for t in active_topics:
        cat_keywords.extend(TOPIC_CATEGORY_MAP.get(t, [t]))
    # Deduplicate while preserving order (primary field still appears 3x before dedup at match time)
    cat_keywords = [kw.lower() for kw in cat_keywords]

    # 3. Paper matching function — uses role + experience + reading_goal
    def paper_matches_prefs(p_dict) -> bool:
        venue    = (p_dict.get('venue')    or '').lower()
        p_cats   = ' '.join(safe_array(p_dict.get('categories', []))).lower()
        abstract = (p_dict.get('abstract') or '').lower()
        year     = p_dict.get('year') or 0

        # Topic filter — must match at least one keyword
        if cat_keywords:
            if not any(kw in p_cats or kw in venue for kw in cat_keywords):
                return False

        # Reading goal: stay_current → exclude papers older than 3 years (with 30% pass-through)
        if reading_goal == 'stay_current' and year and year < 2022:
            if random.random() < 0.70:
                return False

        # Role + reading goal: active researchers staying current want very recent
        if role == 'researcher' and reading_goal == 'stay_current':
            if year and year < 2020:
                return False

        # Experience level overrides difficulty for beginners
        effective_diff = active_diff
        if effective_diff == 'any' and experience_level == 'beginner':
            effective_diff = 'accessible'

        # Difficulty filter
        if effective_diff == 'accessible':
            heavy = ['math.ag', 'math.nt', 'hep-th', 'gr-qc']
            if any(kw in p_cats for kw in heavy):
                return False
        elif effective_diff == 'expert':
            if not p_cats or not abstract:
                return False

        return True

    # 4. Get taste vector + seen papers
    taste   = await _taste_vector(uid, conn) if index_ready else None
    seen    = {str(r["paper_id"]) for r in await conn.fetch(
        "SELECT paper_id FROM interactions WHERE user_id = $1", uid
    )}
    all_ids = list(paper_meta.keys())
    papers  = []

    # 5. Vector search (primary path — only when user has interactions)
    if taste is not None and paper_index is not None:
        matches = paper_index.search(taste, limit * 15)
        for m in matches:
            pid = paper_id_list[int(m.key)]
            if pid in seen:
                continue
            p_dict = paper_meta.get(pid, {})
            if paper_matches_prefs(p_dict):
                papers.append(_paper_dict(pid))
            if len(papers) >= limit:
                break

    # 6. Fallback — when no taste vector OR vector search didn't fill limit
    if len(papers) < limit:
        if cat_keywords:
            sample      = random.sample(all_ids, min(50000, len(all_ids)))
            candidates  = []
            for pid in sample:
                if pid in seen:
                    continue
                p_dict = paper_meta.get(pid, {})
                if not paper_matches_prefs(p_dict):
                    continue
                citation_count = p_dict.get('citation_count') or 0
                year           = p_dict.get('year') or 2000
                score          = citation_count + max(0, year - 2015) * 50
                candidates.append((score, pid))
            candidates.sort(reverse=True)
            existing_ids = {p['id'] for p in papers}
            for _, pid in candidates[:limit * 3]:
                if len(papers) >= limit:
                    break
                if pid not in existing_ids:
                    papers.append(_paper_dict(pid))
        else:
            sample     = random.sample(all_ids, min(50000, len(all_ids)))
            candidates = [
                (paper_meta[pid].get('citation_count') or 0, pid)
                for pid in sample if pid not in seen
            ]
            candidates.sort(reverse=True)
            papers = [_paper_dict(pid) for _, pid in candidates[:limit]]

    # 7. Re-sort based on reading_goal (applied after vector search)
    if reading_goal == 'stay_current' and papers:
        papers.sort(key=lambda p: p.get('year') or 0, reverse=True)
    elif reading_goal == 'deep_dive' and papers:
        papers.sort(key=lambda p: p.get('citation_count') or 0, reverse=True)
    # broad and specific: leave in taste-model order

    # 8. Diversity injection — 10% exploration outside taste results
    if taste is not None:
        explore_count  = max(2, limit // 10)
        taste_ids      = {p['id'] for p in papers}
        explore_sample = random.sample(all_ids, min(10000, len(all_ids)))
        explore_pool   = [pid for pid in explore_sample if pid not in seen and pid not in taste_ids]
        random.shuffle(explore_pool)
        explore_added  = 0
        for pid in explore_pool:
            if explore_added >= explore_count:
                break
            p_dict = paper_meta.get(pid, {})
            if paper_matches_prefs(p_dict):
                papers.append(_paper_dict(pid))
                explore_added += 1

    # 9. Safety net
    if len(papers) < 3:
        fallback = random.sample(list(paper_meta.keys()), min(limit, len(paper_meta)))
        papers   = [_paper_dict(pid) for pid in fallback]

    # 10. Final shuffle (preserves rough ordering but adds variety)
    # For stay_current/deep_dive we already sorted — don't re-shuffle
    if reading_goal not in ('stay_current', 'deep_dive'):
        random.shuffle(papers)

    return papers[:limit]
    
@app.get("/feed/daily")
async def daily_ten(
    uid:  str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    papers = await discover(limit=10, uid=uid, conn=conn)
    return {"papers": papers, "generated_at": datetime.utcnow().isoformat()}

@app.post("/feed/interaction")
async def log_interaction(
    req:  InteractionIn,
    uid:  str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    await conn.execute(
        "INSERT INTO interactions (user_id, paper_id, type, duration_seconds, swipe_velocity) "
        "VALUES ($1,$2,$3,$4,$5)",
        uid, req.paper_id, req.type, req.duration_seconds, req.swipe_velocity,
    )
    if req.type == "save":
        await conn.execute(
            "INSERT INTO saved_papers (user_id, paper_id) VALUES ($1,$2) ON CONFLICT DO NOTHING",
            uid, req.paper_id,
        )
    return {"status": "ok"}


# Library
@app.get("/library/saved")
async def get_saved(uid: str = Depends(current_user), conn: asyncpg.Connection = Depends(get_db)):
    rows = await conn.fetch(
        "SELECT paper_id FROM saved_papers WHERE user_id = $1 ORDER BY created_at DESC", uid
    )
    return [_paper_dict(r["paper_id"]) for r in rows]

@app.post("/library/saved/{paper_id}")
async def save_paper(
    paper_id: str,
    uid:  str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    await conn.execute(
        "INSERT INTO saved_papers (user_id, paper_id) VALUES ($1,$2) ON CONFLICT DO NOTHING",
        uid, paper_id,
    )
    return {"status": "saved"}

@app.delete("/library/saved/{paper_id}")
async def unsave_paper(
    paper_id: str,
    uid:  str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    await conn.execute(
        "DELETE FROM saved_papers WHERE user_id=$1 AND paper_id=$2", uid, paper_id
    )
    return {"status": "removed"}


# Playlists
@app.get("/library/playlists")
async def get_playlists(uid: str = Depends(current_user), conn: asyncpg.Connection = Depends(get_db)):
    rows = await conn.fetch(
        "SELECT id, name, description, is_public, created_at "
        "FROM playlists WHERE user_id=$1 ORDER BY created_at DESC",
        uid,
    )
    return [dict(r) for r in rows]

@app.post("/library/playlists")
async def create_playlist(
    req:  PlaylistIn,
    uid:  str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    row = await conn.fetchrow(
        "INSERT INTO playlists (user_id, name, description) VALUES ($1,$2,$3) "
        "RETURNING id, name, description, created_at",
        uid, req.name, req.description,
    )
    return dict(row)

@app.post("/library/playlists/{playlist_id}/papers/{paper_id}")
async def add_to_playlist(
    playlist_id: str,   # ← str, not int
    paper_id: str,
    uid: str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    pl = await conn.fetchrow(
        "SELECT id FROM playlists WHERE id::text=$1 AND user_id=$2", playlist_id, uid
    )
    if not pl:
        raise HTTPException(404, "Playlist not found")
    await conn.execute(
        "INSERT INTO playlist_papers (playlist_id, paper_id) VALUES ($1::uuid,$2) ON CONFLICT DO NOTHING",
        playlist_id, paper_id
    )
    return {"status": "added"}

@app.get("/feed/hot")
async def hot_papers(
    category: str = Query("cs_ml"),
    limit:    int = Query(20),
    uid:      str = Depends(current_user),
):
    import httpx
    import random
 
    ML_CATEGORIES = {'cs_ai', 'cs_ml', 'cs_nlp', 'cs_cv', 'cs_ro'}
    papers = []
 
    if category in ML_CATEGORIES:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://huggingface.co/api/daily_papers",
                    params={"limit": limit * 2},
                    headers={"User-Agent": "Cadence Research App"}
                )
                if r.status_code == 200:
                    data = r.json()
                    for item in data:
                        paper = item.get("paper", {})
                        title    = paper.get("title", "").strip()
                        abstract = paper.get("summary", "").strip()
                        if not title or not abstract:
                            continue
                        arxiv_id = paper.get("id", "")
                        
                        found_pid = None
                        if arxiv_id:
                            norm = arxiv_id.replace('/', '_')
                            for prefix in ['arxiv_', '']:
                                pid = f"{prefix}{norm}"
                                if pid in paper_meta:
                                    found_pid = pid
                                    break
                        
                        if found_pid:
                            p = _paper_dict(found_pid)
                        else:
                            p = {
                                "id":              f"hf_{arxiv_id.replace('/', '_')}",
                                "title":           title,
                                "abstract":        abstract,
                                "authors":         [a.get("name","") for a in paper.get("authors",[])],
                                "year":            2026,
                                "venue":           "arXiv",
                                "doi":             None,
                                "arxiv_id":        arxiv_id,
                                "categories":      ["cs.LG"],
                                "source":          "huggingface",
                                "citation_count":  0,
                                "open_access_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
                            }
                        
                        # ✅ CACHE IT: Store the constructed paper metadata in memory
                        transient_papers[p["id"]] = p
                        papers.append(p)
                        if len(papers) >= limit:
                            break
        except Exception as e:
            print(f"[hot] HuggingFace failed: {e}")
 
    if len(papers) < limit:
        try:
            FIELD_MAP = {
                'cs_ai': 'Computer Science', 'cs_ml': 'Computer Science',
                'cs_nlp': 'Computer Science', 'cs_cv': 'Computer Science',
                'bio_genomics': 'Biology', 'bio_neuro': 'Neuroscience',
                'med_clinical': 'Medicine', 'phys_quantum': 'Physics',
                'math_pure': 'Mathematics', 'econ_theory': 'Economics',
                'psych_cog': 'Psychology',
            }
            field = FIELD_MAP.get(category, 'Computer Science')
 
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query":            field + " research 2025 2026",
                        "fields":           "title,abstract,authors,year,venue,citationCount,externalIds,openAccessPdf",
                        "limit":            limit * 2,
                        "sort":             "citationCount",
                        "publicationDateOrYear": "2024-01-01:",
                    },
                    headers={"User-Agent": "Cadence Research App"}
                )
                if r.status_code == 200:
                    data = r.json()
                    existing_ids = {p['id'] for p in papers}
                    for item in data.get("data", []):
                        title    = item.get("title", "").strip()
                        abstract = item.get("abstract", "").strip()
                        if not title or len(abstract) < 50:
                            continue
                        arxiv_id = item.get("externalIds", {}).get("ArXiv", "")
                        oa_pdf   = (item.get("openAccessPdf") or {}).get("url", "")
                        pid      = f"s2_{item.get('paperId', '')}"
                        if pid in existing_ids:
                            continue
                        
                        p = {
                            "id":              pid,
                            "title":           title,
                            "abstract":        abstract,
                            "authors":         [a.get("name", "") for a in item.get("authors", [])][:10],
                            "year":            item.get("year"),
                            "venue":           item.get("venue", ""),
                            "doi":             item.get("externalIds", {}).get("DOI"),
                            "arxiv_id":        arxiv_id,
                            "categories":      [],
                            "source":          "semantic_scholar",
                            "citation_count":  item.get("citationCount", 0),
                            "open_access_url": oa_pdf or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None),
                        }
                        
                        # ✅ CACHE IT: Store Semantic Scholar trending items too
                        transient_papers[pid] = p
                        papers.append(p)
                        if len(papers) >= limit:
                            break
        except Exception as e:
            print(f"[hot] Semantic Scholar failed: {e}")
 
    # ── Internal fallback ─────────────────────────────────────────────────────
    if len(papers) < limit // 2:
        cat_kws = TOPIC_CATEGORY_MAP.get(category, [category])
        sample  = random.sample(list(paper_meta.keys()), min(100_000, len(paper_meta)))
        candidates = []
        for pid in sample:
            p = paper_meta.get(pid, {})
            p_cats = ' '.join(safe_array(p.get('categories', []))).lower()
            year   = p.get('year') or 0
            if not any(kw.lower() in p_cats for kw in cat_kws):
                continue
            if year and year < 2020:
                continue
            score = (p.get('citation_count') or 0) + max(0, year - 2015) * 100
            candidates.append((score, pid))
        candidates.sort(reverse=True)
        for _, pid in candidates[:limit]:
            papers.append(_paper_dict(pid))
 
    return papers[:limit]
 

import xml.etree.ElementTree as ET
import httpx
from fastapi import HTTPException

import xml.etree.ElementTree as ET

@app.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    # 1. Check local internal main database
    if paper_id in paper_meta:
        return _paper_dict(paper_id)
    
    # 2. ✅ Check Transient In-Memory Cache (Instant hit for fresh feed papers!)
    if paper_id in transient_papers:
        return transient_papers[paper_id]
    
    # 3. Fallback: If cache missed (e.g. server restarted), try live API proxy
    if paper_id.startswith("s2_"):
        s2_id = paper_id[3:]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}",
                    params={"fields": "title,abstract,authors,year,venue,citationCount,externalIds,openAccessPdf"},
                    headers={"User-Agent": "Cadence Research App"}
                )
                if r.status_code == 200:
                    item = r.json()
                    arxiv_id = item.get("externalIds", {}).get("ArXiv", "")
                    return {
                        "id":              paper_id,
                        "title":           item.get("title", "").strip(),
                        "abstract":        item.get("abstract", "").strip(),
                        "authors":         [a.get("name", "") for a in item.get("authors", [])][:10],
                        "year":            item.get("year"),
                        "venue":           item.get("venue", ""),
                        "doi":             item.get("externalIds", {}).get("DOI"),
                        "arxiv_id":        arxiv_id,
                        "categories":      [],
                        "source":          "semantic_scholar",
                        "citation_count":  item.get("citationCount", 0),
                        "open_access_url": (item.get("openAccessPdf") or {}).get("url") or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None),
                    }
        except Exception:
            pass

    if paper_id.startswith("hf_"):
        arxiv_id = paper_id[3:].replace('_', '/')
        # Safe-fallback query to official raw arXiv XML engine if cache drops out
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                ax_res = await client.get(f"http://export.arxiv.org/api/query?id_list={arxiv_id}")
                if ax_res.status_code == 200:
                    root = ET.fromstring(ax_res.text)
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    entry = root.find('atom:entry', ns)
                    if entry is not None:
                        title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
                        abstract = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
                        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
                        return {
                            "id":              paper_id,
                            "title":           title,
                            "abstract":        abstract,
                            "authors":         authors[:10],
                            "year":            2026,
                            "venue":           "arXiv",
                            "doi":             None,
                            "arxiv_id":        arxiv_id,
                            "categories":      ["cs.LG"],
                            "source":          "huggingface",
                            "citation_count":  0,
                            "open_access_url": f"https://arxiv.org/pdf/{arxiv_id}",
                        }
        except Exception:
            pass

    raise HTTPException(404, "Paper not found in local database, cache, or external networks")
    
# Add "more like this" endpoint
@app.get("/papers/{paper_id}/similar")
async def similar_papers(paper_id: str, limit: int = Query(20)):
    # Feature Upgrade: If the user reads an external trending paper, fetch real external recommendations!
    if paper_id.startswith("s2_") or paper_id.startswith("hf_"):
        s2_lookup_id = paper_id[3:] if paper_id.startswith("s2_") else f"arXiv:{paper_id[3:].replace('_', '/')}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/{s2_lookup_id}/recommendations",
                    params={
                        "fields": "title,abstract,authors,year,venue,citationCount,externalIds,openAccessPdf",
                        "limit": limit
                    },
                    headers={"User-Agent": "Cadence Research App"}
                )
                if r.status_code == 200:
                    data = r.json()
                    results = []
                    for item in data.get("recommendedPapers", []):
                        title = item.get("title", "").strip()
                        if not title:
                            continue
                        ax_id = item.get("externalIds", {}).get("ArXiv", "")
                        results.append({
                            "id":              f"s2_{item.get('paperId', '')}",
                            "title":           title,
                            "abstract":        item.get("abstract", "").strip(),
                            "authors":         [a.get("name", "") for a in item.get("authors", [])][:10],
                            "year":            item.get("year"),
                            "venue":           item.get("venue", ""),
                            "doi":             item.get("externalIds", {}).get("DOI"),
                            "arxiv_id":        ax_id,
                            "categories":      [],
                            "source":          "semantic_scholar",
                            "citation_count":  item.get("citationCount", 0),
                            "open_access_url": (item.get("openAccessPdf") or {}).get("url") or (f"https://arxiv.org/pdf/{ax_id}" if ax_id else None),
                        })
                    return results
        except Exception as e:
            print(f"[similar] External lookup failed: {e}")
        return []

    # Standard Internal Vector/FAISS Search Index Logic
    if not index_ready or paper_index is None:
        raise HTTPException(503, "Index not ready")
    
    try:
        idx = paper_id_list.index(paper_id)
    except ValueError:
        raise HTTPException(404, "Paper not in index")
    
    vec     = embeddings_arr[idx].astype(np.float32)
    matches = paper_index.search(vec, limit + 1)
    results = []
    for m in matches:
        pid = paper_id_list[int(m.key)]
        if pid != paper_id:
            results.append(_paper_dict(pid))
        if len(results) >= limit:
            break
    return results

@app.get("/papers/{paper_id}/unpaywall")
async def get_free_pdf(paper_id: str):
    # Route modification to resolve external DOI variables dynamically
    doi = None
    if paper_id in paper_meta:
        p = paper_meta.get(paper_id)
        doi = p.get("doi")
    elif paper_id.startswith("s2_") or paper_id.startswith("hf_"):
        try:
            # Re-use our dynamic routing logic to extract the transient paper's DOI
            transient_paper = await get_paper(paper_id)
            doi = transient_paper.get("doi")
        except Exception:
            pass

    if not doi:
        # Fallback if it's an arXiv/HF origin anyway (it's already fundamentally Open Access)
        if paper_id.startswith("hf_"):
            arxiv_id = paper_id[3:].replace('_', '/')
            return {"url": f"https://arxiv.org/pdf/{arxiv_id}", "source": "arXiv", "is_oa": True}
        return {"url": None, "source": None, "is_oa": False}
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": "cadence@rohanmarar.com"}
            )
            if r.status_code == 200:
                data = r.json()
                best = data.get("best_oa_location")
                if best:
                    url = best.get("url_for_pdf") or best.get("url")
                    host = best.get("host_type", "")
                    return {"url": url, "source": host, "is_oa": True}
        return {"url": None, "source": None, "is_oa": False}
    except Exception:
        return {"url": None, "source": None, "is_oa": False}

@app.get("/library/playlists/{playlist_id}")
async def get_single_playlist(
    playlist_id: str,
    uid: str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db)
):
    # Verify the playlist belongs to the user
    row = await conn.fetchrow(
        "SELECT id, name, description, is_public, created_at "
        "FROM playlists WHERE id=$1 AND user_id=$2",
        playlist_id, uid,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Playlist not found")
    return dict(row)

@app.patch("/library/playlists/{playlist_id}")
    
async def update_playlist(
    playlist_id: str,
    body: dict,
    uid: str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Name required")
    await conn.execute(
        "UPDATE playlists SET name=$1 WHERE id=$2 AND user_id=$3",
        name, playlist_id, uid
    )
    return {"status": "updated"}

@app.delete("/library/playlists/{playlist_id}/papers/{paper_id}")
async def remove_from_playlist(
    playlist_id: str,   # ← str, not int (UUID)
    paper_id: str,
    uid: str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    pl = await conn.fetchrow(
        "SELECT id FROM playlists WHERE id::text=$1 AND user_id=$2", playlist_id, uid
    )
    if not pl:
        raise HTTPException(404, "Playlist not found")
    await conn.execute(
        "DELETE FROM playlist_papers WHERE playlist_id::text=$1 AND paper_id=$2",
        playlist_id, paper_id
    )
    return {"status": "removed"}

@app.get("/library/playlists/{playlist_id}/papers")
async def get_playlist_papers(
    playlist_id: str,
    uid: str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db)
):
    # 1. Verify the playlist belongs to the user
    pl = await conn.fetchrow(
        "SELECT id FROM playlists WHERE id=$1 AND user_id=$2",
        playlist_id, uid
    )
    if not pl:
        raise HTTPException(status_code=404, detail="Playlist not found")
        
    # 2. Get the paper IDs associated with this playlist
    rows = await conn.fetch(
        "SELECT paper_id FROM playlist_papers WHERE playlist_id=$1",
        playlist_id,
    )
    
    # 3. Fetch paper details from your in-memory metadata
    # (Since you have the global paper_meta dict, this is very fast)
    paper_ids = [r["paper_id"] for r in rows]
    return [_paper_dict(pid) for pid in paper_ids if pid in paper_meta]

@app.delete("/library/playlists/{playlist_id}")
async def delete_playlist(
    playlist_id: str,
    uid: str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    await conn.execute(
        "DELETE FROM playlist_papers WHERE playlist_id::text=$1", playlist_id
    )
    await conn.execute(
        "DELETE FROM playlists WHERE id::text=$1 AND user_id=$2", playlist_id, uid
    )
    return {"status": "deleted"}


# ── Add all of these to main.py ───────────────────────────────────────────────

@app.get("/feed/liked")
async def get_liked_papers(
    limit: int = Query(20),
    uid:   str = Depends(current_user),
    conn:  asyncpg.Connection = Depends(get_db),
):
    """Papers the user has explicitly liked/saved — for Liked by You section."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT paper_id FROM interactions
        WHERE user_id = $1 AND type IN ('save', 'like')
        ORDER BY paper_id
        LIMIT $2
        """,
        uid, limit
    )
    return [_paper_dict(r["paper_id"]) for r in rows if r["paper_id"] in paper_meta]


@app.get("/feed/similar-to-saved")
async def get_similar_to_saved(
    limit: int = Query(20),
    uid:   str = Depends(current_user),
    conn:  asyncpg.Connection = Depends(get_db),
):
    """
    usearch neighbors of the user's saved papers.
    Finds papers similar to what the user explicitly saved.
    This is Spotify's 'More of What You Like' — seeds from saves, not taste vector.
    """
    if not index_ready or paper_index is None:
        return []

    # Get saved paper IDs
    rows = await conn.fetch(
        "SELECT paper_id FROM saved_papers WHERE user_id = $1 LIMIT 10", uid
    )
    if not rows:
        return []

    pid_to_idx = {pid: i for i, pid in enumerate(paper_id_list)}
    seen       = {str(r["paper_id"]) for r in rows}
    results    = []
    seen_result_ids = set()

    # For each saved paper find similar ones
    for row in rows[:5]:  # use top 5 saved papers as seeds
        pid = str(row["paper_id"])
        idx = pid_to_idx.get(pid)
        if idx is None:
            continue
        vec     = embeddings_arr[idx].astype(np.float32)
        matches = paper_index.search(vec, 10)
        for m in matches:
            candidate = paper_id_list[int(m.key)]
            if candidate not in seen and candidate not in seen_result_ids:
                results.append(_paper_dict(candidate))
                seen_result_ids.add(candidate)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    return results[:limit]


@app.get("/feed/trending")
async def get_trending(
    category: str = Query(""),
    limit:    int = Query(20),
    uid:      str = Depends(current_user),
):
    """
    Trending papers in a category — sorted by citation count DESC.
    Uses actual citation signal, not random shuffle.
    """
    cat_lower = category.lower()

    # Filter by category and sort by citation count
    candidates = []
    for pid, p in paper_meta.items():
        cats = ' '.join(safe_array(p.get('categories', []))).lower()
        if cat_lower and cat_lower not in cats:
            continue
        citation_count = p.get('citation_count') or 0
        year           = p.get('year') or 2000
        recency_boost  = max(0, year - 2015) * 50
        score          = citation_count + recency_boost
        candidates.append((score, pid))

    candidates.sort(reverse=True)

    # If no papers with citations, use all papers (or recent ones)
    if not candidates:
        # Rebuild without citation filter
        for pid, p in paper_meta.items():
            cats = ' '.join(safe_array(p.get('categories', []))).lower()
            if cat_lower and cat_lower not in cats:
                continue
            candidates.append((0, pid))
        random.shuffle(candidates)  # shuffle since all have same score

    selected = candidates[:limit]
    return [_paper_dict(pid) for _, pid in selected]


@app.get("/feed/todays-pick")
async def get_todays_pick(
    uid:  str = Depends(current_user),
    conn: asyncpg.Connection = Depends(get_db),
):
    import datetime

    today = datetime.date.today() 
    day_seed = int(datetime.date.today().strftime("%Y%m%d"))

    # 1. Check DB cache first — guarantees same paper all day
    cached = await conn.fetchrow(
        "SELECT paper_id FROM todays_pick_cache WHERE user_id=$1 AND date=$2",
        uid, today
    )
    if cached and cached["paper_id"] in paper_meta:
        return _paper_dict(cached["paper_id"])

    all_ids = list(paper_meta.keys())
    if not all_ids:
        raise HTTPException(404, "No papers available.")

    pick_pid = None
    taste = await _taste_vector(uid, conn) if index_ready else None

    # 2. Path A — personalized + deterministic
    if taste is not None and paper_index is not None:
        matches    = paper_index.search(taste, 100)
        candidates = []
        for m in matches:
            pid            = paper_id_list[int(m.key)]
            p              = paper_meta.get(pid, {})
            citation_count = p.get('citation_count') or 0
            year           = p.get('year') or 0
            score          = citation_count + (max(0, year - 2015) * 100)
            candidates.append((score, pid))

        if candidates:
            # Tiebreaker: sort by (score, pid) — stable across runs
            candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
            top_pool = candidates[:10]
            pick_idx = day_seed % len(top_pool)
            _, pick_pid = top_pool[pick_idx]

    # 3. Path B — cold start deterministic
    if not pick_pid:
        rng    = random.Random(day_seed)
        sample = rng.sample(all_ids, min(1000, len(all_ids)))
        pick_pid = max(sample, key=lambda pid: paper_meta[pid].get('citation_count') or 0)

    # 4. Cache in DB so it never changes today
    await conn.execute(
        """INSERT INTO todays_pick_cache (user_id, date, paper_id)
           VALUES ($1, $2, $3) ON CONFLICT DO NOTHING""",
        uid, today, pick_pid
    )

    return _paper_dict(pick_pid)


@app.post("/admin/reload")
async def hot_reload(secret: str = Query(...)):
    if secret != "cadence-reload-2024":
        raise HTTPException(status_code=403)
    
    global paper_meta, paper_id_list, paper_index, index_ready
    
    print("[reload] Hot reload started...")
    
    # Only add NEW papers (don't reload existing ones)
    new_count = 0
    with open(PAPERS_FILE) as f:
        for line in f:
            try:
                p   = json.loads(line)
                pid = str(p["paper_id"])
                if pid not in paper_meta:
                    paper_meta[pid] = {
                        "title":           p.get("title") or "",
                        "abstract":        p.get("abstract") or "",
                        "authors":         p.get("authors") or [],
                        "year":            p.get("year"),
                        "venue":           p.get("venue") or "",
                        "doi":             p.get("doi"),
                        "arxiv_id":        p.get("arxiv_id"),
                        "categories":      p.get("categories") or [],
                        "source":          p.get("source") or "",
                        "citation_count":  p.get("citation_count"),
                        "open_access_url": p.get("open_access_url"),
                    }
                    new_count += 1
            except: pass
    
    # Reload paper IDs
    with open(PAPER_IDS) as f:
        paper_id_list = json.load(f)
    
    # Reload usearch index
    import gc
    paper_index = None  # release old index
    gc.collect()        # force garbage collection
    # Now load new index
    from usearch.index import Index
    paper_index = Index.restore("cadence.usearch")
    index_ready = True
    
    print(f"[reload] Added {new_count:,} new papers, total: {len(paper_meta):,}")
    return {"status": "ok", "new_papers": new_count, "total": len(paper_meta)}
