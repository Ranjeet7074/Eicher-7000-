"""
Self-hosted RAG for the Eicher Pro 7000 BSVI Operator's Manual (v2: fast-feel).
  python rag.py build docs/manual.pdf   # one-time indexing
  python rag.py ask "How often to change engine oil?"
  python rag.py search "DEF warning lamp"
  python rag.py chat
"""
import sys, re, json, math, os, pickle, subprocess, collections, urllib.request

INDEX = "index.pkl"
CACHE_FILE = "answer_cache.json"
EMB_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
CHUNK_WORDS, OVERLAP = 220, 40
TOP_K = 4

tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())

# Query expansion: truck vocabulary the manual and drivers use differently
SYNONYMS = {
    "oil": "lubricant engine oil dipstick", "adblue": "urea def scr", "def": "adblue urea scr",
    "urea": "adblue def", "brake": "braking air pressure", "tyre": "tire wheel pressure",
    "tire": "tyre wheel pressure", "service": "maintenance schedule interval", "interval": "schedule km service",
    "warning": "lamp indicator light cluster", "light": "lamp indicator", "lamp": "light indicator warning",
    "battery": "electrical charge", "coolant": "radiator cooling temperature", "clutch": "gearbox transmission",
    "breakdown": "emergency eos on-road service", "helpline": "eos toll free 1800", "warranty": "claim coverage",
    "fuel": "diesel filter tank", "ac": "air conditioning climate", "gear": "transmission shift",
}
def expand(q):
    extra = " ".join(SYNONYMS[t] for t in tok(q) if t in SYNONYMS)
    return q + " " + extra if extra else q

# ---------- ingestion ----------
def extract_pages(pdf):
    try:
        import fitz
        return [p.get_text("text") for p in fitz.open(pdf)]
    except ImportError:
        out = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
        return out.split("\f")

def chunk_pages(pages):
    chunks = []
    for pno, text in enumerate(pages, 1):
        text = re.sub(r"[ \t]+", " ", text).strip()
        if len(text) < 40: continue
        title = next((l.strip() for l in text.splitlines() if 3 < len(l.strip()) < 70), "")
        words = text.split(); step = CHUNK_WORDS - OVERLAP
        for i in range(0, max(len(words), 1), step):
            body = " ".join(words[i:i + CHUNK_WORDS])
            if len(body.split()) < 25 and i > 0: break
            chunks.append({"page": pno, "title": title, "text": body})
    return chunks

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [collections.Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / len(docs)
        df = collections.Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + .5) / (f + .5)) for t, f in df.items()}
        self.inv = collections.defaultdict(list)          # inverted index = faster search
        for i, tf in enumerate(self.tf):
            for t in tf: self.inv[t].append(i)
    def scores(self, q):
        s = collections.defaultdict(float)
        for t in set(q):
            if t not in self.idf: continue
            for i in self.inv[t]:
                f = self.tf[i][t]
                s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s

def build(pdf):
    chunks = chunk_pages(extract_pages(pdf))
    print(f"{len(chunks)} chunks")
    docs = [tok(c["title"] + " " + c["text"]) for c in chunks]
    emb = None
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMB_MODEL)
        emb = m.encode([c["title"] + ". " + c["text"] for c in chunks],
                       normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    except ImportError:
        print("sentence-transformers not installed -> keyword (BM25) search only")
    pickle.dump({"chunks": chunks, "docs": docs, "emb": emb}, open(INDEX, "wb"))
    print("saved", INDEX)

# ---------- retrieval ----------
_state = {}
def load():
    if not _state:
        d = pickle.load(open(INDEX, "rb"))
        d["bm25"] = BM25(d["docs"])
        if d["emb"] is not None:
            from sentence_transformers import SentenceTransformer
            d["model"] = SentenceTransformer(EMB_MODEL)
        _state.update(d)
    return _state

def retrieve(q, k=TOP_K):
    d = load(); q = expand(q)
    bm = d["bm25"].scores(tok(q))
    rankings = [sorted(bm, key=bm.get, reverse=True)[:50]]
    if d.get("model") is not None:
        qv = d["model"].encode([q], normalize_embeddings=True)[0]
        sims = d["emb"] @ qv
        rankings.append(list(sims.argsort()[::-1][:50]))
    rrf = collections.defaultdict(float)
    for r in rankings:
        for rank, i in enumerate(r): rrf[int(i)] += 1 / (60 + rank)
    return [d["chunks"][i] for i in sorted(rrf, key=rrf.get, reverse=True)[:k]]

# ---------- generation ----------
PROMPT = """You answer questions about the Eicher Pro 7000 BSVI truck Operator's Manual.
Use ONLY the context provided. Answer in 2-4 short sentences or a short list. Cite pages like [p.45].
If the context does not contain the answer, reply exactly: I couldn't find that in the manual."""

GREETINGS = {"hi", "hello", "hey", "hii", "good morning", "good evening"}
GREETING_REPLY = "Hello! Ask me anything about the Eicher Pro 7000 BSVI manual, e.g. how to check engine oil level."

_cache = {}
if os.path.exists(CACHE_FILE):
    try: _cache = json.load(open(CACHE_FILE))
    except Exception: _cache = {}
def _save_cache():
    try: json.dump(_cache, open(CACHE_FILE, "w"))
    except Exception: pass

def warmup():
    """Load the model into memory so the first real question is not slow."""
    try:
        body = json.dumps({"model": LLM_MODEL, "keep_alive": "60m", "stream": False, "think": False,
                           "messages": [{"role": "user", "content": "hi"}], "options": {"num_predict": 1}}).encode()
        urllib.request.urlopen(urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"}), timeout=120)
    except Exception: pass

def _snippet(c, n=350):
    t = c["text"]; return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "..."

def stream_ask(q):
    """Generator: yields progressively longer text. 1) instant best passage, 2) streamed LLM answer."""
    key = q.strip().lower()
    if key in GREETINGS:
        yield GREETING_REPLY; return
    if key in _cache:
        yield _cache[key]; return
    ctx = retrieve(q)
    pages = sorted({c["page"] for c in ctx})
    best = ctx[0]
    source_block = f"\n\n---\n📄 **Best match (p.{best['page']}):** {_snippet(best)}\n\n*Sources: pages {pages}*"
    yield f"📄 **Best match (p.{best['page']}):** {_snippet(best)}\n\n⏳ *Writing summary...*"
    context = "\n\n".join(f"[p.{c['page']}] {c['title']}\n{c['text']}" for c in ctx)
    body = json.dumps({"model": LLM_MODEL, "stream": True, "think": False, "keep_alive": "60m",
        "options": {"temperature": 0.1, "num_predict": 250, "num_ctx": 3072},
        "messages": [{"role": "system", "content": PROMPT},
                     {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q} /no_think"}]}).encode()
    answer = ""
    try:
        r = urllib.request.urlopen(urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"}), timeout=300)
        for line in r:
            if not line.strip(): continue
            piece = json.loads(line).get("message", {}).get("content", "")
            answer += piece
            shown = re.sub(r"<think>.*?(</think>|$)", "", answer, flags=re.S).strip()
            if shown: yield shown + source_block
        final = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip() + source_block
        _cache[key] = final; _save_cache()
        yield final
    except Exception as e:
        yield (f"📄 **Best match (p.{best['page']}):** {_snippet(best, 600)}\n\n*Sources: pages {pages}*\n\n"
               f"(LLM unavailable: {e}. Run `ollama serve &` and check `ollama list`.)")

def ask(q):
    out = ""
    for out in stream_ask(q): pass
    return out

def show(q):
    for c in retrieve(q):
        print(f"--- p.{c['page']} | {c['title']}\n{c['text'][:400]}\n")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build": build(sys.argv[2])
    elif cmd == "ask": print(ask(" ".join(sys.argv[2:])))
    elif cmd == "search": show(" ".join(sys.argv[2:]))
    elif cmd == "chat":
        while (q := input("\nQ> ").strip()) not in ("", "exit"): print(ask(q))
    else: print(__doc__)
