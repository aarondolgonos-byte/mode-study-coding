"""
Collective diversity per system prompt variant (Gemini 3.1 Flash-Lite, 900 outputs).

Method (paper, Section 3.4):
  1. Embed every output with OpenAI text-embedding-3-large.
  2. For each variant and each user prompt (1-5), take the 5 outputs that answered that prompt
     (one per conversation) and compute the 10 pairwise cosine distances (1 - cosine similarity).
  3. Average those 10 distances -> diversity for that variant on that prompt.
  4. Average across the 5 prompts -> diversity score for the variant; SD is across the 5 prompts
     (sample SD, n - 1).

Reads the outputs from the raw_gemini31 tab of your Google Sheet (or from outputs/gemini31/log.jsonl
with --local), writes CSVs to outputs/diversity/, and adds two tabs to the same Google Sheet:
diversity_gemini31 (one row per variant) and diversity_gemini31_by_prompt.

Embeddings are cached in outputs/diversity/, so running it again costs nothing.

Usage:
  python diversity.py            # read outputs from the Google Sheet
  python diversity.py --local    # read outputs from outputs/gemini31/log.jsonl instead
  python diversity.py --no-upload
"""
import csv
import hashlib
import itertools
import json
import math
import os
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "diversity")
LOCAL_LOG = os.path.join(HERE, "outputs", "gemini31", "log.jsonl")
CACHE = os.path.join(OUT, "embeddings_text-embedding-3-large.jsonl")
EMBED_MODEL = "text-embedding-3-large"
SOURCE_TAB = "raw_gemini31"
MODES = ["Q", "Seed", "S(D)", "S(C)"]
VARIANTS = ["F%d/S%d" % (f, s) for f in (1, 2, 3) for s in (1, 2, 3)]
FORMATS = {"F1": "plain text", "F2": "markdown", "F3": "JSON"}
STRATEGIES = {"S1": "role", "S2": "rules", "S3": "examples"}
BATCH = 50


def load_env():
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


SSL_CTX = ssl_context()


# ------------------------------------------------------------------ read outputs
def read_local():
    if not os.path.exists(LOCAL_LOG):
        sys.exit("Can't find %s" % LOCAL_LOG)
    with open(LOCAL_LOG, encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    return [{"mode": r["mode"], "variant": r["variant"], "conversation": int(r["conversation"]),
             "prompt_number": int(r["prompt_number"]), "output": r["output"]} for r in recs]


def open_sheet():
    try:
        import gspread
    except ImportError:
        sys.exit("gspread isn't available. Install it first:  pip install -r requirements.txt")
    sheet_id = os.environ.get("DATA_SHEET_ID")
    if not sheet_id:
        sys.exit("Add DATA_SHEET_ID to .env")
    path = os.path.expanduser(os.environ.get("GCP_SECRETS_TOML",
                                             ".streamlit/secrets.toml"))
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)
    else:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(path, "rb") as fh:
            info = dict(tomllib.load(fh)["gcp_service_account"])
    return gspread.service_account_from_dict(info).open_by_key(sheet_id)


def read_sheet(sh):
    vals = sh.worksheet(SOURCE_TAB).get_all_values()
    head = vals[0]
    idx = {h: i for i, h in enumerate(head)}
    for c in ("mode", "variant", "conversation", "prompt_number", "output"):
        if c not in idx:
            sys.exit("Column '%s' missing from the %s tab." % (c, SOURCE_TAB))
    rows = []
    for v in vals[1:]:
        if not any(v):
            continue
        g = lambda c: v[idx[c]] if idx[c] < len(v) else ""
        rows.append({"mode": g("mode"), "variant": g("variant"), "conversation": int(g("conversation")),
                     "prompt_number": int(g("prompt_number")), "output": g("output")})
    return rows


def check(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["mode"], r["variant"], r["prompt_number"]), []).append(r)
    problems = []
    if len(rows) != 900:
        problems.append("expected 900 outputs, found %d" % len(rows))
    for m in MODES:
        for v in VARIANTS:
            for p in range(1, 6):
                g = groups.get((m, v, p), [])
                if sorted(r["conversation"] for r in g) != [1, 2, 3, 4, 5]:
                    problems.append("%s %s prompt %d has conversations %s"
                                    % (m, v, p, sorted(r["conversation"] for r in g)))
    empty = [r for r in rows if not r["output"].strip()]
    if problems:
        sys.exit("Data check failed:\n  " + "\n  ".join(problems[:20]))
    if empty:
        sys.exit("%d outputs are empty; the embeddings API can't embed empty text." % len(empty))
    print("Data check OK: 900 outputs, 36 variants x 5 prompts x 5 conversations.")
    return groups


# ------------------------------------------------------------------ embeddings
def text_key(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def load_cache():
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    cache[d["key"]] = d["embedding"]
    return cache


def embed_batch(texts, key):
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8")
    delay = 5
    for attempt in range(8):
        req = urllib.request.Request(base + "/embeddings", data=body, method="POST")
        req.add_header("Authorization", "Bearer " + key)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120, context=SSL_CTX) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            data = sorted(d["data"], key=lambda x: x["index"])
            return [x["embedding"] for x in data], d.get("usage", {}).get("total_tokens", 0)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                if "insufficient_quota" in msg:
                    sys.exit("OpenAI says the account has no credit. Add a few dollars of credit, then rerun.")
                print("   OpenAI busy (%d), retrying in %ds" % (e.code, delay))
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if e.code == 401:
                sys.exit("OpenAI rejected the key (401). Check OPENAI_API_KEY in .env.")
            sys.exit("OpenAI error %d: %s" % (e.code, msg))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(getattr(e, "reason", None), ssl.SSLError):
                sys.exit("SSL certificate problem. Install certificates (pip install certifi) and try again.")
            if attempt < 7:
                print("   network problem (%s), retrying in %ds" % (e, delay))
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise


def get_embeddings(texts):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    cache = load_cache()
    todo, seen = [], set()
    for t in texts:
        k = text_key(t)
        if k not in cache and k not in seen:
            todo.append(t)
            seen.add(k)
    if todo:
        if not key:
            sys.exit("No OpenAI key. Put a line  OPENAI_API_KEY=your_key  in .env")
        print("Embedding %d outputs with %s (%d already cached)..." % (len(todo), EMBED_MODEL, len(texts) - len(todo)))
        tokens = 0
        with open(CACHE, "a", encoding="utf-8") as fh:
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i + BATCH]
                embs, used = embed_batch(chunk, key)
                tokens += used
                for t, e in zip(chunk, embs):
                    cache[text_key(t)] = e
                    fh.write(json.dumps({"key": text_key(t), "model": EMBED_MODEL, "embedding": e}) + "\n")
                fh.flush()
                print("   %d/%d" % (min(i + BATCH, len(todo)), len(todo)))
        print("Done. %d tokens (about $%.3f)." % (tokens, tokens * 0.13 / 1e6))
    else:
        print("All embeddings already cached; no API calls needed.")
    return {text_key(t): cache[text_key(t)] for t in texts}


def cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (na * nb)


# ------------------------------------------------------------------ analysis
def analyse(groups, emb):
    pair_rows, prompt_rows, variant_rows = [], [], []
    for m in MODES:
        for v in VARIANTS:
            per_prompt = []
            for p in range(1, 6):
                g = sorted(groups[(m, v, p)], key=lambda r: r["conversation"])
                dists = []
                for a, b in itertools.combinations(g, 2):
                    d = cosine_distance(emb[text_key(a["output"])], emb[text_key(b["output"])])
                    dists.append(d)
                    pair_rows.append({"mode": m, "variant": v, "prompt_number": p,
                                      "conversation_a": a["conversation"], "conversation_b": b["conversation"],
                                      "cosine_distance": round(d, 6)})
                mean_p = statistics.mean(dists)
                per_prompt.append(mean_p)
                prompt_rows.append({"mode": m, "variant": v, "format": FORMATS[v[:2]],
                                    "strategy": STRATEGIES[v[3:]], "prompt_number": p, "n_outputs": len(g),
                                    "n_pairs": len(dists), "mean_pairwise_cosine_distance": round(mean_p, 6)})
            row = {"mode": m, "variant": v, "format": FORMATS[v[:2]], "strategy": STRATEGIES[v[3:]],
                   "diversity_mean": round(statistics.mean(per_prompt), 6),
                   "diversity_sd": round(statistics.stdev(per_prompt), 6)}
            for p, val in enumerate(per_prompt, 1):
                row["prompt_%d" % p] = round(val, 6)
            variant_rows.append(row)
    return pair_rows, prompt_rows, variant_rows


VARIANT_COLS = ["mode", "variant", "format", "strategy", "diversity_mean", "diversity_sd",
                "prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5"]
PROMPT_COLS = ["mode", "variant", "format", "strategy", "prompt_number", "n_outputs", "n_pairs",
               "mean_pairwise_cosine_distance"]
PAIR_COLS = ["mode", "variant", "prompt_number", "conversation_a", "conversation_b", "cosine_distance"]


def write_csv(name, cols, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def upload(sh, tabs):
    existing = {w.title: w for w in sh.worksheets()}
    for name, cols, rows in tabs:
        if name in existing:
            sh.del_worksheet(existing[name])
        ws = sh.add_worksheet(title=name, rows=len(rows) + 1, cols=len(cols))
        ws.update(values=[cols] + [[r[c] for c in cols] for r in rows], range_name="A1",
                  value_input_option="RAW")
        ws.freeze(rows=1)
        print("   %s -> %s (%d rows)" % (name, sh.title, len(rows)))


def main():
    load_env()
    os.makedirs(OUT, exist_ok=True)
    local = "--local" in sys.argv
    do_upload = "--no-upload" not in sys.argv
    sh = None
    if local:
        print("Reading outputs from %s" % LOCAL_LOG)
        rows = read_local()
    else:
        sh = open_sheet()
        print("Reading outputs from the %s tab of '%s'" % (SOURCE_TAB, sh.title))
        rows = read_sheet(sh)
        if os.path.exists(LOCAL_LOG):  # extra safety: sheet text should match the original log
            loc = {(r["mode"], r["variant"], r["conversation"], r["prompt_number"]): r["output"] for r in read_local()}
            diff = sum(1 for r in rows if loc.get((r["mode"], r["variant"], r["conversation"],
                                                   r["prompt_number"])) != r["output"])
            print("Compared with outputs/gemini31/log.jsonl: %d of %d outputs differ." % (diff, len(rows)))
    groups = check(rows)
    emb = get_embeddings([r["output"] for r in rows])
    pair_rows, prompt_rows, variant_rows = analyse(groups, emb)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    p1 = write_csv("diversity_gemini31_by_variant.csv", VARIANT_COLS, variant_rows)
    write_csv("diversity_gemini31_by_prompt.csv", PROMPT_COLS, prompt_rows)
    write_csv("diversity_gemini31_pairs.csv", PAIR_COLS, pair_rows)
    with open(os.path.join(OUT, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump({"run_date_utc": stamp, "embedding_model": EMBED_MODEL,
                   "dimensions": len(next(iter(emb.values()))), "source": "local log" if local else SOURCE_TAB,
                   "n_outputs": len(rows), "distance": "1 - cosine similarity",
                   "sd": "sample SD (n-1) across the 5 prompt-level means"}, fh, indent=2)

    print("\nDiversity per variant (mean pairwise cosine distance, mean +/- SD across 5 prompts)")
    print("%-5s " % "" + " ".join("%-13s" % v for v in VARIANTS))
    for m in MODES:
        cells = ["%.3f+/-%.3f" % (r["diversity_mean"], r["diversity_sd"]) for r in variant_rows if r["mode"] == m]
        print("%-5s " % m + " ".join("%-13s" % c for c in cells))
    print("\nSaved CSVs in %s" % os.path.dirname(p1))

    if do_upload:
        if sh is None:
            try:
                sh = open_sheet()
            except SystemExit as e:
                print("Google Sheets upload skipped: %s" % e)
                return
        print("Uploading to Google Sheets...")
        upload(sh, [("diversity_gemini31", VARIANT_COLS, variant_rows),
                    ("diversity_gemini31_by_prompt", PROMPT_COLS, prompt_rows)])
        print("   https://docs.google.com/spreadsheets/d/%s" % sh.id)


if __name__ == "__main__":
    main()
