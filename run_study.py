#!/usr/bin/env python3
"""
Mode-adherence study runner (no UI, standard library only).

Runs every system-prompt variant x Latin-square conversation through the
Gemini API, logs every turn, resumes safely after a crash, and writes the raw
data and blinded rating sheets straight into Google Sheets (plus CSV backups).

Commands (run from this folder):
    python3 run_study.py check    # confirm key, model IDs and settings work (1 tiny call per model)
    python3 run_study.py show     # print the exact prompts/orders the script will send
    python3 run_study.py test     # small trial run into outputs/test/
    python3 run_study.py main     # Gemini 3.1 Flash-Lite: 36 variants x 5 conversations x 5 turns = 900
    python3 run_study.py sub38    # Gemini 3.8 Flash: Conversation 1 of all 36 variants = 180
    python3 run_study.py upload   # (re)build the CSVs and push them to Google Sheets again
    python3 run_study.py all      # main + sub38 + sheets

Safe to re-run: finished turns are never regenerated. If a run stops halfway,
run the same command again and it continues where it left off.
"""

import csv
import json
import os
import random
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(HERE, "prompts.json")
OUT_DIR = os.path.join(HERE, "outputs")
SETTINGS_FILE = os.path.join(OUT_DIR, "resolved_settings.json")

# ---------------------------------------------------------------- study design
MODES = ["Q", "Seed", "S(D)", "S(C)"]
FORMATS = {"F1": "Plain text", "F2": "Markdown", "F3": "JSON"}
STRATEGIES = {"S1": "Role assignment", "S2": "Rules", "S3": "Examples"}
VARIANTS = ["F%d/S%d" % (f, s) for f in (1, 2, 3) for s in (1, 2, 3)]
CONVERSATIONS = [1, 2, 3, 4, 5]
HARD_PROMPTS = {4, 5}              # prompts 4 and 5 conflict with the mode
DIVERGENT_PROMPTS = {1, 2, 5}      # per paper: 1, 2, 5 divergent; 3, 4 convergent

GEN_CONFIG = {
    "temperature": 1.0,
    "maxOutputTokens": 8192,
    "thinkingConfig": {"thinkingLevel": "low"},
}

# Which model each run uses. Preference order; overridable in .env.
MODEL_CANDIDATES = {
    "gemini31": ["gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview"],
    "gemini38": ["gemini-3.8-flash", "gemini-3.8-flash-preview"],
}
MODEL_ENV_OVERRIDE = {"gemini31": "MAIN_MODEL", "gemini38": "SUB_MODEL"}
MODEL_PREFIX = {"gemini31": "gemini-3.1-flash-lite", "gemini38": "gemini-3.8-flash"}

ENDPOINTS = {
    "aistudio": "https://generativelanguage.googleapis.com/v1beta",
    "vertex": "https://aiplatform.googleapis.com/v1/publishers/google",
}

SHUFFLE_SEED = 20261107
MAX_RETRIES = 8
REQUEST_TIMEOUT = 180

_print_lock = threading.Lock()
_file_lock = threading.Lock()


def say(*a):
    with _print_lock:
        print(*a, flush=True)


# ---------------------------------------------------------------- setup helpers
def load_env():
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("No API key. Put a line  GEMINI_API_KEY=your_key  in the .env file in this folder.")
    return key


def ssl_context():
    try:
        import certifi  # present if you have requests/streamlit installed
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


SSL_CTX = ssl_context()


def load_prompts():
    with open(PROMPTS_FILE, encoding="utf-8") as fh:
        d = json.load(fh)
    square = {int(k): v for k, v in d["latin_square"].items()}
    users = {m: {int(k): v for k, v in d["user_prompts"][m].items()} for m in MODES}
    systems = d["system_prompts"]
    # sanity checks
    for c in CONVERSATIONS:
        assert sorted(square[c]) == [1, 2, 3, 4, 5], "Latin square row %d is not 1-5" % c
    for pos in range(5):
        assert sorted(square[c][pos] for c in CONVERSATIONS) == [1, 2, 3, 4, 5], "Latin square column %d" % (pos + 1)
    for m in MODES:
        assert sorted(users[m]) == [1, 2, 3, 4, 5], "Mode %s needs prompts 1-5" % m
        assert sorted(systems[m]) == sorted(VARIANTS), "Mode %s needs all 9 variants" % m
    return square, users, systems


class FatalAPIError(Exception):
    pass


def http_json(method, url, key, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("x-goog-api-key", key)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def gen_url(endpoint, model):
    base = os.environ.get("GEMINI_BASE_URL") or ENDPOINTS[endpoint]
    return "%s/models/%s:generateContent" % (base.rstrip("/"), model)


def call_gemini(key, endpoint, model, system_text, contents):
    """One generateContent call with retries for transient errors only.
    Returns (response_json, retries_used, latency_seconds)."""
    body = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": contents,
        "generationConfig": GEN_CONFIG,
    }
    url = gen_url(endpoint, model)
    delay = 4.0
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = http_json("POST", url, key, body)
            return resp, attempt, round(time.time() - t0, 2)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:800]
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                say("   ...API busy (HTTP %d), retrying in %.0fs" % (e.code, delay))
                time.sleep(delay + random.random() * 2)
                delay = min(delay * 2, 120)
                continue
            raise FatalAPIError("HTTP %d from Gemini: %s" % (e.code, detail))
        except ssl.SSLError as e:
            raise FatalAPIError(
                "SSL certificate problem (%s). On a Mac, open Finder > Applications > Python 3.x and "
                "double-click 'Install Certificates.command', then run again." % e)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise FatalAPIError(
                    "SSL certificate problem. On a Mac, open Finder > Applications > Python 3.x and "
                    "double-click 'Install Certificates.command', then run again.")
            if attempt < MAX_RETRIES:
                say("   ...network problem (%s), retrying in %.0fs" % (e, delay))
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise FatalAPIError("Network error: %s" % e)
    raise FatalAPIError("Gave up after %d retries" % MAX_RETRIES)


def parse_response(resp):
    """Pull the visible text, finish reason, usage and the raw model turn out of a response."""
    cands = resp.get("candidates") or []
    cand = cands[0] if cands else {}
    content = cand.get("content")
    parts = (content or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    finish = cand.get("finishReason") or ("BLOCKED:" + (resp.get("promptFeedback") or {}).get("blockReason", "?")
                                          if not cands else "NONE")
    usage = resp.get("usageMetadata") or {}
    return {
        "output": text,
        "finish_reason": finish,
        "model_version": resp.get("modelVersion", ""),
        "response_id": resp.get("responseId", ""),
        "prompt_tokens": usage.get("promptTokenCount", ""),
        "output_tokens": usage.get("candidatesTokenCount", ""),
        "thinking_tokens": usage.get("thoughtsTokenCount", ""),
        "total_tokens": usage.get("totalTokenCount", ""),
        # Raw model turn (incl. thought signatures) is replayed as history, the way the SDK chat does.
        "model_content": content if parts else None,
    }


# ---------------------------------------------------------------- model resolution
def list_models(key):
    base = (os.environ.get("GEMINI_BASE_URL") or ENDPOINTS["aistudio"]).rstrip("/")
    names, page = [], None
    while True:
        url = base + "/models?pageSize=1000" + ("&pageToken=" + page if page else "")
        r = http_json("GET", url, key)
        names += [m["name"].split("/", 1)[-1] for m in r.get("models", [])]
        page = r.get("nextPageToken")
        if not page:
            return names


def resolve_settings(key, force=False):
    if os.path.exists(SETTINGS_FILE) and not force:
        with open(SETTINGS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    os.makedirs(OUT_DIR, exist_ok=True)
    settings = {"endpoint": None, "models": {}}
    available = None
    if os.environ.get("GEMINI_ENDPOINT") == "vertex":
        settings["endpoint"] = "vertex"
    else:
        try:
            available = list_models(key)
            settings["endpoint"] = "aistudio"
        except urllib.error.HTTPError as e:
            say("Model list on the Gemini API (AI Studio) failed with HTTP %d; will try Vertex AI express."
                % e.code)
            settings["endpoint"] = "vertex"
        except Exception as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                sys.exit("SSL certificate problem. On a Mac, open Finder > Applications > Python 3.x and "
                         "double-click 'Install Certificates.command', then run again.")
            sys.exit("Could not reach Google: %s" % e)

    for run, cands in MODEL_CANDIDATES.items():
        override = os.environ.get(MODEL_ENV_OVERRIDE[run])
        if override:
            settings["models"][run] = override
            continue
        chosen = None
        if available is not None:
            for c in cands:
                if c in available:
                    chosen = c
                    break
            if not chosen:
                similar = [n for n in available if n.startswith(MODEL_PREFIX[run])
                           and not any(x in n for x in ("image", "tts", "live", "audio"))]
                if similar:
                    chosen = sorted(similar)[0]
            if not chosen:
                flash = sorted(n for n in available if "flash" in n)
                say("Could not find a %s model. Models with 'flash' your key can use:\n  %s"
                    % (MODEL_PREFIX[run], "\n  ".join(flash)))
                say("Set %s=<model id> in .env and run 'check' again." % MODEL_ENV_OVERRIDE[run])
        else:
            chosen = cands[0]
        settings["models"][run] = chosen
    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)
    return settings


# ---------------------------------------------------------------- logging + resume
def log_paths(run_name):
    d = os.path.join(OUT_DIR, run_name)
    os.makedirs(d, exist_ok=True)
    return d, os.path.join(d, "log.jsonl"), os.path.join(d, "errors.jsonl")


def append_jsonl(path, rec):
    with _file_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def read_log(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                say("Warning: skipped an unreadable line %d in %s (probably a crash mid-write)." % (i, path))
    return recs


def conv_key(r):
    return (r["mode"], r["variant"], r["conversation"])


# ---------------------------------------------------------------- running
def run_conversation(key, settings, run_name, model, mode, variant, conv, square, users, systems,
                     done_turns, log_file, run_id):
    order = square[conv]
    system_text = systems[mode][variant]
    contents = []
    for r in sorted(done_turns, key=lambda x: x["turn"]):  # rebuild history from the log
        contents.append({"role": "user", "parts": [{"text": r["user_prompt"]}]})
        if r.get("model_content"):
            contents.append(r["model_content"])
    start_turn = len(done_turns) + 1
    for turn in range(start_turn, 6):
        pnum = order[turn - 1]
        user_text = users[mode][pnum]
        contents.append({"role": "user", "parts": [{"text": user_text}]})
        resp, retries, latency = call_gemini(key, settings["endpoint"], model, system_text, contents)
        p = parse_response(resp)
        rec = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run": run_name,
            "model_requested": model,
            "model_version": p["model_version"],
            "mode": mode,
            "variant": variant,
            "format": FORMATS[variant[:2]],
            "strategy": STRATEGIES[variant[3:]],
            "conversation": conv,
            "turn": turn,
            "prompt_number": pnum,
            "prompt_type": "hard" if pnum in HARD_PROMPTS else "standard",
            "stage": "divergent" if pnum in DIVERGENT_PROMPTS else "convergent",
            "user_prompt": user_text,
            "output": p["output"],
            "finish_reason": p["finish_reason"],
            "prompt_tokens": p["prompt_tokens"],
            "output_tokens": p["output_tokens"],
            "thinking_tokens": p["thinking_tokens"],
            "total_tokens": p["total_tokens"],
            "retries": retries,
            "latency_s": latency,
            "response_id": p["response_id"],
            "temperature": GEN_CONFIG["temperature"],
            "thinking_level": GEN_CONFIG["thinkingConfig"]["thinkingLevel"],
            "max_output_tokens": GEN_CONFIG["maxOutputTokens"],
            "model_content": p["model_content"],
        }
        append_jsonl(log_file, rec)
        if p["model_content"]:
            contents.append(p["model_content"])
        flag = "" if p["finish_reason"] == "STOP" else "  <-- finish: %s" % p["finish_reason"]
        say("  %-5s %s conv %d turn %d (prompt %d) done%s" % (mode, variant, conv, turn, pnum, flag))


def execute(key, settings, run_name, model_run, jobs, square, users, systems):
    model = settings["models"].get(model_run)
    if not model:
        sys.exit("No model resolved for %s. Run 'python3 run_study.py check' first." % model_run)
    out_dir, log_file, err_file = log_paths(run_name)
    existing = read_log(log_file)
    by_conv = {}
    for r in existing:
        if r.get("model_requested") != model:
            sys.exit("outputs/%s already has data from model %s, but this run would use %s. "
                     "Move that folder away first." % (run_name, r.get("model_requested"), model))
        by_conv.setdefault(conv_key(r), []).append(r)
    todo = [j for j in jobs if len(by_conv.get(j, [])) < 5]
    done_calls = sum(min(len(by_conv.get(j, [])), 5) for j in jobs)
    total_calls = len(jobs) * 5
    say("\n%s | model %s | %d/%d outputs already done | %d conversations to run"
        % (run_name, model, done_calls, total_calls, len(todo)))
    if not todo:
        return
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workers = int(os.environ.get("WORKERS", "4"))
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_conversation, key, settings, run_name, model, m, v, c, square, users,
                          systems, by_conv.get((m, v, c), []), log_file, run_id): (m, v, c)
                for (m, v, c) in todo}
        for f in as_completed(futs):
            try:
                f.result()
            except FatalAPIError as e:
                m, v, c = futs[f]
                append_jsonl(err_file, {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                        "mode": m, "variant": v, "conversation": c, "error": str(e)})
                failed.append((futs[f], str(e)))
                say("  !! %s %s conv %d stopped: %s" % (m, v, c, str(e)[:300]))
    final = read_log(log_file)
    say("%s: %d/%d outputs logged." % (run_name, len(final), total_calls))
    if failed:
        say("%d conversations stopped on errors (see %s). Fix the problem and run the same command "
            "again; finished turns are kept." % (len(failed), err_file))
        sys.exit(1)


def jobs_main():
    return [(m, v, c) for m in MODES for v in VARIANTS for c in CONVERSATIONS]


def jobs_sub38():
    return [(m, v, 1) for m in MODES for v in VARIANTS]


def jobs_test():
    return [(m, "F1/S2", 1) for m in MODES]


# ---------------------------------------------------------------- sheets
RAW_COLS = ["run_id", "timestamp_utc", "model_requested", "model_version", "mode", "variant", "format",
            "strategy", "conversation", "turn", "prompt_number", "prompt_type", "stage", "user_prompt",
            "output", "finish_reason", "prompt_tokens", "output_tokens", "thinking_tokens", "total_tokens",
            "retries", "latency_s", "response_id", "temperature", "thinking_level", "max_output_tokens",
            "blinded_conversation_id"]
VIOLATION_COL = "violation (checked = yes)"
RATE_COLS = ["row", "blinded_conversation_id", "mode", "turn", "user_prompt", "model_output",
             VIOLATION_COL, "open_code"]
KEY_COLS = ["blinded_conversation_id", "model", "mode", "variant", "conversation"]


def blinded_ids(recs, salt):
    rng = random.Random("%s-%s" % (SHUFFLE_SEED, salt))
    convs = sorted({conv_key(r) for r in recs})
    ids, used = {}, set()
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    for k in convs:
        while True:
            cid = "C-" + "".join(rng.choice(alphabet) for _ in range(5))
            if cid not in used:
                used.add(cid)
                ids[k] = cid
                break
    return ids


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def rating_rows(recs, ids, salt):
    rng = random.Random("%s-order-%s" % (SHUFFLE_SEED, salt))
    convs = sorted({conv_key(r) for r in recs})
    rng.shuffle(convs)
    by = {}
    for r in recs:
        by.setdefault(conv_key(r), []).append(r)
    rows, n = [], 0
    for k in convs:
        for r in sorted(by[k], key=lambda x: x["turn"]):
            n += 1
            rows.append({"row": n, "blinded_conversation_id": ids[k], "mode": r["mode"], "turn": r["turn"],
                         "user_prompt": r["user_prompt"], "model_output": r["output"],
                         VIOLATION_COL: "", "open_code": ""})
    return rows


def build_tables(run_names):
    """Returns a list of tables: (tab_name, destination, columns, rows)."""
    test = "test" in run_names
    pre = "TEST_" if test else ""
    tables, key_rows = [], []
    for run in run_names:
        _, log_file, _ = log_paths(run)
        recs = read_log(log_file)
        if not recs:
            continue
        ids = blinded_ids(recs, run)
        for r in recs:
            r["blinded_conversation_id"] = ids[conv_key(r)]
        recs.sort(key=lambda r: (MODES.index(r["mode"]), r["variant"], r["conversation"], r["turn"]))
        label = "gemini31" if run == "test" else run
        tables.append((pre + "raw_" + label, "data", RAW_COLS, recs))
        for k, cid in ids.items():
            key_rows.append({"blinded_conversation_id": cid, "model": recs[0]["model_requested"],
                             "mode": k[0], "variant": k[1], "conversation": k[2]})
        tables.append((pre + "rate_" + label, "rater1", RATE_COLS, rating_rows(recs, ids, run + "-r1")))
        if run in ("gemini31", "test"):
            sub = [r for r in recs if r["conversation"] == 1]
            tables.append((pre + "rate_conversation1", "rater2", RATE_COLS, rating_rows(sub, ids, run + "-r2")))
    if key_rows:
        tables.append((pre + "blinding_key", "data", KEY_COLS, key_rows))
    return tables


def build_sheets(run_names=("gemini31", "gemini38"), upload=True):
    test = "test" in run_names
    sheets_dir = os.path.join(OUT_DIR, "test_sheets" if test else "sheets")
    os.makedirs(sheets_dir, exist_ok=True)
    tables = build_tables(run_names)
    say("\nCSV backups saved in %s:" % os.path.relpath(sheets_dir, HERE))
    for name, dest, cols, rows in tables:
        fname = ("BLINDING_KEY_do_not_share_with_raters" if name.endswith("blinding_key") else name) + ".csv"
        write_csv(os.path.join(sheets_dir, fname), cols, rows)
        say("  %s (%d rows)" % (fname, len(rows)))
    if upload and tables:
        upload_to_google_sheets(tables, test)


# ---------------------------------------------------------------- Google Sheets upload
SHEET_ENV = {"data": "DATA_SHEET_ID", "rater1": "RATER1_SHEET_ID", "rater2": "RATER2_SHEET_ID"}


def _load_service_account():
    path = os.path.expanduser(os.environ.get("GCP_SECRETS_TOML",
                                             ".streamlit/secrets.toml"))
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # Python < 3.11
    with open(path, "rb") as fh:
        return dict(tomllib.load(fh)["gcp_service_account"])


def _with_retry(fn, what):
    delay = 5
    for attempt in range(6):
        try:
            return fn()
        except Exception as e:  # gspread.exceptions.APIError on quota etc.
            msg = str(e)
            if attempt < 5 and any(s in msg for s in ("429", "500", "503", "Quota", "quota", "RATE_LIMIT")):
                say("   ...Google Sheets busy (%s), retrying in %ds" % (what, delay))
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise


def _has_ratings(ws):
    vals = ws.get_all_values()
    if not vals:
        return False
    head = vals[0]
    cols = [i for i, h in enumerate(head) if h in (VIOLATION_COL, "open_code")]
    for row in vals[1:]:
        for i in cols:
            if i < len(row) and row[i].strip() and row[i].strip().upper() != "FALSE":
                return True
    return False


def _format_tab(sh, ws, cols, rows):
    sid = ws.id
    reqs = [
        {"updateSheetProperties": {"properties": {"sheetId": sid, "gridProperties": {"frozenRowCount": 1}},
                                   "fields": "gridProperties.frozenRowCount"}},
        {"repeatCell": {"range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 1},
                        "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
                        "fields": "userEnteredFormat.textFormat.bold"}},
        {"repeatCell": {"range": {"sheetId": sid, "startRowIndex": 1},
                        "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"}},
                        "fields": "userEnteredFormat.wrapStrategy,userEnteredFormat.verticalAlignment"}},
    ]
    wide = {"user_prompt": 380, "output": 480, "model_output": 480, "open_code": 220}
    for i, c in enumerate(cols):
        if c in wide:
            reqs.append({"updateDimensionProperties": {
                "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
                "properties": {"pixelSize": wide[c]}, "fields": "pixelSize"}})
    if VIOLATION_COL in cols:
        vi = cols.index(VIOLATION_COL)
        reqs.append({"setDataValidation": {
            "range": {"sheetId": sid, "startRowIndex": 1, "endRowIndex": len(rows) + 1,
                      "startColumnIndex": vi, "endColumnIndex": vi + 1},
            "rule": {"condition": {"type": "BOOLEAN"}, "strict": True}}})
        # shade every other conversation so each conversation reads as one block
        start, shade, prev = 1, False, None
        for n, r in enumerate(rows + [None], start=1):
            cid = r["blinded_conversation_id"] if r else None
            if prev is not None and cid != prev:
                if shade:
                    reqs.append({"repeatCell": {
                        "range": {"sheetId": sid, "startRowIndex": start, "endRowIndex": n},
                        "cell": {"userEnteredFormat": {"backgroundColor": {"red": 0.93, "green": 0.95, "blue": 0.98}}},
                        "fields": "userEnteredFormat.backgroundColor"}})
                shade, start = not shade, n
            prev = cid
    _with_retry(lambda: sh.batch_update({"requests": reqs}), "formatting " + ws.title)


def upload_to_google_sheets(tables, test):
    try:
        import gspread
    except ImportError:
        say("\nGoogle Sheets upload skipped: the 'gspread' package isn't available in this Python.\n"
            "Install it first:  pip install -r requirements.txt\n"
            "then run:  python run_study.py upload")
        return
    missing = [v for v in SHEET_ENV.values() if not os.environ.get(v)]
    if missing:
        say("\nGoogle Sheets upload skipped: add %s to .env." % ", ".join(missing))
        return
    try:
        gc = gspread.service_account_from_dict(_load_service_account())
    except Exception as e:
        say("\nGoogle Sheets upload skipped: could not load the service account (%s). "
            "Set GCP_SECRETS_TOML in .env to your secrets.toml or key .json file." % e)
        return
    say("\nUploading to Google Sheets...")
    books = {}
    for name, dest, cols, rows in tables:
        sheet_id = os.environ[SHEET_ENV[dest]]
        if sheet_id not in books:
            books[sheet_id] = _with_retry(lambda s=sheet_id: gc.open_by_key(s), "opening sheet")
        sh = books[sheet_id]
        existing = {w.title: w for w in _with_retry(sh.worksheets, "listing tabs")}
        ws = existing.get(name)
        if ws is not None and name.startswith("rate") and _has_ratings(ws):
            say("  SKIPPED %s: it already has ratings in it, so it was left untouched." % name)
            continue
        if ws is not None:
            _with_retry(lambda w=ws: sh.del_worksheet(w), "replacing " + name)
        ws = _with_retry(lambda: sh.add_worksheet(title=name, rows=len(rows) + 1, cols=len(cols)), "adding " + name)
        values = [cols] + [[("" if r.get(c) is None else r.get(c)) for c in cols] for r in rows]
        for i in range(0, len(values), 300):  # chunked to stay under request size limits
            chunk = values[i:i + 300]
            _with_retry(lambda c=chunk, s=i: ws.update(values=c, range_name="A%d" % (s + 1),
                                                       value_input_option="RAW"), "writing " + name)
        _format_tab(sh, ws, cols, rows)
        if dest == "data":  # keep raw data + blinding key out of sight while rating (View > Hidden sheets)
            _with_retry(lambda w=ws: sh.batch_update({"requests": [{"updateSheetProperties": {
                "properties": {"sheetId": w.id, "hidden": True}, "fields": "hidden"}}]}), "hiding " + name)
        say("  %-22s -> %s (%d rows)%s" % (name, sh.title, len(rows), " [hidden tab]" if dest == "data" else ""))
    for sh in books.values():  # tidy up: drop the empty default tab, and old TEST tabs after a real run
        tabs = _with_retry(sh.worksheets, "listing tabs")
        for w in tabs:
            drop = (w.title in ("Sheet1", "Hoja 1") and len(tabs) > 1 and not any(w.get_all_values())) or \
                   (not test and w.title.startswith("TEST_"))
            if drop:
                _with_retry(lambda w=w: sh.del_worksheet(w), "tidying tabs")
        say("  %s: https://docs.google.com/spreadsheets/d/%s" % (sh.title, sh.id))


# ---------------------------------------------------------------- commands
def cmd_show(square, users, systems):
    for c in CONVERSATIONS:
        say("Conversation %d order: %s" % (c, square[c]))
    for m in MODES:
        say("\n================ MODE %s ================" % m)
        for n in range(1, 6):
            say("\n[User prompt %d]%s\n%s" % (n, " (hard)" if n in HARD_PROMPTS else "", users[m][n]))
        for v in VARIANTS:
            say("\n--- System prompt %s %s ---\n%s" % (m, v, systems[m][v]))


def cmd_check(key):
    settings = resolve_settings(key, force=True)
    say("Endpoint: %s" % settings["endpoint"])
    ok = True
    for run, model in settings["models"].items():
        if not model:
            ok = False
            continue
        try:
            resp, _, lat = call_gemini(key, settings["endpoint"], model, "Answer in one word.",
                                       [{"role": "user", "parts": [{"text": "Say hello."}]}])
            p = parse_response(resp)
            say("%s -> %s (served as %s): %r, finish %s, %.1fs"
                % (run, model, p["model_version"] or "?", p["output"][:60], p["finish_reason"], lat))
        except FatalAPIError as e:
            ok = False
            say("%s -> %s FAILED: %s" % (run, model, e))
    if not ok and settings["endpoint"] == "aistudio":
        say("\nIf the error mentions authentication, try adding  GEMINI_ENDPOINT=vertex  to .env.")
    say("\nAll good, settings saved." if ok else "\nFix the problem above, then run 'check' again.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("check", "show", "test", "main", "sub38", "sheets",
                                                "upload", "all"):
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    square, users, systems = load_prompts()
    if cmd == "show":
        cmd_show(square, users, systems)
        return
    key = load_env()
    if cmd in ("sheets", "upload"):
        build_sheets()
        return
    if cmd == "check":
        cmd_check(key)
        return
    settings = resolve_settings(key)
    if os.environ.get("GEMINI_ENDPOINT"):
        settings["endpoint"] = os.environ["GEMINI_ENDPOINT"]
    if cmd == "test":
        execute(key, settings, "test", "gemini31", jobs_test(), square, users, systems)
        build_sheets(("test",))
        return
    if cmd in ("main", "all"):
        execute(key, settings, "gemini31", "gemini31", jobs_main(), square, users, systems)
    if cmd in ("sub38", "all"):
        execute(key, settings, "gemini38", "gemini38", jobs_sub38(), square, users, systems)
    build_sheets()


if __name__ == "__main__":
    main()
