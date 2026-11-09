"""
Blinded coding website for the mode-adherence study.

Conversations are shown exactly as a chat. Under every model reply the coder
picks Compliant / Violation and can type a short open code. Every rating saves
immediately to the "Mode Study – Coding Results" Google Sheet with full detail
(mode, variant, conversation, turn, prompts, outputs, model info, timings).

Run:
    streamlit run code_app.py
"""

import hashlib
import hmac
import json
import os
import random
import re
import time
from datetime import datetime, timezone

import gspread
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "Gemini 3.1 · all 900 outputs": ("rate_gemini31", "raw_gemini31"),
    "Gemini 3.8 · all 180 outputs": ("rate_gemini38", "raw_gemini38"),
    "Gemini 3.1 · second rater (10% of Conversation 1)": ("rate_conversation1", "raw_gemini31"),
    "Gemini 3.8 · second rater (10% of Conversation 1)": ("rate_gemini38", "raw_gemini38"),
}
RATERS = {
    "Rater 1": ("rater1", ["Gemini 3.1 · all 900 outputs", "Gemini 3.8 · all 180 outputs"]),
    "Rater 2": ("rater2", ["Gemini 3.1 · second rater (10% of Conversation 1)",
                           "Gemini 3.8 · second rater (10% of Conversation 1)"]),
}
# Every user prompt starts with this same sentence. The site hides it to save reading time;
# the Google Sheets and saved ratings keep the full prompt.
SHARED_OPENING = ("I am trying to come up with solutions to help elderly people, especially widows and otherwise "
                  "single individuals, be less lonely, do more enjoyable new activities, and overall have more fun "
                  "time with others.")


def short_prompt(text):
    t = text.strip()
    return t[len(SHARED_OPENING):].strip() if t.startswith(SHARED_OPENING) else t


# Rater 2 (intercoder reliability set): a random 10% of each model's Conversation 1 responses
# (18 of 180 for Gemini 3.1 Flash-lite + 18 of 180 for Gemini 3.8 Flash = 36 responses), drawn with a fixed
# seed so the same set is shown on every load. The full conversation is shown for context; only the sampled
# responses are rated.
R2_FRACTION = 0.10
R2_SEED = 20261108


def rater2_sample(dataset_label, order, convs):
    """Return {conversation id: set of turns} for Rater 2's random sample of this dataset."""
    items = sorted((cid, str(t["turn"])) for cid in order for t in convs[cid])
    k = round(R2_FRACTION * len(items))
    picked = random.Random("%d-%s" % (R2_SEED, dataset_label)).sample(items, k)
    assign = {}
    for cid, turn in picked:
        assign.setdefault(cid, set()).add(turn)
    return assign


ASSIGNED = {}  # cid -> set of turns to rate (filled in main for the current rater)


def to_rate(cid, turns):
    keep = ASSIGNED.get(cid)
    return [t for t in turns if keep is None or str(t["turn"]) in keep]


MODE_NAMES = [("Q", "Question"), ("Seed", "Seed"), ("S(D)", "Divergent Suggestion"),
              ("S(C)", "Convergent Suggestion")]

RESULT_COLS = [
    "rater", "dataset", "rating_order", "blinded_conversation_id", "turn",
    "rating", "violation", "code",
    "mode", "variant", "format", "strategy", "conversation", "prompt_number", "prompt_type", "stage",
    "user_prompt", "model_output",
    "model_requested", "model_version", "finish_reason",
    "prompt_tokens", "output_tokens", "thinking_tokens", "total_tokens",
    "output_generated_at_utc", "generation_latency_s", "response_id", "generation_run_id",
    "first_rated_at_utc", "last_saved_at_utc", "seconds_from_opening_conversation_to_first_rating", "times_saved",
]

CSS = """
<style>
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(59, 130, 246, 0.10);
    border: 1px solid rgba(59, 130, 246, 0.25);
    width: 88% !important;
    margin-left: auto;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(120, 120, 135, 0.08);
    border: 1px solid rgba(120, 120, 135, 0.22);
    width: 88% !important;
    margin-right: auto;
}
[class*="st-key-conv_grid"] button {
    min-height: 92px;
    border-radius: 16px;
    white-space: pre-line;
}
[class*="st-key-conv_grid"] button p, [class*="st-key-conv_grid"] button div {
    white-space: pre-line !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
</style>
"""


# ---------------------------------------------------------------- config + Google access
def load_env():
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def setting(name):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


def service_account_info():
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass
    path = os.path.expanduser(os.environ.get("GCP_SECRETS_TOML",
                                             ".streamlit/secrets.toml"))
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    import tomllib
    with open(path, "rb") as fh:
        return dict(tomllib.load(fh)["gcp_service_account"])


@st.cache_resource
def client():
    return gspread.service_account_from_dict(service_account_info())


def retry(fn):
    delay = 3
    for attempt in range(6):
        try:
            return fn()
        except gspread.exceptions.APIError as e:
            if attempt < 5 and any(s in str(e) for s in ("429", "500", "503", "Quota", "quota")):
                time.sleep(delay)
                delay *= 2
                continue
            raise


def records(ws):
    vals = retry(ws.get_all_values)
    if not vals:
        return []
    head = vals[0]
    return [dict(zip(head, row + [""] * (len(head) - len(row)))) for row in vals[1:]]


@st.cache_data(show_spinner="Loading conversations from Google Sheets...")
def load_dataset(rate_tab, raw_tab):
    sh = retry(lambda: client().open_by_key(setting("DATA_SHEET_ID")))
    rate = records(retry(lambda: sh.worksheet(rate_tab)))
    raw = records(retry(lambda: sh.worksheet(raw_tab)))
    raw_by = {(r["blinded_conversation_id"], str(r["turn"])): r for r in raw}
    convs, order = {}, []
    for r in rate:
        cid = r["blinded_conversation_id"]
        if cid not in convs:
            convs[cid] = []
            order.append(cid)
        convs[cid].append(r)
    for cid in convs:
        convs[cid].sort(key=lambda r: int(r["turn"]))
    return order, convs, raw_by


def results_ws(rater):
    key = "ws_" + rater
    if key in st.session_state:
        return st.session_state[key]
    sh = retry(lambda: client().open_by_key(setting("RESULTS_SHEET_ID")))
    title = "codings_" + rater
    try:
        ws = retry(lambda: sh.worksheet(title))
        vals = retry(ws.get_all_values)
        if vals and vals[0] != RESULT_COLS:
            if any(any(c.strip() for c in row) for row in vals[1:]):
                st.error("The '%s' tab has data with different columns than this version of the site. "
                         "Rename that tab in Google Sheets (e.g. add '_old') and reload." % title)
                st.stop()
            if ws.col_count < len(RESULT_COLS):
                retry(lambda: ws.add_cols(len(RESULT_COLS) - ws.col_count))
            retry(lambda: ws.update(values=[RESULT_COLS], range_name="A1"))
    except gspread.exceptions.WorksheetNotFound:
        ws = retry(lambda: sh.add_worksheet(title=title, rows=1200, cols=len(RESULT_COLS)))
        retry(lambda: ws.update(values=[RESULT_COLS], range_name="A1"))
        retry(lambda: ws.freeze(rows=1))
        try:
            default = sh.worksheet("Sheet1")
            if not any(default.get_all_values()):
                sh.del_worksheet(default)
        except Exception:
            pass
    st.session_state[key] = ws
    return ws


def load_codings(rater):
    """(dataset, cid, turn) -> [sheet row number, record]"""
    out = {}
    for i, r in enumerate(records(results_ws(rater)), start=2):
        out[(r["dataset"], r["blinded_conversation_id"], str(r["turn"]))] = [i, r]
    return out


def col_letter(n):
    s = ""
    while n:
        n, rem = divmod(n - 1, 26)
        s = chr(65 + rem) + s
    return s


def save_rating(rater, dataset_label, order_index, cid, t, raw, rating, code, opened_at):
    """Insert or update one row for one model response."""
    codings = st.session_state.codings
    ws = results_ws(rater)
    turn = str(t["turn"])
    k = (dataset_label, cid, turn)
    row_no, prev = codings.get(k, [None, {}])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    first_rated = prev.get("first_rated_at_utc") or (now if rating else "")
    secs = prev.get("seconds_from_opening_conversation_to_first_rating") or \
        (round(time.time() - opened_at) if rating else "")
    rec = {
        "rater": rater, "dataset": dataset_label, "rating_order": order_index + 1,
        "blinded_conversation_id": cid, "turn": turn,
        "rating": rating or "", "violation": {"Violation": 1, "Compliant": 0}.get(rating, ""),
        "code": (code or "").strip(),
        "mode": raw.get("mode", t.get("mode", "")), "variant": raw.get("variant", ""),
        "format": raw.get("format", ""), "strategy": raw.get("strategy", ""),
        "conversation": raw.get("conversation", ""), "prompt_number": raw.get("prompt_number", ""),
        "prompt_type": raw.get("prompt_type", ""), "stage": raw.get("stage", ""),
        "user_prompt": t["user_prompt"], "model_output": t["model_output"],
        "model_requested": raw.get("model_requested", ""), "model_version": raw.get("model_version", ""),
        "finish_reason": raw.get("finish_reason", ""),
        "prompt_tokens": raw.get("prompt_tokens", ""), "output_tokens": raw.get("output_tokens", ""),
        "thinking_tokens": raw.get("thinking_tokens", ""), "total_tokens": raw.get("total_tokens", ""),
        "output_generated_at_utc": raw.get("timestamp_utc", ""), "generation_latency_s": raw.get("latency_s", ""),
        "response_id": raw.get("response_id", ""), "generation_run_id": raw.get("run_id", ""),
        "first_rated_at_utc": first_rated, "last_saved_at_utc": now,
        "seconds_from_opening_conversation_to_first_rating": secs,
        "times_saved": int(prev.get("times_saved") or 0) + 1,
    }
    row = [rec[c] for c in RESULT_COLS]
    if row_no:
        rng = "A%d:%s%d" % (row_no, col_letter(len(RESULT_COLS)), row_no)
        retry(lambda: ws.update(values=[row], range_name=rng, value_input_option="RAW"))
    else:
        resp = retry(lambda: ws.append_rows([row], value_input_option="RAW"))
        try:
            rng = resp["updates"]["updatedRange"]
            row_no = int(re.search(r"![A-Z]+(\d+)", rng).group(1))
        except Exception:
            st.session_state.codings = load_codings(rater)
            return
    codings[k] = [row_no, {c: str(v) for c, v in rec.items()}]


# ---------------------------------------------------------------- UI helpers
def okey(*parts):
    """Opaque widget key so IDs never appear in the page HTML."""
    return "k" + hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]


def show(text):
    return (text or "").replace("$", "\\$")


@st.dialog("Full Rubric", width="large")
def rubric_dialog():
    with open(os.path.join(HERE, "rubric.md"), encoding="utf-8") as fh:
        st.markdown(show(fh.read()).replace("\n", "  \n"))


def rubric_button(key):
    if st.button("📋 Open Full Rubric", key=key):
        rubric_dialog()


def flash(kind, msg):
    st.session_state["flash_" + kind] = msg


def show_flash():
    for kind in ("error", "warning", "success"):
        msg = st.session_state.pop("flash_" + kind, None)
        if msg:
            getattr(st, kind)(msg)


# ---------------------------------------------------------------- pages
def conv_filter(ctx, conv_of):
    """Which real conversation (1-5, from the Latin square) to show. Variants stay hidden."""
    nums = sorted({v for v in conv_of.values() if v}, key=int)
    options = ["Conversation %s" % n for n in nums] + (["All"] if len(nums) > 1 else [])
    key = okey("convfilter", ctx)  # kept outside the widget so it survives opening a conversation
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    if len(options) > 1:
        wkey = key + "_widget"

        def keep():
            st.session_state[key] = st.session_state[wkey]
        st.radio("Show", options, horizontal=True, key=wkey, index=options.index(st.session_state[key]),
                 on_change=keep)
    choice = st.session_state[key]
    return None if choice == "All" else choice.split()[-1]


def visible(order, conv_of, which, convs=None, num_of=None):
    ids = [c for c in order if which is None or conv_of.get(c) == which]
    if convs is not None:  # same order as the cards: conversation, then mode tab, then card number
        modes = [m for m, _ in MODE_NAMES]
        ids.sort(key=lambda c: (int(conv_of[c]) if conv_of[c].isdigit() else 99,
                                modes.index(convs[c][0]["mode"]) if convs[c][0]["mode"] in modes else 9, num_of[c]))
    return ids


def simple_list(ctx, order, convs, conv_of, num_of):
    """Rater 2: one plain list of her conversations, no tabs or filters."""
    rater, dataset_label = ctx
    codings = st.session_state.codings
    seq = visible(order, conv_of, None, convs, num_of)
    rated = {k for k, (_, r) in codings.items() if k[0] == dataset_label and r.get("rating")}
    total = sum(len(to_rate(c, convs[c])) for c in seq)
    done = sum(1 for c in seq for t in to_rate(c, convs[c]) if (dataset_label, c, str(t["turn"])) in rated)
    st.metric("Responses rated", "%d / %d" % (done, total))
    st.progress(done / max(1, total))
    if done == total:
        st.success("All done. Thank you!")
    st.markdown("Open each conversation, read it, and mark every response **Compliant** or **Violation**. "
                "No notes needed. Ratings save automatically.")
    rubric_button(okey("rubric_overview", ctx))
    st.write("")
    for i, cid in enumerate(seq, start=1):
        need = to_rate(cid, convs[cid])
        k = sum(1 for t in need if (dataset_label, cid, str(t["turn"])) in rated)
        mode = dict(MODE_NAMES).get(convs[cid][0]["mode"], convs[cid][0]["mode"])
        extra = "" if len(need) == len(convs[cid]) else " (response%s %s only)" % (
            "s" if len(need) > 1 else "", ", ".join(str(t["turn"]) for t in need))
        label = "%d. %s%s · %d/%d rated%s" % (i, mode, extra, k, len(need), "  ✓" if k == len(need) else "")
        if st.button(label, key=okey("open", ctx, cid), width="stretch", type="secondary"):
            st.session_state.open_cid = cid
            st.session_state.opened_at = time.time()
            st.rerun()


def overview(ctx, order, convs, conv_of, num_of):
    rater, dataset_label = ctx
    codings = st.session_state.codings
    rated = {k for k, (_, r) in codings.items() if k[0] == dataset_label and r.get("rating")}
    rated = {k for k in rated if k[1] in order and any(str(t["turn"]) == k[2] for t in to_rate(k[1], convs[k[1]]))}
    total = sum(len(to_rate(c, convs[c])) for c in order)
    done_convs = sum(1 for cid in order if all((dataset_label, cid, str(t["turn"])) in rated
                                               for t in to_rate(cid, convs[cid])))
    c1, c2 = st.columns(2)
    c1.metric("Responses rated (all)", "%d / %d" % (len(rated), total))
    c2.metric("Conversations complete (all)", "%d / %d" % (done_convs, len(order)))
    st.progress(len(rated) / max(1, total))
    rubric_button(okey("rubric_overview", ctx))

    which = conv_filter(ctx, conv_of)
    shown = visible(order, conv_of, which)
    if which is not None:
        k = sum(1 for cid in shown for t in to_rate(cid, convs[cid]) if (dataset_label, cid, str(t["turn"])) in rated)
        n = sum(len(to_rate(cid, convs[cid])) for cid in shown)
        st.caption("Conversation %s: %d / %d responses rated" % (which, k, n))

    present = [(code, name) for code, name in MODE_NAMES if any(convs[c][0]["mode"] == code for c in shown)]
    tabs = st.tabs([name for _, name in present])
    for tab, (mcode, _) in zip(tabs, present):
        with tab:
            groups = [which] if which is not None else sorted({conv_of[c] for c in shown}, key=int)
            for g in groups:
                ids = [c for c in shown if convs[c][0]["mode"] == mcode and conv_of.get(c) == g]
                if which is None:
                    st.markdown("**Conversation %s**" % g)
                with st.container(key="conv_grid_%s_%s" % (mcode.replace("(", "").replace(")", ""), g)):
                    for i in range(0, len(ids), 3):
                        cols = st.columns(3)
                        for col, cid in zip(cols, ids[i:i + 3]):
                            need = to_rate(cid, convs[cid])
                            k = sum(1 for t in need if (dataset_label, cid, str(t["turn"])) in rated)
                            label = "Conversation %s · #%d\n%d/%d rated%s" % (
                                g, num_of[cid], k, len(need), " · ✓" if k == len(need) else "")
                            if col.button(label, key=okey("open", ctx, cid), width="stretch"):
                                st.session_state.open_cid = cid
                                st.session_state.opened_at = time.time()
                                st.rerun()


def on_change(ctx, order_index, cid, t, raw, rkey, ckey):
    rater, dataset_label = ctx
    rating = st.session_state.get(rkey)
    code = st.session_state.get(ckey, "")
    try:
        save_rating(rater, dataset_label, order_index, cid, t, raw, rating, code,
                    st.session_state.get("opened_at", time.time()))
    except Exception as e:
        flash("error", "NOT saved (%s). Check your internet and try again." % e)


def conversation(ctx, order, convs, raw_by, cid, conv_of, num_of):
    rater, dataset_label = ctx
    codings = st.session_state.codings
    idx = order.index(cid)
    turns = convs[cid]
    mode = dict(MODE_NAMES).get(turns[0]["mode"], turns[0]["mode"])

    if st.button("← Back to list", key=okey("back", ctx)):
        st.session_state.pop("open_cid", None)
        st.rerun()
    if rater == "rater2":
        seq2 = visible(order, conv_of, None, convs, num_of)
        st.markdown("### %d of %d · %s" % (seq2.index(cid) + 1, len(seq2), mode))
    else:
        st.markdown("### %s · Conversation %s · #%d" % (mode, conv_of.get(cid, "?"), num_of[cid]))
    rubric_button(okey("rubric", ctx, cid))
    need = {str(t["turn"]) for t in to_rate(cid, turns)}
    rated = sum(1 for t in turns if str(t["turn"]) in need
                and codings.get((dataset_label, cid, str(t["turn"])), [0, {}])[1].get("rating"))
    st.caption("%d of %d responses rated in this conversation. Ratings save automatically." % (rated, len(need)))
    st.caption("Every prompt starts with: \u201c%s\u201d (hidden below to save reading time)." % SHARED_OPENING)

    for n, t in enumerate(turns, start=1):
        turn = str(t["turn"])
        prev = codings.get((dataset_label, cid, turn), [None, {}])[1]
        raw = raw_by.get((cid, turn), {})
        with st.chat_message("user"):
            st.markdown(show(short_prompt(t["user_prompt"])))
        with st.chat_message("assistant"):
            if t["model_output"].strip():
                st.markdown(show(t["model_output"]))
            else:
                st.markdown("*[empty response (no text returned)]*")
        if turn not in need:
            st.caption("Response %d: not in your set. Read it for context only." % n)
            st.write("")
            continue
        rkey, ckey = okey("rate", ctx, cid, turn), okey("code", ctx, cid, turn)
        current = prev.get("rating") or None
        st.radio("Rating for response %d" % n, ["Compliant", "Violation"],
                 index=None if current is None else ["Compliant", "Violation"].index(current),
                 horizontal=True, key=rkey,
                 on_change=on_change, args=(ctx, idx, cid, t, raw, rkey, ckey))
        if rater != "rater2":
            st.text_input("Code for response %d" % n, value=prev.get("code", ""), key=ckey,
                          placeholder="Short open code (press Enter to save)",
                          on_change=on_change, args=(ctx, idx, cid, t, raw, rkey, ckey))
        st.write("")

    if rated == len(need):
        st.success("All responses in this conversation are rated.")
    if rater == "rater2":
        seq = visible(order, conv_of, None, convs, num_of)
        j = seq.index(cid)
        b1, b2 = st.columns(2)
        if b1.button("← Back to list", width="stretch", key=okey("back2", ctx)):
            st.session_state.pop("open_cid", None)
            st.rerun()
        if j < len(seq) - 1 and b2.button("Next conversation →", type="primary", width="stretch",
                                          key=okey("next", ctx)):
            st.session_state.open_cid = seq[j + 1]
            st.session_state.opened_at = time.time()
            st.rerun()
        return
    fkey = okey("convfilter", ctx)
    which = (st.session_state.get(fkey) or "All").split()[-1]
    seq = visible(order, conv_of, None if which == "All" else which, convs, num_of)
    j = seq.index(cid) if cid in seq else 0
    b1, b2 = st.columns(2)
    if b1.button("← Previous conversation", disabled=j == 0, width="stretch", key=okey("prev", ctx)):
        st.session_state.open_cid = seq[j - 1]
        st.session_state.opened_at = time.time()
        st.rerun()
    if b2.button("Next conversation →", type="primary", disabled=j >= len(seq) - 1, width="stretch",
                 key=okey("next", ctx)):
        st.session_state.open_cid = seq[j + 1]
        st.session_state.opened_at = time.time()
        st.rerun()


def login(passwords):
    st.title("Mode Adherence Study")
    st.caption("Please sign in.")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in", type="primary")
    if ok:
        fails = st.session_state.get("login_fails", 0)
        if fails >= 5:
            st.error("Too many failed attempts. Close this tab and try again later.")
            return
        for label, pw in passwords.items():
            if pw and hmac.compare_digest(password, pw):
                st.session_state.clear()
                st.session_state.signed_in_as = label
                st.rerun()
        st.session_state.login_fails = fails + 1
        st.error("Incorrect password.")


def main():
    st.set_page_config(page_title="Mode Adherence Study", page_icon="💬", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    load_env()
    for need in ("DATA_SHEET_ID", "RESULTS_SHEET_ID"):
        if not setting(need):
            st.error("Missing %s in .env" % need)
            st.stop()

    passwords = {"Rater 1": setting("RATER1_PASSWORD"), "Rater 2": setting("RATER2_PASSWORD")}
    if any(passwords.values()):  # online: sign in decides who is coding
        rater_label = st.session_state.get("signed_in_as")
        if rater_label not in RATERS:
            login(passwords)
            return
    with st.sidebar:
        if any(passwords.values()):
            st.write("Signed in as **%s**" % rater_label)
            if st.button("Sign out"):
                st.session_state.clear()
                st.rerun()
        else:
            rater_label = st.selectbox("Who is coding?", list(RATERS))
        rater, allowed = RATERS[rater_label]
        if len(allowed) > 1:
            dataset_label = st.selectbox("Which outputs?", allowed)
        else:
            dataset_label = allowed[0]
        if st.button("Reload from Google Sheets"):
            st.cache_data.clear()
            st.session_state.pop("codings", None)

    ctx = (rater, dataset_label)
    if st.session_state.get("ctx") != ctx:
        st.session_state.ctx = ctx
        st.session_state.pop("codings", None)
        st.session_state.pop("open_cid", None)
    order, convs, raw_by = load_dataset(*DATASETS[dataset_label])
    ASSIGNED.clear()
    if rater == "rater2":
        assign = rater2_sample(dataset_label, order, convs)
        order = [c for c in order if c in assign]
        ASSIGNED.update(assign)
    conv_of = {cid: str(raw_by.get((cid, str(convs[cid][0]["turn"])), {}).get("conversation", "")) for cid in order}
    order = sorted(order, key=lambda c: int(conv_of[c]) if conv_of[c].isdigit() else 99)  # stable: keeps shuffle
    num_of, seen = {}, {}
    for cid in order:
        g = (conv_of[cid], convs[cid][0]["mode"])
        seen[g] = seen.get(g, 0) + 1
        num_of[cid] = seen[g]
    if "codings" not in st.session_state:
        st.session_state.codings = load_codings(rater)

    st.title("Evaluation")
    show_flash()
    cid = st.session_state.get("open_cid")
    if cid in convs:
        conversation(ctx, order, convs, raw_by, cid, conv_of, num_of)
    elif rater == "rater2":
        simple_list(ctx, order, convs, conv_of, num_of)
    else:
        overview(ctx, order, convs, conv_of, num_of)


main()
