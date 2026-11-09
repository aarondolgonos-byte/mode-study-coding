"""
Side check (not the paper's method): diversity among ALL 25 outputs of each variant,
pooled across the 5 prompts and 5 conversations (300 pairwise cosine distances per variant).

Uses the embeddings diversity.py already saved, so it makes no API calls and costs nothing.
Run diversity.py first.

Usage:
  python diversity_pooled.py              # also adds a diversity_gemini31_pooled tab to the Google Sheet
  python diversity_pooled.py --no-upload
"""
import itertools
import os
import statistics
import sys

import diversity as d

COLS = ["mode", "variant", "format", "strategy", "n_outputs", "n_pairs", "pooled_mean", "pooled_sd"]


def main():
    d.load_env()
    rows = d.read_local()
    d.check(rows)
    cache = d.load_cache()
    missing = [r for r in rows if d.text_key(r["output"]) not in cache]
    if missing:
        sys.exit("%d outputs have no saved embedding. Run  python diversity.py  first." % len(missing))

    out = []
    for m in d.MODES:
        for v in d.VARIANTS:
            outs = [r["output"] for r in rows if r["mode"] == m and r["variant"] == v]
            dists = [d.cosine_distance(cache[d.text_key(a)], cache[d.text_key(b)])
                     for a, b in itertools.combinations(outs, 2)]
            out.append({"mode": m, "variant": v, "format": d.FORMATS[v[:2]], "strategy": d.STRATEGIES[v[3:]],
                        "n_outputs": len(outs), "n_pairs": len(dists),
                        "pooled_mean": round(statistics.mean(dists), 6),
                        "pooled_sd": round(statistics.stdev(dists), 6)})

    path = d.write_csv("diversity_gemini31_pooled.csv", COLS, out)
    print("\nPooled diversity per variant (mean +/- SD of all 300 pairwise distances among its 25 outputs)")
    print("%-5s " % "" + " ".join("%-13s" % v for v in d.VARIANTS))
    for m in d.MODES:
        cells = ["%.3f+/-%.3f" % (r["pooled_mean"], r["pooled_sd"]) for r in out if r["mode"] == m]
        print("%-5s " % m + " ".join("%-13s" % c for c in cells))
    print("\nSaved %s" % path)

    if "--no-upload" not in sys.argv:
        try:
            sh = d.open_sheet()
        except SystemExit as e:
            print("Google Sheets upload skipped: %s" % e)
            return
        print("Uploading to Google Sheets...")
        d.upload(sh, [("diversity_gemini31_pooled", COLS, out)])


if __name__ == "__main__":
    main()
