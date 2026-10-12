"""
generate_clause_tables.py
=========================
Regenerate the three learned-clause tables in 0A_ILP_primer.tex
(tab:arch-signatures-full / -14m / -llama) directly from the stored
results JSONs, adding the LOO-F1 column.

Every printed number (clause body, confidence, LOO-F1, complexity) comes
from analysis/{model}_multi/results/{task}_s0_results.json — no hand-typed
values. Re-run after any ILP re-run to keep the paper in sync.

Usage:
    python generate_clause_tables.py            # rewrites 0A_ILP_primer.tex
    python generate_clause_tables.py --dry-run  # print tables to stdout
"""

import argparse
import json
import os
import re

TEX_PATH = "paper/From_Functions_to_Logic/0A_ILP_primer.tex"

TASKS = ["ioi", "greaterthan", "beneficiary", "goal", "instrument",
         "location", "path", "source", "time", "topic"]

TASK_LABEL = {"ioi": "IOI", "greaterthan": "GT"}
HEAD_NAME = {"greaterthan": "gt"}

MODELS = {
    "full": ("pythia1b", "Pythia-1B"),
    "14m": ("pythia14m", "Pythia-14M"),
    "llama": ("Llama3.21B", "LLaMA-3.2-1B"),
}

LEGEND = (r"\fcolorbox{motiffg}{motifbg}{\textcolor{motiffg}{\texttt{named "
          r"structural motif}}}; \fcolorbox{spanfg}{spanbg}{\textcolor{spanfg}"
          r"{\texttt{normalised layer position}}}; \fcolorbox{compfg}{compbg}"
          r"{\textcolor{compfg}{\texttt{circuit composition}}}")

METRIC_NOTE = (
    "Confidence = in-sample F1 on the one-vs-rest classification used for "
    "clause selection (3 positive, 27 negative circuits). LOO = leave-one-out "
    "F1: the clause is re-learned with one positive circuit held out and "
    "scored on the held-out circuit and all negatives, pooled over folds "
    "(`--' when fewer than 3 positive circuits are available). "
    r"Complexity = $(n_\ell \times \bar{d})/10$.")

TAIL = {
    "full": (r" The \pmotif{motif} column is the only one that varies across "
             r"tasks; \pspan{layer\_span} and \pcomp{composition} predicates "
             r"are shared structural scaffolding present in every clause."),
    "14m": (" IOI is the only well-discriminated circuit and has two positive "
            "circuits on this model, so its LOO-F1 is undefined; Goal "
            "degenerates to a size-only predicate."),
    "llama": (" Greater-Than has a single circuit on this model, so its "
              "confidence is a consistency check and its LOO-F1 is "
              "undefined."),
}

# Confidence-band banners (Pythia-1B table only, as in the submitted version)
BANNERS = [
    (0.85, "High confidence: single discriminative motif"),
    (0.40, "Intermediate confidence: multi-motif conjunctions"),
    (0.00, "Low confidence: structurally variable binding roles"),
]


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def load_rows(model_dir):
    rows = []
    for t in TASKS:
        path = f"analysis/{model_dir}_multi/results/{t}_s0_results.json"
        if not os.path.isfile(path):
            continue
        l3 = json.load(open(path)).get("layer3") or {}
        clause = (l3.get("tau_arch_text") or "").replace("\n", " ")
        body = clause.split(":-", 1)[1].strip() if ":-" in clause else clause
        sig = l3.get("ilp_signature") or {}
        rows.append({
            "task": t,
            "label": TASK_LABEL.get(t, t.capitalize()),
            "head": HEAD_NAME.get(t, t),
            "motifs": re.findall(r"has_motif\(C, (\w+)\)", body),
            "has_span": "layer_span(" in body,
            "ratio": (m.group(1).rstrip(".")
                      if (m := re.search(r"R > ([\d.]+)", body)) else None),
            "size": (m.group(1) if (m := re.search(r"N < (\d+)", body)) else None),
            "conf": l3.get("tau_arch_confidence"),
            "loo": l3.get("loo_f1"),
            "cmplx": sig.get("complexity"),
        })
    return rows


def _ratio_val(body_ratio):
    return f"{float(body_ratio):.2f}"


def render_row(r, last=False):
    parts = []
    for m in r["motifs"]:
        parts.append(rf"\pmotif{{has\_motif(C,\,{esc(m)})}}")
    if r["has_span"]:
        parts.append(r"\pspan{layer\_span(C,\,$L$,\,$L'$)}")
    if r["ratio"]:
        parts.append(rf"\pcomp{{comp\_ratio(C,\,attn,\,R)}}, R\,>\,{_ratio_val(r['ratio'])}")
    if r["size"]:
        parts.append(rf"\pcomp{{size(C,\,N)}}, N\,<\,{r['size']}")
    body = ",\n    ".join(parts) + "."

    conf = f"{r['conf']:.2f}" if r["conf"] is not None else "--"
    loo = f"{r['loo']:.2f}" if r["loo"] is not None else "--"
    cmplx = f"{r['cmplx']:.1f}" if r["cmplx"] is not None else "--"
    end = r" \\" if last else r" \\[6pt]"
    return (f"{r['label']}\n"
            f"  & \\texttt{{arch\\_{r['head']}(C)~:-}}\n"
            f"    \\newline\\hspace*{{6pt}}{body}\n"
            f"  & {conf} & {loo} & {cmplx}{end}\n")


def render_table(key):
    model_dir, model_name = MODELS[key]
    rows = load_rows(model_dir)
    rows.sort(key=lambda r: -(r["conf"] or 0))

    caption_head = (
        rf"\caption{{\textbf{{Learned architectural signatures ({model_name}, "
        rf"10-task pool).}} Each row is a complete Horn clause; colour "
        rf"identifies predicate type within the conjunction. {LEGEND}. "
        rf"Numeric thresholds (plain text) are learned per task and not "
        rf"colour-coded. {METRIC_NOTE}{TAIL[key]}}}")

    out = ["\\begin{table*}[h]", "\\centering", "\\small", caption_head,
           r"\setlength{\fboxsep}{1.2pt}", r"\setlength{\fboxrule}{0pt}",
           r"\setlength{\tabcolsep}{6pt}",
           r"\begin{tabular}{@{}l p{10.35cm} ccc@{}}", r"\toprule",
           r"\textbf{Task}",
           rf"  & \textbf{{Learned clause $\tau_{{\text{{arch}}}}$ ({model_name}, 10-task pool)}}",
           r"  & \textbf{Conf.} & \textbf{LOO} & \textbf{Cmplx.} \\",
           r"\midrule"]

    use_banners = (key == "full")
    banner_idx = -1
    for i, r in enumerate(rows):
        if use_banners:
            for bi, (thr, text) in enumerate(BANNERS):
                if (r["conf"] or 0) >= thr:
                    if bi != banner_idx:
                        banner_idx = bi
                        out.append(rf"\multicolumn{{5}}{{@{{}}l@{{}}}}"
                                   rf"{{\footnotesize\textit{{{text}}}}} \\[2pt]")
                        out.append("")
                    break
        out.append(render_row(r, last=(i == len(rows) - 1)))

    out += [r"\bottomrule", r"\end{tabular}"]
    if key == "full":
        out += ["", r"\vspace{4pt}", r"\noindent{\small",
                r"\fcolorbox{motiffg}{motifbg}{\textcolor{motiffg}{\texttt{named motif (has\_motif)}}}\;",
                r"\fcolorbox{spanfg}{spanbg}{\textcolor{spanfg}{\texttt{layer position (layer\_span)}}}\;",
                r"\fcolorbox{compfg}{compbg}{\textcolor{compfg}{\texttt{composition (comp\_ratio, size)}}}",
                r"}"]
    out += [rf"\label{{tab:arch-signatures-{key}}}", r"\end{table*}"]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tex = open(TEX_PATH).read()
    for key in MODELS:
        table = render_table(key)
        if args.dry_run:
            print(table)
            print("\n" + "%" * 70 + "\n")
            continue
        # tempered dot: the match must not cross an \end{table*}, so it always
        # starts at the block that actually carries this label
        pattern = re.compile(
            r"\\begin\{table\*\}\[h\](?:(?!\\end\{table\*\}).)*?"
            rf"\\label\{{tab:arch-signatures-{key}\}}\s*\n\\end\{{table\*\}}",
            re.DOTALL)
        if not pattern.search(tex):
            raise SystemExit(f"Could not locate table block for '{key}'")
        # replacement via function: the return value is inserted literally,
        # with no backslash-escape processing
        tex = pattern.sub(lambda _m: table, tex, count=1)

    if not args.dry_run:
        open(TEX_PATH, "w").write(tex)
        print(f"Rewrote 3 tables in {TEX_PATH}")


if __name__ == "__main__":
    main()
