"""
filter_variants.py
==================
Coverage gate: validate the generated prompt
variants for ONE model and write pipeline-ready CSVs.

For every (task, variant) under data/variants/:
  1. single-token check: " " + target_word and " " + incorrect_word must each
     encode to exactly one token for this model's tokenizer;
  2. behavioural check (both_correct):
       semantic roles / IOI:  clean:     p(target) > p(foil)
                              corrupted: p(foil)  > p(target)
       Greater-Than:          clean:     top-1 two-digit >= label
                              corrupted: top-1 two-digit <  label
     (the GT foil is the corrupted top-1 digit, as in filter_greater_than.py);
  3. write kept rows with model-specific token ids in the standard pipeline
     schema (clean, corrupted, correct_idx, incorrect_idx, ...).

Outputs:
    data/variants_filtered/{model_short}/{task}_{v}/prompts.csv
    data/variants_filtered/{model_short}/coverage.json

Usage (GPU server):
    python filter_variants.py --model EleutherAI/pythia-1b --device auto
    python filter_variants.py --model meta-llama/Llama-3.2-1B --device cuda
"""

import argparse
import glob
import json
import os

import pandas as pd
import torch

from pool_id import pool_id


def model_short(name: str) -> str:
    return name.split("/")[-1].replace("-", "").replace("_", "")


def single_token_id(tokenizer, word: str):
    """Token id of ' word' if single-token, else None."""
    ids = tokenizer.encode(" " + str(word), add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


@torch.no_grad()
def last_token_logprobs(model, tokenizer, prompts, device, batch_size=16):
    """Log-softmax over the vocab at the last position of each prompt."""
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True)
        tokens = enc["input_ids"].to(device)
        lengths = enc["attention_mask"].sum(dim=1).to(device)
        logits = model(tokens)
        idx = torch.arange(len(batch), device=device)
        last = logits[idx, lengths - 1]
        out.append(torch.log_softmax(last.float(), dim=-1).cpu())
    return torch.cat(out, dim=0)


def two_digit_token_ids(tokenizer):
    mapping = {}
    for i in range(100):
        d = str(i).zfill(2)
        for prefix in [" ", ""]:
            ids = tokenizer.encode(prefix + d, add_special_tokens=False)
            if len(ids) == 1:
                mapping[d] = ids[0]
                break
    return mapping


def filter_contrastive(df, model, tokenizer, device, batch_size,
                       require_argmax=False):
    """
    Semantic roles + IOI: target-vs-strongest-foil win checks.

    Foils per row: incorrect_word plus foil_word2 (from the source foil pool)
    when present; the win criterion uses the strongest foil under the clean
    prompt. Full-vocabulary argmax accuracy is recorded per row and
    aggregated; with require_argmax=True the target must also be the
    top-1 token over the whole vocabulary on the clean prompt.
    """
    stats = {"n_raw": len(df), "n_single_token": 0, "n_both_correct": 0,
             "n_clean_only": 0, "n_argmax_correct": 0}
    tid, foil_ids, foil_words, keep_idx = [], [], [], []
    for i, r in df.iterrows():
        # EAP aligns clean/corrupted positions: token-length parity required
        # (held by construction for semroles; not for e.g. SVA sg/pl swaps)
        if len(tokenizer(str(r["clean"]))["input_ids"]) != \
           len(tokenizer(str(r["corrupted"]))["input_ids"]):
            continue
        t = single_token_id(tokenizer, r["target_word"])
        if t is None:
            continue
        cands = {}
        for col in ("incorrect_word", "foil_word2"):
            w = str(r.get(col, "") or "").strip()
            if not w or w == str(r["target_word"]):
                continue
            f = single_token_id(tokenizer, w)
            if f is not None and f != t:
                cands[f] = w
        if not cands:
            continue
        keep_idx.append(i)
        tid.append(t)
        foil_ids.append(list(cands.keys()))
        foil_words.append(cands)
    stats["n_single_token"] = len(keep_idx)
    if not keep_idx:
        return pd.DataFrame(), stats

    sub = df.loc[keep_idx].reset_index(drop=True)
    lp_clean = last_token_logprobs(model, tokenizer, list(sub["clean"]),
                                   device, batch_size)
    lp_corr = last_token_logprobs(model, tokenizer, list(sub["corrupted"]),
                                  device, batch_size)

    rows = []
    for j, r in sub.iterrows():
        t = tid[j]
        fids = foil_ids[j]
        # strongest foil under the clean prompt
        f = max(fids, key=lambda fi: lp_clean[j, fi].item())
        cpt, cpf = lp_clean[j, t].item(), lp_clean[j, f].item()
        opt, opf = lp_corr[j, t].item(), lp_corr[j, f].item()
        argmax_correct = int(lp_clean[j].argmax()) == t
        if argmax_correct:
            stats["n_argmax_correct"] += 1
        clean_win = cpt > cpf
        corr_win = opf > opt
        if require_argmax:
            clean_win = clean_win and argmax_correct
        if clean_win:
            stats["n_clean_only"] += 1
        if clean_win and corr_win:
            stats["n_both_correct"] += 1
            rows.append({
                "clean": r["clean"], "corrupted": r["corrupted"],
                "correct_idx": t, "incorrect_idx": f,
                "target_word": r["target_word"],
                "incorrect_word": foil_words[j][f],
                "target_role": r["target_role"],
                "base_id": r["base_id"], "variant": r["variant"],
                "transform": r["transform"],
                "clean_p_target": float(torch.tensor(cpt).exp()),
                "clean_p_foil": float(torch.tensor(cpf).exp()),
                "corr_p_target": float(torch.tensor(opt).exp()),
                "corr_p_foil": float(torch.tensor(opf).exp()),
                "clean_win": True, "corr_win": True,
                "clean_argmax_correct": argmax_correct,
            })
    return pd.DataFrame(rows), stats


def filter_greater_than(df, model, tokenizer, device, batch_size):
    """GT: top-1 two-digit >= label on clean, < label on corrupted."""
    stats = {"n_raw": len(df), "n_single_token": 0, "n_both_correct": 0,
             "n_clean_only": 0}
    digit_ids = two_digit_token_ids(tokenizer)
    if len(digit_ids) < 100:
        print(f"    warning: only {len(digit_ids)}/100 two-digit tokens single-token")
    id_list = list(digit_ids.values())
    d_list = list(digit_ids.keys())

    keep = [i for i, r in df.iterrows()
            if str(r["target_word"]).zfill(2) in digit_ids]
    stats["n_single_token"] = len(keep)
    if not keep:
        return pd.DataFrame(), stats
    sub = df.loc[keep].reset_index(drop=True)

    lp_clean = last_token_logprobs(model, tokenizer, list(sub["clean"]),
                                   device, batch_size)
    lp_corr = last_token_logprobs(model, tokenizer, list(sub["corrupted"]),
                                  device, batch_size)

    rows = []
    for j, r in sub.iterrows():
        label = int(str(r["target_word"]))
        clean_digits = lp_clean[j, id_list]
        corr_digits = lp_corr[j, id_list]
        clean_pred = d_list[int(clean_digits.argmax())]
        corr_pred = d_list[int(corr_digits.argmax())]
        clean_ok = int(clean_pred) >= label
        corr_ok = int(corr_pred) < label
        if clean_ok:
            stats["n_clean_only"] += 1
        if clean_ok and corr_ok:
            stats["n_both_correct"] += 1
            label_str = str(label).zfill(2)
            rows.append({
                "clean": r["clean"], "corrupted": r["corrupted"],
                "correct_idx": digit_ids[label_str],
                "incorrect_idx": digit_ids[corr_pred],
                "target_word": label_str, "incorrect_word": corr_pred,
                "target_role": "GreaterThan",
                "base_id": r["base_id"], "variant": r["variant"],
                "transform": r["transform"],
                "clean_p_target": float(lp_clean[j, digit_ids[label_str]].exp()),
                "clean_p_foil": float(lp_clean[j, digit_ids[corr_pred]].exp()),
                "corr_p_target": float(lp_corr[j, digit_ids[label_str]].exp()),
                "corr_p_foil": float(lp_corr[j, digit_ids[corr_pred]].exp()),
                "clean_win": True, "corr_win": True,
                "label": label, "clean_pred": clean_pred,
                "corr_pred": corr_pred,
            })
    return pd.DataFrame(rows), stats


def main():
    parser = argparse.ArgumentParser(description="Filter prompt variants for one model")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--in-base", default="./data/variants")
    parser.add_argument("--out-base", default="./data/variants_filtered")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Debug: cap raw rows per (task, variant)")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--require-argmax", action="store_true",
                        help="Also require the target to be the full-vocabulary "
                             "top-1 token on the clean prompt (strict accuracy)")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")

    import transformer_lens as tl
    from transformers import AutoTokenizer

    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = tl.HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    short = model_short(args.model)
    out_root = os.path.join(args.out_base, short)
    os.makedirs(out_root, exist_ok=True)

    coverage = {}
    for raw_path in sorted(glob.glob(os.path.join(args.in_base, "*", "v*",
                                                  "prompts_raw.csv"))):
        parts = raw_path.split(os.sep)
        task, vkey = parts[-3], parts[-2]
        if args.tasks and task not in [t.lower() for t in args.tasks]:
            continue

        df = pd.read_csv(raw_path)
        if args.max_rows:
            df = df.head(args.max_rows)

        print(f"\n  {task} {vkey}: {len(df)} raw rows")
        if task == "greaterthan":
            kept, stats = filter_greater_than(df, model, tokenizer, device,
                                              args.batch_size)
        else:
            kept, stats = filter_contrastive(df, model, tokenizer, device,
                                             args.batch_size,
                                             require_argmax=args.require_argmax)

        coverage[f"{task}_{vkey}"] = stats
        rate = stats["n_both_correct"] / stats["n_raw"] if stats["n_raw"] else 0
        acc = (stats.get("n_argmax_correct", 0) / stats["n_single_token"]
               if stats["n_single_token"] else 0)
        print(f"    single-token {stats['n_single_token']}, "
              f"argmax-acc {acc:.0%}, clean-only {stats['n_clean_only']}, "
              f"both_correct {stats['n_both_correct']} ({rate:.0%})")

        if len(kept):
            out_dir = os.path.join(out_root, f"{task}_{vkey}")
            os.makedirs(out_dir, exist_ok=True)
            kept.to_csv(os.path.join(out_dir, "prompts.csv"), index=False)

    cov_path = os.path.join(out_root, "coverage.json")
    with open(cov_path, "w") as f:
        json.dump({"model": args.model, "device": device,
                   "pool_id": pool_id(args.in_base),
                   "coverage": coverage}, f, indent=2)
    print(f"\nCoverage report: {cov_path}")


if __name__ == "__main__":
    main()
