"""
filter_greater_than.py
======================
Test Greater-Than contrastive pairs on multiple models and store
correctly-predicted examples in the format expected by the COMPASS pipeline.

The Greater-Than task: given "The X lasted from the year 1352 to the year 13",
the model should predict a token >= 52 (the last two digits of the first year).

For the corrupted version, the first year ends in 01 (e.g. 1301), so the
model's prediction should now be unconstrained.

A pair is "both_correct" when:
  - clean:     model's top-1 two-digit prediction >= label
  - corrupted: model's top-1 two-digit prediction < label

Usage:
    python filter_greater_than.py

Outputs (per model):
    data/greater_than_filtered_{model}/both_correct/prompts.csv
"""

import os
import torch
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────

GT_CSV = "./data/greater_than_data.csv"

MODELS = {
    # "pythia14m": "EleutherAI/pythia-14m",  # already done
    # "pythia1b":  "EleutherAI/pythia-1b",   # already done
    "llama1b":   "meta-llama/Llama-3.2-1B",
}

OUTPUT_BASE = "./data"

TWO_DIGIT_STRS = [str(i).zfill(2) for i in range(100)]


def get_two_digit_token_ids(tokenizer):
    """Map two-digit strings to their token IDs."""
    mapping = {}
    for d in TWO_DIGIT_STRS:
        for prefix in [" ", ""]:
            ids = tokenizer.encode(prefix + d, add_special_tokens=False)
            if len(ids) == 1:
                mapping[d] = ids[0]
                break
    return mapping


def predict_top1_digit(model, tokenizer, prompt, digit_token_ids):
    """Return (top1_digit_str, top1_prob, all_digit_probs_dict)."""
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(tokens)  # TransformerLens: returns logits directly
        last_logits = logits[0, -1, :]

    probs = torch.softmax(last_logits, dim=-1)
    digit_probs = {d: probs[tid].item() for d, tid in digit_token_ids.items()}
    top1 = max(digit_probs, key=digit_probs.get)
    return top1, digit_probs[top1], digit_probs


def filter_for_model(model_key, gt_df, digit_token_ids, tokenizer, model):
    """Filter GT pairs for one model. Returns DataFrame."""
    rows = []
    n = len(gt_df)
    n_both = 0

    for idx, row in gt_df.iterrows():
        clean = row["clean"]
        corrupted = row["corrupted"]
        label = int(row["label"])
        label_str = str(label).zfill(2)

        clean_top1, _, clean_probs = predict_top1_digit(
            model, tokenizer, clean, digit_token_ids)
        corr_top1, _, corr_probs = predict_top1_digit(
            model, tokenizer, corrupted, digit_token_ids)

        clean_correct = int(clean_top1) >= label
        corr_correct = int(corr_top1) < label

        if clean_correct and corr_correct:
            n_both += 1
            rows.append({
                "clean": clean,
                "corrupted": corrupted,
                "correct_idx": digit_token_ids.get(label_str),
                "incorrect_idx": digit_token_ids.get(corr_top1),
                "target_word": label_str,
                "incorrect_word": corr_top1,
                "target_role": "GreaterThan",
                "clean_p_target": clean_probs.get(label_str, 0.0),
                "clean_p_foilmax": clean_probs.get(corr_top1, 0.0),
                "corr_p_target": corr_probs.get(label_str, 0.0),
                "corr_p_foilmax": corr_probs.get(corr_top1, 0.0),
                "clean_win": True,
                "corr_win": True,
                "label": label,
                "clean_pred": clean_top1,
                "corr_pred": corr_top1,
            })

        if (idx + 1) % 100 == 0:
            print(f"  [{model_key}] {idx+1}/{n}, {n_both} both_correct")

    print(f"\n  [{model_key}] Total={n}, Both correct={n_both} "
          f"({100*n_both/n:.1f}%)")
    return pd.DataFrame(rows)


def main():
    import transformer_lens as tl

    gt_df = pd.read_csv(GT_CSV)
    print(f"Loaded {len(gt_df)} Greater-Than pairs from {GT_CSV}")

    for model_key, model_name in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name} ({model_key})")
        print(f"{'='*60}")

        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(model_name)
        model = tl.HookedTransformer.from_pretrained(model_name, device="cpu")

        digit_ids = get_two_digit_token_ids(tokenizer)
        print(f"  {len(digit_ids)}/100 two-digit tokens found")

        filtered = filter_for_model(model_key, gt_df, digit_ids, tokenizer, model)

        out_dir = os.path.join(OUTPUT_BASE,
                               f"greater_than_filtered_{model_key}",
                               "both_correct")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "prompts.csv")
        filtered.to_csv(out_path, index=False)
        print(f"  Saved {len(filtered)} pairs to {out_path}")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")
    for mk in MODELS:
        print(f"  data/greater_than_filtered_{mk}/both_correct/prompts.csv")


if __name__ == "__main__":
    main()
