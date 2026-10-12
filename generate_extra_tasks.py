"""
generate_extra_tasks.py
=======================
Two EXTRA tasks beyond the role/IOI/GT inventory, each
in three prompt variants, written to a SEPARATE pool root so the main
template pool's fingerprint (and every finished model's coverage gate)
stays untouched:

    data/variants_extra/coloredobjects/v{1,2,3}/prompts_raw.csv
    data/variants_extra/sva/v{1,2,3}/prompts_raw.csv

Colored Objects (attribute binding): three coloured objects in context, the
prompt queries the first object's colour. Corruption swaps the colours of
the first two objects, so the corrupted context licenses the foil colour.
Token parity holds by construction (a colour-word permutation).

SVA (subject-verb agreement): plural subject with a singular attractor,
prompt ends right before the verb; target "are", foil "is" (+"were" as the
second foil). Corruption makes the subject singular. Singular/plural noun
pairs are chosen to be single-token in common vocabularies; the per-model
filter drops any pair whose clean/corrupted token lengths differ.

Schema matches the main pool (filter_variants.py works unchanged):
    clean, corrupted, target_word, incorrect_word, foil_word2,
    target_role, base_id, variant, transform
"""

import csv
import os
import random

OUT_BASE = "./data/variants_extra"

COLORS = ["red", "blue", "green", "black", "white", "brown", "orange",
          "yellow", "purple", "pink", "grey", "silver", "gold"]
OBJECTS = ["pencil", "mug", "notebook", "key", "ball", "book", "cup", "pen",
           "plate", "chair", "box", "lamp", "coin", "sock", "hat", "bag",
           "phone", "clock", "shirt", "bottle"]

CO_FRAMES = {
    "v1": ("color_of_query",
           "On the table, there is {a1} {c1} {o1}, {a2} {c2} {o2}, and "
           "{a3} {c3} {o3}. The color of the {o1} is"),
    "v2": ("copular_query",
           "In the box, you can see {a1} {c1} {o1}, {a2} {c2} {o2}, and "
           "{a3} {c3} {o3}. The {o1} is"),
    "v3": ("question_query",
           "There is {a1} {c1} {o1}, {a2} {c2} {o2}, and {a3} {c3} {o3} "
           "on the desk. What color is the {o1}? It is"),
}

SG_PL = [("key", "keys"), ("door", "doors"), ("cat", "cats"), ("car", "cars"),
         ("book", "books"), ("girl", "girls"), ("boy", "boys"),
         ("tree", "trees"), ("lamp", "lamps"), ("chair", "chairs"),
         ("dog", "dogs"), ("bird", "birds"), ("star", "stars"),
         ("plan", "plans"), ("game", "games"), ("song", "songs"),
         ("coin", "coins"), ("card", "cards")]
ATTR_SURFACE = ["cabinet", "table", "shelf", "desk", "counter", "bench",
                "tray", "floor", "carpet", "balcony", "porch", "wall"]
ATTR_ANIMATE = ["manager", "teacher", "farmer", "doctor", "artist", "guard",
                "lawyer", "nurse", "clerk", "driver", "waiter", "judge"]
SVA_ADJS = ["old", "new", "small", "large", "heavy", "wooden", "broken",
            "shiny"]
SVA_RC_VERBS = ["likes", "owns", "keeps", "watches", "cleans", "paints"]
SVA_LOCS = ["kitchen", "garden", "office", "hall", "yard", "garage"]


def _art(word):
    return "an" if word[0] in "aeiou" else "a"


def gen_colored_objects(rng, n_per_variant=1200):
    rows = {v: [] for v in CO_FRAMES}
    seen = set()
    base = 0
    while min(len(r) for r in rows.values()) < n_per_variant:
        c1, c2, c3 = rng.sample(COLORS, 3)
        o1, o2, o3 = rng.sample(OBJECTS, 3)
        key = (c1, c2, c3, o1, o2, o3)
        if key in seen:
            continue
        seen.add(key)
        fmt = dict(c1=c1, c2=c2, c3=c3, o1=o1, o2=o2, o3=o3,
                   a1=_art(c1), a2=_art(c2), a3=_art(c3))
        # corruption: swap the colours of o1 and o2 (a token permutation)
        fmt_x = dict(fmt, c1=c2, c2=c1, a1=_art(c2), a2=_art(c1))
        for v, (transform, frame) in CO_FRAMES.items():
            if len(rows[v]) >= n_per_variant:
                continue
            rows[v].append({
                "clean": frame.format(**fmt),
                "corrupted": frame.format(**fmt_x),
                "target_word": c1, "incorrect_word": c2, "foil_word2": c3,
                "target_role": "ColoredObjects",
                "base_id": f"colored_{base:05d}", "variant": v,
                "transform": transform,
            })
        base += 1
    return rows


def gen_sva(rng):
    frames = {
        "v1": ("pp_attractor",
               [("The {adj} {subj} on the {attr}",
                 dict(adj=a, attr=t))
                for a in SVA_ADJS for t in ATTR_SURFACE]),
        "v2": ("rc_attractor",
               [("The {subj} that the {attr} {rcv}",
                 dict(attr=t, rcv=r))
                for t in ATTR_ANIMATE for r in SVA_RC_VERBS]),
        "v3": ("double_pp_attractor",
               [("The {subj} near the {attr} in the {loc}",
                 dict(attr=t, loc=l))
                for t in ATTR_SURFACE for l in SVA_LOCS]),
    }
    rows = {v: [] for v in frames}
    for v, (transform, slots) in frames.items():
        base = 0
        for sg, pl in SG_PL:
            for frame, fmt in slots:
                clean = frame.format(subj=pl, **fmt)
                corrupted = frame.format(subj=sg, **fmt)
                rows[v].append({
                    "clean": clean, "corrupted": corrupted,
                    "target_word": "are", "incorrect_word": "is",
                    "foil_word2": "were", "target_role": "SVA",
                    "base_id": f"sva_{v}_{base:05d}", "variant": v,
                    "transform": transform,
                })
                base += 1
    return rows


def write(task, rows):
    cols = ["clean", "corrupted", "target_word", "incorrect_word",
            "foil_word2", "target_role", "base_id", "variant", "transform"]
    for v, rlist in rows.items():
        out_dir = os.path.join(OUT_BASE, task, v)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "prompts_raw.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rlist)
        print(f"  {task}/{v}: {len(rlist)} rows -> {path}")


def main():
    rng = random.Random(93)
    print("Colored Objects:")
    write("coloredobjects", gen_colored_objects(rng))
    print("SVA:")
    write("sva", gen_sva(rng))


if __name__ == "__main__":
    main()
