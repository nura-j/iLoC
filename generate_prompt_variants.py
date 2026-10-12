"""
generate_prompt_variants.py
===========================
Generate prompt VARIANTS for every task: different
syntactic framings of the same task, so that circuits can be discovered per
variant and pooled for generalisation testing.

Model-independent (no GPU, no tokenizer): emits raw text pairs. Per-model
validity (single-token targets, clean/corrupted win checks, token IDs) is
established afterwards by filter_variants.py.

Variant design
--------------
Semantic roles (source = union of the existing per-model filtered pools;
the clean prompt must END right before the role filler, so all transforms
change the *scaffold* while keeping the frame target-final):
    v1  temporal adverbial preamble   "Later that day, <s>"
    v2  clausal embedding             "Everyone remembers that <s>"
    v3  clause coordination           "The day had just begun, and <s>"
Corruption is inherited from the source rows (role-preposition swap) with the
same transform applied. Rows whose source corruption is degenerate
(clean == corrupted) are dropped.

IOI (regenerated from name/object/place pools, BABA-style; target = indirect
object B, foil = subject A; corruption replaces the second mention of A with
an out-of-context name C):
    v1  "Afterwards, {A} and {B} went to the {place}. {A} decided to give a {obj} to"
    v2  "When {A} and {B} arrived at the {place}, {A} handed a {obj} to"
    v3  "{A} and {B} spent the day at the {place}. Then {A} passed the {obj} to"

Greater-Than (regenerated; label = last two digits of the start year; the
corrupted start year ends in 01):
    v1  "The {noun} went on from the year {Y} until the year {CC}"
    v2  "The {noun} continued from the year {Y} through the year {CC}"
    v3  "Historical records date the {noun} from the year {Y} to the year {CC}"

Output:
    data/variants/{task_lower}/v{j}/prompts_raw.csv
        columns: clean, corrupted, target_word, incorrect_word,
                 target_role, base_id, variant, transform
"""

import argparse
import itertools
import os
import random

import pandas as pd

SEMROLE_DIRS = [
    "./semantic_role_generation/semroles_v1_filtered_14m",
    "./semantic_role_generation/semroles_v1_filtered_1b",
    "./semantic_role_generation/semroles_v1_Llama_3_2_1B_filtered",
]
ROLES = ["Beneficiary", "Goal", "Instrument", "Location",
         "Path", "Source", "Time", "Topic"]

OUT_BASE = "./data/variants"

# ── Semantic role variants ────────────────────────────────────────────────
#
# Two generation styles:
#   template (default): mine the role lexicon (agents, verbs, themes, role
#     preposition pairs, fillers, foils) from the validated role pools with
#     spaCy, then generate NEW sentences in three distinct syntactic frames:
#       v1 active_new       "The {agent} {verb} [the {theme} ]{prep} {det}"
#       v2 passive/event    "The {theme} was {participle} {prep} {det}"
#                           (themeless roles use an event-subject frame)
#       v3 relative_clause  "The {agent} we saw {verb} [the {theme} ]{prep} {det}"
#     Corruption = same frame with a mined wrong-role preposition; foils are
#     sampled from the mined foil vocabulary (2 per row for the
#     strongest-foil filter).
#   scaffold: the earlier prefix transforms over the pooled base sentences
#     (kept for reproducibility of the pilot run).

SEMROLE_TRANSFORMS = {
    "v1": ("temporal_preamble", "Later that day, {s}"),
    "v2": ("clausal_embedding", "Everyone remembers that {s}"),
    "v3": ("clause_coordination", "The day had just begun, and {s}"),
}

PARTICIPLES = {
    "put": "put", "gave": "given", "sent": "sent", "took": "taken",
    "drew": "drawn", "wrote": "written", "made": "made", "kept": "kept",
    "brought": "brought", "held": "held", "drove": "driven", "ran": "run",
    "sold": "sold", "taught": "taught", "found": "found", "left": "left",
    "built": "built", "threw": "thrown", "spoke": "spoken", "read": "read",
    "sang": "sung", "bought": "bought", "got": "got", "hid": "hidden",
    "laid": "laid", "set": "set", "cut": "cut", "chose": "chosen",
}

EVENT_SUBJECTS = {
    "Time":  ["meeting", "lesson", "ceremony", "rehearsal", "interview",
              "festival", "journey", "celebration", "concert", "match",
              "storm", "parade", "wedding", "protest", "auction", "seminar"],
    "Path":  ["parade", "convoy", "procession", "tour", "expedition",
              "caravan", "march", "race", "pipeline", "trail", "shipment",
              "patrol"],
    "Topic": ["lecture", "meeting", "report", "debate", "article",
              "discussion", "story", "letter", "argument", "film", "song",
              "chapter"],
}

EVENT_VERBS = {
    "Time":  ["took place", "was held", "happened", "started", "began",
              "ended"],
    "Path":  ["passed", "went", "travelled", "moved", "marched"],
    "Topic": ["was", "seemed to be"],
}


def _lcfirst(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def _participle(verb: str):
    if verb in PARTICIPLES:
        return PARTICIPLES[verb]
    if verb.endswith("ed"):
        return verb
    return None


def load_semrole_pool(role: str) -> pd.DataFrame:
    """Union of a role's filtered pools across models, dedup by clean text."""
    frames = []
    for base in SEMROLE_DIRS:
        path = f"{base}/{role}/both_correct/prompts.csv"
        if os.path.isfile(path):
            df = pd.read_csv(path)
            cols = ["clean", "corrupted", "target_word",
                    "incorrect_word", "target_role"]
            if "foil_pool" in df.columns:
                cols.append("foil_pool")
            frames.append(df[cols])
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.drop_duplicates(subset=["clean", "target_word"])
    # degenerate corruptions carry no contrast for EAP
    pool = pool[pool["clean"] != pool["corrupted"]].reset_index(drop=True)
    return pool


def _split_ending(text: str):
    """(prep, det) from a prompt ending '... PREP [the|a|an]'."""
    words = str(text).split()
    if not words:
        return "", ""
    if words[-1].lower() in ("the", "a", "an"):
        return (words[-2] if len(words) >= 2 else ""), words[-1]
    return words[-1], ""


def mine_role_lexicon(role: str, nlp) -> dict:
    """
    Extract the role's validated vocabulary from the pooled validated role sentences:
    agents (subjects), verbs, themes (direct objects), (clean prep, det) /
    (corrupt prep, det) pairs, fillers (targets), and foils.
    """
    pool = load_semrole_pool(role)
    agents, verbs, themes = set(), set(), set()
    prep_pairs = {}
    fillers, foils = set(), set()
    bindings = []

    for _, r in pool.iterrows():
        clean = str(r["clean"])
        prep_c, det_c = _split_ending(clean)
        prep_x, det_x = _split_ending(str(r["corrupted"]))
        if not prep_c or not prep_x or prep_c == prep_x:
            continue
        doc = nlp(clean)
        subj = [t.text for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
        root = [t.text for t in doc if t.dep_ == "ROOT"]
        dobj = [t.text for t in doc
                if t.dep_ == "dobj" and t.text.lower() not in ("a", "the", "an")]
        if not subj or not root:
            continue
        agents.add(subj[0].lower())
        verbs.add(root[0].lower())
        if dobj:
            themes.add(dobj[0].lower())
        key = (prep_c, det_c)
        prep_pairs.setdefault(key, set()).add((prep_x, det_x))
        fillers.add(str(r["target_word"]))
        foils.add(str(r["incorrect_word"]))
        foil2 = (str(r["foil_pool"]).strip()
                 if "foil_pool" in r and str(r.get("foil_pool", "")).strip()
                 else "")
        if foil2:
            foils.add(foil2)
        # the validated contrast skeleton: this preposition pair licenses this
        # target against these foils (e.g. Time: "on"->Friday vs "at"->midnight)
        bindings.append({
            "prep_c": prep_c, "det_c": det_c,
            "prep_x": prep_x, "det_x": det_x,
            "target": str(r["target_word"]),
            "foil": str(r["incorrect_word"]),
            "foil2": foil2,
        })

    # dedup bindings
    seen_b, uniq = set(), []
    for b in bindings:
        k = tuple(b.values())
        if k not in seen_b:
            seen_b.add(k)
            uniq.append(b)

    return {
        "agents": sorted(agents), "verbs": sorted(verbs),
        "themes": sorted(themes),
        "prep_pairs": {k: sorted(v) for k, v in prep_pairs.items()},
        "fillers": sorted(fillers), "foils": sorted(foils - fillers),
        "bindings": uniq,
    }


def _semrole_frames(role: str, lex: dict, rng):
    """Yield (variant_key, frame_name, sentence_builder) for this role."""
    themed = bool(lex["themes"])

    def active(agent, verb, theme, prep, det):
        core = f"The {agent} {verb}"
        if themed and theme:
            core += f" the {theme}"
        return f"{core} {prep} {det}".strip()

    def passive(agent, verb, theme, prep, det):
        part = _participle(verb)
        if themed and theme and part:
            return f"The {theme} was {part} {prep} {det}".strip()
        events = EVENT_SUBJECTS.get(role)
        if events:
            ev = rng.choice(events)
            ev_verb = rng.choice(EVENT_VERBS[role])
            return f"The {ev} {ev_verb} {prep} {det}".strip()
        return None

    def relative(agent, verb, theme, prep, det):
        core = f"The {agent} we saw {verb}"
        if themed and theme:
            core += f" the {theme}"
        return f"{core} {prep} {det}".strip()

    return [("v1", "active_new", active),
            ("v2", "passive_event", passive),
            ("v3", "relative_clause", relative)]


def generate_semrole_variants(role: str, n_rows: int = 2000,
                              seed: int = 42, nlp=None) -> dict:
    """Template-family generation from the mined role lexicon."""
    rng = random.Random(seed + hash(role) % 1000)
    lex = mine_role_lexicon(role, nlp)
    if not lex["agents"] or not lex["fillers"] or not lex["prep_pairs"]:
        print(f"  {role}: lexicon too sparse — skipped")
        return {}

    out = {}
    for vkey, fname, builder in _semrole_frames(role, lex, rng):
        rows, seen = [], set()
        attempts = 0
        while len(rows) < n_rows and attempts < n_rows * 20:
            attempts += 1
            # validated contrast skeleton (preps + target + foils travel
            # together); only the frame lexicon is randomised around it
            b = rng.choice(lex["bindings"])
            agent = rng.choice(lex["agents"])
            verb = rng.choice(lex["verbs"])
            theme = rng.choice(lex["themes"]) if lex["themes"] else ""
            prep_c, det_c = b["prep_c"], b["det_c"]
            prep_x, det_x = b["prep_x"], b["det_x"]
            target, foil, foil2 = b["target"], b["foil"], b["foil2"]

            clean = builder(agent, verb, theme, prep_c, det_c)
            corrupted = builder(agent, verb, theme, prep_x, det_x)
            if clean is None or corrupted is None or clean == corrupted:
                continue
            key = (clean, target)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "clean": clean, "corrupted": corrupted,
                "target_word": target, "incorrect_word": foil,
                "foil_word2": foil2, "target_role": role,
                "base_id": f"{role.lower()}_{len(rows):05d}",
                "variant": vkey, "transform": fname,
            })
        out[vkey] = pd.DataFrame(rows)
    return out


def generate_semrole_scaffold_variants(role: str) -> dict:
    """Earlier style: prefix scaffolds over the pooled base sentences."""
    pool = load_semrole_pool(role)
    out = {}
    for vkey, (name, template) in SEMROLE_TRANSFORMS.items():
        rows = []
        for i, r in pool.iterrows():
            rows.append({
                "clean": template.format(s=_lcfirst(str(r["clean"]))),
                "corrupted": template.format(s=_lcfirst(str(r["corrupted"]))),
                "target_word": r["target_word"],
                "incorrect_word": r["incorrect_word"],
                "foil_word2": r.get("foil_pool", ""),
                "target_role": r["target_role"],
                "base_id": f"{role.lower()}_{i:04d}",
                "variant": vkey, "transform": name,
            })
        out[vkey] = pd.DataFrame(rows)
    return out


# ── IOI variants ──────────────────────────────────────────────────────────

IOI_NAMES = ["John", "Mary", "Tom", "Anna", "Mark", "Sarah", "Paul", "Emma",
             "Peter", "Alice", "James", "Karen", "David", "Laura", "Kevin",
             "Julia", "Robert", "Susan", "Michael", "Linda", "Richard",
             "Nancy", "Daniel", "Helen", "Steven", "Diana", "George",
             "Rachel", "Andrew", "Monica", "Frank", "Teresa"]
IOI_FOIL_NAMES = ["Tiffany", "Brandon", "Whitney", "Marcus", "Crystal",
                  "Trevor"]
IOI_OBJECTS = ["drink", "book", "ring", "ball", "letter", "apple"]
IOI_PLACES = ["house", "park", "store", "office", "school", "garden"]

IOI_TEMPLATES = {
    "v1": ("afterwards_give",
           "Afterwards, {A} and {B} went to the {place}. {A} decided to give a {obj} to"),
    "v2": ("arrival_hand",
           "When {A} and {B} arrived at the {place}, {A} handed a {obj} to"),
    "v3": ("day_pass",
           "{A} and {B} spent the day at the {place}. Then {A} passed the {obj} to"),
}


def generate_ioi_variants(n_rows: int, seed: int) -> dict:
    rng = random.Random(seed)
    combos = [(a, b) for a, b in itertools.permutations(IOI_NAMES, 2)]
    rng.shuffle(combos)
    out = {}
    for vkey, (name, template) in IOI_TEMPLATES.items():
        rows = []
        for i, (a, b) in enumerate(combos[:n_rows]):
            place = rng.choice(IOI_PLACES)
            obj = rng.choice(IOI_OBJECTS)
            c = rng.choice([n for n in IOI_FOIL_NAMES if n not in (a, b)])
            clean = template.format(A=a, B=b, place=place, obj=obj)
            # corrupt the SECOND mention of A only (the sentence-2 subject)
            first_a = clean.find(a)
            second_a = clean.find(a, first_a + len(a))
            corrupted = clean[:second_a] + c + clean[second_a + len(a):]
            rows.append({
                "clean": clean,
                "corrupted": corrupted,
                "target_word": b,       # indirect object
                "incorrect_word": a,    # subject (foil)
                "target_role": "IOI",
                "base_id": f"ioi_{i:04d}",
                "variant": vkey,
                "transform": name,
            })
        out[vkey] = pd.DataFrame(rows)
    return out


# ── Greater-Than variants ─────────────────────────────────────────────────

GT_NOUNS = ["war", "contract", "dynasty", "empire", "famine", "project",
            "marriage", "treaty", "occupation", "renovation", "expedition",
            "monarchy", "rebellion", "drought", "pilgrimage"]

GT_TEMPLATES = {
    "v1": ("went_on_until",
           "The {noun} went on from the year {y} until the year {cc}"),
    "v2": ("continued_through",
           "The {noun} continued from the year {y} through the year {cc}"),
    "v3": ("records_date",
           "Historical records date the {noun} from the year {y} to the year {cc}"),
}


def generate_gt_variants(n_rows: int, seed: int) -> dict:
    rng = random.Random(seed)
    out = {}
    for vkey, (name, template) in GT_TEMPLATES.items():
        rows = []
        for i in range(n_rows):
            century = rng.randint(11, 17)
            yy = rng.randint(2, 98)
            y_clean = century * 100 + yy
            y_corr = century * 100 + 1          # ends in 01
            cc = str(century)
            clean = template.format(noun=rng.choice(GT_NOUNS), y=y_clean, cc=cc)
            # same noun/frame, only the start year changes
            corrupted = clean.replace(str(y_clean), str(y_corr).zfill(4))
            rows.append({
                "clean": clean,
                "corrupted": corrupted,
                "target_word": str(yy).zfill(2),   # label; filter re-derives foil
                "incorrect_word": "",              # decided at filter time (top corr digit)
                "target_role": "GreaterThan",
                "base_id": f"gt_{i:04d}",
                "variant": vkey,
                "transform": name,
            })
        out[vkey] = pd.DataFrame(rows)
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate prompt variants (§1.1a)")
    parser.add_argument("--out", default=OUT_BASE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ioi", type=int, default=992)
    parser.add_argument("--n-gt", type=int, default=960)
    parser.add_argument("--roles", nargs="*", default=None,
                        help="Subset of semantic roles (default: all 8)")
    parser.add_argument("--n-role", type=int, default=2000,
                        help="Rows per role per variant (template style)")
    parser.add_argument("--style", choices=["template", "scaffold"],
                        default="template",
                        help="template: new sentences in 3 syntactic frames "
                             "(default); scaffold: prefix transforms of the "
                             "pooled base sentences (pilot style)")
    args = parser.parse_args()

    nlp = None
    if args.style == "template":
        import spacy
        nlp = spacy.load("en_core_web_sm")

    tasks: dict = {}
    for role in (args.roles or ROLES):
        if args.style == "template":
            tasks[role.lower()] = generate_semrole_variants(
                role, n_rows=args.n_role, seed=args.seed, nlp=nlp)
        else:
            tasks[role.lower()] = generate_semrole_scaffold_variants(role)
    tasks["ioi"] = generate_ioi_variants(args.n_ioi, args.seed)
    tasks["greaterthan"] = generate_gt_variants(args.n_gt, args.seed)

    total = 0
    for task, variants in tasks.items():
        for vkey, df in variants.items():
            if df.empty:
                print(f"  {task}/{vkey}: EMPTY — skipped")
                continue
            out_dir = os.path.join(args.out, task, vkey)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "prompts_raw.csv")
            df.to_csv(path, index=False)
            total += len(df)
            print(f"  {task:<12} {vkey}  {len(df):5} rows -> {path}")

    print(f"\nTotal: {total} raw variant rows under {args.out}/")
    print("Next: filter_variants.py --model <hf_name>  (per model, GPU)")


if __name__ == "__main__":
    main()
