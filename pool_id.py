"""
pool_id.py
==========
Content fingerprint of the raw prompt-variant pool (data/variants). Stored in
each model's coverage.json by filter_variants.py and checked by
run_variants_server.sh, so a filter produced from a DIFFERENT pool (e.g. the
scaffold pilot) can never gate away re-filtering. Stdlib only.
"""

import glob
import hashlib
import os


def pool_id(in_base="./data/variants"):
    h = hashlib.md5()
    for path in sorted(glob.glob(os.path.join(in_base, "*", "v*",
                                              "prompts_raw.csv"))):
        h.update(os.path.relpath(path, in_base).encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


if __name__ == "__main__":
    print(pool_id())
