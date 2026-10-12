import json, re, statistics as st

D = json.load(open('analysis/tables/subsumption_all12.json'))

def sig(cl):
    motifs = set(re.findall(r'has_motif\(C,\s*([a-z_0-9]+)\)', cl))
    preds  = set(re.findall(r'(\w+)\(C', cl))
    nums   = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\b\d+\b', cl)]
    return {'n': len(re.findall(r',', cl)) + 1, 'd': max(nums) if nums else 0,
            'P': preds, 'l': len(cl.split(',')), 'c': len(preds) + len(motifs), 'M': motifs}

def J(a, b):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0

def dd(a, b): return abs(a - b) / max(a, b, 1)

def dist(s1, s2, w):
    return (w[0]*abs(s1['n']-s2['n']) + w[1]*dd(s1['d'], s2['d'])
          + w[2]*(1-J(s1['P'], s2['P'])) + w[3]*dd(s1['l'], s2['l'])
          + w[4]*dd(s1['c'], s2['c']) + w[5]*(1-J(s1['M'], s2['M'])))

sigs = {}
for model, v in D.items():
    for task, cl in (v.get('clauses') or {}).items():
        if isinstance(cl, str):
            sigs[(model, task)] = sig(cl)

SCHEMES = {
    'reported (.10 .10 .20 .10 .10 .40)': [.10, .10, .20, .10, .10, .40],
    'uniform':                          [1/6]*6,
    'no motif term':                    [.20, .20, .40, .10, .10, .00],
    'motif only':                       [0, 0, 0, 0, 0, 1.0],
    'predicate-dominant':               [.10, .10, .50, .10, .10, .10],
}

print(f'signatures: {len(sigs)}\n')
print(f"{'weighting':<34}{'same-task':>10}{'cross-task':>12}{'ratio':>8}")
for name, w in SCHEMES.items():
    same, diff = [], []
    for (m1, t1), s1 in sigs.items():
        for (m2, t2), s2 in sigs.items():
            if (m1, t1) >= (m2, t2): continue
            (same if t1 == t2 else diff).append(dist(s1, s2, w))
    r = st.mean(diff) / st.mean(same) if st.mean(same) > 0 else float('inf')
    print(f'{name:<34}{st.mean(same):>10.3f}{st.mean(diff):>12.3f}{r:>8.2f}')
