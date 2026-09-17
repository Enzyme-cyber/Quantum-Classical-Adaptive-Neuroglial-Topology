#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('output_dir')
ap.add_argument('--negative-control-tolerance', type=float, default=0.08,
                help='Allowed |history disambiguation| in glia_equalized / glia_ablated. Increase for low-shot QPU runs.')
args = ap.parse_args()
p = Path(args.output_dir)

m = pd.read_csv(p/'sample2b_state_matching_diagnostics.csv')
s = pd.read_csv(p/'sample2b_seed_summary.csv')

checks = []
def add(name, ok, detail): checks.append((name, bool(ok), detail))
add('neuronal traces exactly matched after reset', np.allclose(m['neuronal_trace_L1_after_reset'],0,atol=1e-12), f"max={m['neuronal_trace_L1_after_reset'].max():.6g}")
add('no dynamic direct N->N candidates', int(m['dynamic_direct_NN_candidate_count'].max())==0, f"max={m['dynamic_direct_NN_candidate_count'].max()}")
add('no base direct N->N edges', int(m['base_direct_NN_edge_count'].max())==0, f"max={m['base_direct_NN_edge_count'].max()}")
add('residual glial separation exists', bool(m['glial_separation_present'].all()), f"pass_fraction={m['glial_separation_present'].mean():.3f}")

for cond in ['glia_equalized','glia_ablated']:
    q=s[s.condition==cond]
    if len(q):
        vmax=float(q['bidirectional_history_disambiguation_index'].abs().max())
        add(f'{cond} negative control near zero', vmax <= args.negative_control_tolerance, f'max_abs={vmax:.4f}')

q=s[s.condition=='intact_glia']
if len(q):
    add('intact glia disambiguation is positive', float(q['bidirectional_history_disambiguation_index'].mean())>0, f"mean={q['bidirectional_history_disambiguation_index'].mean():.4f}")

for name, ok, detail in checks:
    print(('PASS' if ok else 'FAIL').ljust(5), '-', name, '-', detail)
print('\nOVERALL:', 'PASS' if all(x[1] for x in checks) else 'FAIL')
raise SystemExit(0 if all(x[1] for x in checks) else 2)
