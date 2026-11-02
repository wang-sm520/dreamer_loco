"""judge_fn for tools/label_qa.py that re-checks a label with the stronger
simulator oracle (embodied/envs/legmm/oracle.py) from the packet's state.
Deterministic and reproducible; the default reviewer (see PROGRESS M2).

python tools/label_qa.py --stage judge --packets <data>/qa/packets \\
    --judge legmm_judge_oracle:judge --verdicts <data>/qa/verdicts.jsonl --workers 16
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def judge(packet):
  from legmm_judge_eval import reconstruct
  from embodied.envs.legmm import oracle
  st = packet['state']
  which = 0 if st['which'] == 'pick' else 1
  row = dict(seed=st['seed'], base=st['base_xy_yaw_h'], obj_center=st['object_center'],
             **st.get('reconstruct', {}))
  if which == 1 and 'obj_offset' not in row:
    return {'verdict': 'uncertain', 'confidence': 0.0, 'reason': 'oracle: no in-hand state stored'}
  env = reconstruct(st['split'], row)
  rate, info = oracle.oracle_rate(env, which, seed=int(st['sample_id']) % (2 ** 31))
  verdict = 'feasible' if rate >= 0.5 else 'infeasible'
  conf = abs(rate - 0.5) * 2
  return {'verdict': verdict, 'confidence': conf, 'reason': f'oracle rate {rate:.2f} ({info})',
          'model': 'oracle'}
