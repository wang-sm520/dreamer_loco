"""judge_fn for tools/label_qa.py using the Codex CLI on air-5090 (user choice:
model gpt-6-astra). Each packet is one non-interactive `codex exec` call with
the three PNGs attached, a read-only sandbox and a JSON output schema.

python tools/label_qa.py --stage judge --packets <data>/qa/packets \\
    --judge legmm_judge_codex:judge --verdicts <data>/qa/verdicts.jsonl
"""
import json
import os
import pathlib
import subprocess
import tempfile

from legmm_judge_sonnet import SYSTEM, parse

MODEL = os.environ.get('LEGMM_JUDGE_MODEL', 'gpt-6-astra')
CODEX = os.environ.get('CODEX_BIN', '/home/user/.local/nodejs/bin/codex')
TIMEOUT = int(os.environ.get('LEGMM_JUDGE_TIMEOUT', '300'))

SCHEMA = {
    'type': 'object',
    'properties': {
        'verdict': {'type': 'string', 'enum': ['feasible', 'infeasible', 'uncertain']},
        'confidence': {'type': 'number'},
        'reason': {'type': 'string'},
    },
    'required': ['verdict', 'confidence', 'reason'],
    'additionalProperties': False,
}


def build_prompt(packet):
  state = dict(packet['state'])
  question = state.pop('question', '')
  return (f'{SYSTEM}\n\nThe three attached images are: 1) top-down sketch, 2) head depth, '
          f'3) wrist depth.\nState: {json.dumps(state)}\n'
          'Do not trust aff_rollout or aff_geom blindly; they are the two signals being checked. '
          'Do not run any commands; answer from the images and numbers only.\n'
          f'Question: {question}')


def judge(packet, runner=subprocess.run, prompt_fn=None):
  with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    schema = tmp / 'schema.json'
    schema.write_text(json.dumps(SCHEMA))
    out = tmp / 'answer.txt'
    cmd = [CODEX, 'exec', '-m', MODEL, '-s', 'read-only', '--skip-git-repo-check', '--ephemeral',
           '--color', 'never', '--output-schema', str(schema), '-o', str(out), '-C', str(tmp)]
    for img in packet['images']:
      cmd += ['-i', img]
    cmd.append('-')
    env = dict(os.environ)
    env['PATH'] = str(pathlib.Path(CODEX).parent) + os.pathsep + env.get('PATH', '')
    try:
      proc = runner(cmd, input=(prompt_fn or build_prompt)(packet), capture_output=True, text=True,
                    timeout=TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
      return {'verdict': 'uncertain', 'confidence': 0.0, 'reason': 'codex timeout', 'model': MODEL}
    text = out.read_text() if out.exists() else ''
    if not text:
      return {'verdict': 'uncertain', 'confidence': 0.0, 'model': MODEL,
              'reason': f'codex exit {proc.returncode}: {(proc.stderr or "")[-300:]}'}
    res = parse(text)
    res['model'] = MODEL
    return res


GUIDE_V2 = """How to decide (the numbers under "geometry" are computed exactly from the simulator state):
- The base may only shift by about +-0.15 m and turn by about +-17 degrees. It cannot walk
  around the object or turn towards a target that is beside or behind it.
- The arm is mounted behind the neck and works in front of the robot. A target whose
  bearing is outside about +-60 degrees (after the allowed turn) is NOT reachable.
- reach_ok tells whether some body height puts the wrist within the arm's reach for a
  grasp along a free approach direction (already including the small base shift).
  If reach_ok is false the answer is infeasible.
- free_approach_bearings_deg lists the free approach directions as seen from the robot
  (0 = straight ahead of the robot pointing at it). The gripper must come from one of them,
  so at least one must lie within about +-60 degrees of the direction robot->object.
- Nearby furniture can still block the forearm; when the numbers are borderline answer
  "uncertain" rather than guessing.
Give "feasible" or "infeasible" with confidence >= 0.8 only when these rules clearly decide it."""


def build_prompt_v2(packet):
  state = dict(packet['state'])
  question = state.pop('question', '')
  geometry = state.pop('geometry', {})
  for k in ('aff_rollout', 'aff_geom'):
    state.pop(k, None)
  return (f'{SYSTEM}\n\n{GUIDE_V2}\n\nThe three attached images are: 1) top-down sketch, '
          f'2) head depth, 3) wrist depth.\nState: {json.dumps(state)}\n'
          f'geometry: {json.dumps(geometry)}\n'
          'Do not run any commands; answer from the images and numbers only.\n'
          f'Question: {question}')


def judge_v2(packet, runner=subprocess.run):
  """Same call as judge() with the guided prompt; the two label signals are hidden."""
  return judge(packet, runner, build_prompt_v2)
