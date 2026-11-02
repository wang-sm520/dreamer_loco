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


def judge(packet, runner=subprocess.run):
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
      proc = runner(cmd, input=build_prompt(packet), capture_output=True, text=True,
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
