"""judge_fn for tools/label_qa.py backed by Claude Sonnet 5 (user choice, §11.5).

Usage (any machine with API access and ANTHROPIC_API_KEY set):
  pip install anthropic
  python tools/label_qa.py --stage judge --packets <data>/qa/packets \\
      --judge legmm_judge_sonnet:judge --verdicts <data>/qa/verdicts.jsonl

The packet holds three PNGs (top-down sketch, head depth, wrist depth) and the
state JSON; the model answers with JSON {verdict, confidence, reason}.
"""
import base64
import json
import os
import pathlib
import re

MODEL = os.environ.get('LEGMM_JUDGE_MODEL', 'claude-sonnet-5-5')

SYSTEM = """You review training labels for a mobile manipulation robot in a kinematic simulator.
Robot: Unitree A2 quadruped (body 0.82 x 0.44 m, base height 0.08-0.385 m, pitch not controlled)
with an AIRBOT Play arm mounted on the back behind the neck. The shoulder is 0.21 m above the
base origin; upper arm 0.27 m, forearm 0.29 m, wrist-to-gripper 0.24 m (elbow bends one way,
up). Gripper opening 8 cm. A head depth camera looks forward; a wrist depth camera rides on a
stand behind the gripper.
Label meaning: the probability that the robot completes the grasp (or the placement, while
holding the object) from the shown base pose using ONLY small base adjustments
(about +-0.15 m, +-0.3 rad) and the arm, without re-navigating.
Top-down sketch: grey = walls, brown = tables, blue = obstacles, purple = cubby panels,
red dot = object, green square = place zone, short green/red rays = approachable / blocked
approach directions of the object, black rectangle + arrow = robot body and heading.
Depth images: darker = closer, red = invalid (closer than the sensor minimum or no return).
Answer ONLY with a JSON object: {"verdict": "feasible" | "infeasible" | "uncertain",
"confidence": number in [0, 1], "reason": short string}. Use "uncertain" unless the images
and numbers make the answer clear."""


def _image_block(path):
  data = base64.standard_b64encode(pathlib.Path(path).read_bytes()).decode()
  return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}}


def build_messages(packet):
  state = dict(packet['state'])
  question = state.pop('question', '')
  content = [_image_block(p) for p in packet['images']]
  content.append({'type': 'text', 'text': (
      'Images: 1) top-down sketch, 2) head depth, 3) wrist depth.\n'
      f'State: {json.dumps(state)}\n'
      'Do not trust aff_rollout or aff_geom blindly; they are the two signals being checked.\n'
      f'Question: {question}')})
  return [{'role': 'user', 'content': content}]


def parse(text):
  m = re.search(r'\{.*\}', text, re.S)
  if not m:
    return {'verdict': 'uncertain', 'confidence': 0.0, 'reason': f'unparseable: {text[:200]}'}
  try:
    out = json.loads(m.group(0))
  except json.JSONDecodeError:
    return {'verdict': 'uncertain', 'confidence': 0.0, 'reason': f'bad json: {text[:200]}'}
  verdict = out.get('verdict', 'uncertain')
  if verdict not in ('feasible', 'infeasible', 'uncertain'):
    verdict = 'uncertain'
  try:
    conf = float(out.get('confidence', 0.0))
  except (TypeError, ValueError):
    conf = 0.0
  return {'verdict': verdict, 'confidence': min(max(conf, 0.0), 1.0),
          'reason': str(out.get('reason', ''))[:500]}


_CLIENT = None


def judge(packet, client=None):
  global _CLIENT
  if client is None:
    if _CLIENT is None:
      import anthropic
      _CLIENT = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    client = _CLIENT
  resp = client.messages.create(model=MODEL, max_tokens=300, system=SYSTEM,
                                messages=build_messages(packet))
  text = ''.join(getattr(b, 'text', '') for b in resp.content)
  out = parse(text)
  out['model'] = MODEL
  return out
