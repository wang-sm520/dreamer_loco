import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'tools'))
from legmm_wm_eval import (binary_auroc, depth_mae, ece_equal_width,
                           phase_accuracy, phase_transition_events,
                           reward_event_errors)


def test_auroc_and_ece_numpy_references():
  y = np.array([0, 0, 1, 1])
  p = np.array([0.1, 0.2, 0.8, 0.9])
  auc, pos, neg = binary_auroc(y, p)
  assert auc == 1.0 and (pos, neg) == (2, 2)
  ece, info = ece_equal_width(y, p, bins=2)
  # Bin means: (0.0 vs .15), (1.0 vs .85), each with half the samples.
  np.testing.assert_allclose(ece, 0.15)
  assert info['count'] == 4 and sum(x['count'] for x in info['bins']) == 4


def test_depth_mae_masks_zero_targets():
  pred = np.array([[[1.0], [3.0]]])
  target = np.array([[[0.0], [2.0]]])
  mae, count = depth_mae(pred, target)
  assert mae == 1.0 and count == 1


def test_phase_accuracy_per_class():
  result = phase_accuracy([0, 1, 2, 2], [0, 2, 2, 1])
  assert result['accuracy'] == 0.5
  assert result['per_class']['0']['count'] == 1
  assert result['per_class']['1']['accuracy'] == 0.0


def test_reward_event_offset_indexing():
  reward = np.arange(8, dtype=float)
  predicted = reward + 1
  result = reward_event_errors(predicted, reward, [2, 6], offsets=(-1, 0, 1))
  assert result['-1']['count'] == 2
  assert result['0']['mean'] == 1.0
  assert result['1']['count'] == 2


def test_phase_events_and_offsets_stay_in_sequence():
  phase = np.array([
      [0, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2],
      [0, 1, 1, 2, 2, 0, 0, 0, 1, 1, 2, 2]], np.int32)
  first = np.zeros((2, 12), bool)
  first[:, 0] = True
  first[1, 5] = True
  events = phase_transition_events(phase, first)
  np.testing.assert_array_equal(np.argwhere(events['pick_success']),
                                [[0, 2], [0, 7], [1, 1], [1, 8]])
  np.testing.assert_array_equal(np.argwhere(events['place_success']),
                                [[0, 4], [0, 9], [1, 3], [1, 10]])
  reward = np.arange(24, dtype=float).reshape(2, 12)
  predicted = reward + 1
  result = reward_event_errors(predicted, reward, events['pick_success'],
                               offsets=(-5, 0, 5), is_first=first)
  assert result['0']['count'] == 4
  assert result['-5']['count'] == 1
  assert result['5']['count'] == 1

