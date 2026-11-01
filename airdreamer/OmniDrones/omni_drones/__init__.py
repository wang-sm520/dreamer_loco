# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import sys

import torch
from isaacsim import SimulationApp
from tensordict import TensorDict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")


def init_simulation_app(cfg):
    # launch the simulator
    config = {
        "headless": cfg["headless"],
        "anti_aliasing": 1,
        "multi_gpu": False,
        "max_gpu_count": 1,
    }
    for gpu_key in ("active_gpu", "physics_gpu"):
      if gpu_key in cfg:
        config[gpu_key] = int(cfg[gpu_key])
    active_gpu = int(config.get("active_gpu", 0))
    physics_gpu = int(config.get("physics_gpu", 0))
    for flag in (
        f"--/renderer/multiGpu/enabled=false",
        f"--/renderer/activeGpu={active_gpu}",
        f"--/physics/cudaDevice={physics_gpu}",
    ):
      if flag not in sys.argv:
        sys.argv.append(flag)
    exp_path = os.environ.get("EXP_PATH")
    legacy_experience = f"{exp_path}/omni.isaac.sim.python.kit" if exp_path else None
    if legacy_experience and os.path.isfile(legacy_experience):
      # Standalone (non-pip) Isaac Sim install layout.
      simulation_app = SimulationApp(config, experience=legacy_experience)
    else:
      # pip-installed isaacsim ships its own default experience file under
      # a different name/layout; EXP_PATH (when set, e.g. by Kit itself, or
      # left over from an unrelated install) does not apply here.
      simulation_app = SimulationApp(config)
    return simulation_app


def _get_shapes(self: TensorDict):
    return {
        k: v.shape if isinstance(v, torch.Tensor) else v.shapes for k, v in self.items()
    }


def _get_devices(self: TensorDict):
    return {
        k: v.device if isinstance(v, torch.Tensor) else v.devices
        for k, v in self.items()
    }


TensorDict.shapes = property(_get_shapes)
TensorDict.devices = property(_get_devices)
