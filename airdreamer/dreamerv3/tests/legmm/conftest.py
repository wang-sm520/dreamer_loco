import os
import pathlib
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
