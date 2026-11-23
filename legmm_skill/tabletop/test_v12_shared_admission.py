import pytest
from .gpu_restore_launch import check_shared_admission

def test_busy_compute_admitted():
    check_shared_admission(dict(used_mib=1579,free_mib=22503),100,34*1024**3,30*1024**3)

@pytest.mark.parametrize('used,free,available',[(4097,22503,34),(1579,18431,34),(1579,22503,29)])
def test_memory_rejected(used,free,available):
    with pytest.raises((RuntimeError,MemoryError)):
        check_shared_admission(dict(used_mib=used,free_mib=free),100,available*1024**3,30*1024**3)
