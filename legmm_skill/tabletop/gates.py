"""Current user-authorised training gates; historical result fields are read-only."""
import math
from .spec import SPLITS

TEACHER_SUCCESS_THRESHOLD=.90
STUDENT_SUCCESS_THRESHOLD=.85


def fixed_validation_gate(episodes, teacher=True):
    if sorted(e['seed'] for e in episodes)!=list(SPLITS['val']):
        raise ValueError('Gate requires all 256 unique fixed validation seeds')
    threshold=TEACHER_SUCCESS_THRESHOLD if teacher else STUDENT_SUCCESS_THRESHOLD
    required=math.ceil(threshold*len(episodes))
    success=sum(bool(e['success']) for e in episodes)
    return dict(threshold=threshold,required_success=required,success=success,
        total=len(episodes),passed=success>=required)
