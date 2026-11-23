import pytest
from tabletop.gates import fixed_validation_gate


def rows(k):
    return [dict(seed=10000+i,success=i<k) for i in range(256)]


def test_teacher_new_90_boundary_student_unchanged():
    assert fixed_validation_gate(rows(230))['passed'] is False
    decision=fixed_validation_gate(rows(231))
    assert decision['passed'] is True and decision['required_success']==231
    assert fixed_validation_gate(rows(217),teacher=False)['passed'] is False
    assert fixed_validation_gate(rows(218),teacher=False)['passed'] is True


def test_partial_or_duplicate_cannot_pass_gate():
    with pytest.raises(ValueError):fixed_validation_gate(rows(256)[:224])
    duplicate=rows(256);duplicate[-1]['seed']=10000
    with pytest.raises(ValueError):fixed_validation_gate(duplicate)
