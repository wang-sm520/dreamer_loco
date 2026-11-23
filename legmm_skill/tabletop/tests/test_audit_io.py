import json
import pytest
from tabletop.audit_io import write_physics_audit,read_physics_audit


def test_full_audit_roundtrip_both_formats_and_corruption(tmp_path):
    audit={'global':{'min':4,'max':4,'velocity':0},'actors':[{'path':f'/env_{i}/robot/link','iterations':[4,0],'self_collision':True} for i in range(64)]}
    folder=tmp_path/'packed';folder.mkdir();record=write_physics_audit(folder,audit)
    assert read_physics_audit(folder)==audit and record['no_actor_or_field_omitted']
    assert record['stored_bytes']<record['json_bytes']
    packed=folder/'physics_audit.json.gz';packed.write_bytes(packed.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA'):read_physics_audit(folder)
    legacy=tmp_path/'legacy';legacy.mkdir();(legacy/'physics_audit.json').write_text(json.dumps(audit))
    assert read_physics_audit(legacy)==audit
    with pytest.raises(FileExistsError):write_physics_audit(legacy,audit)
