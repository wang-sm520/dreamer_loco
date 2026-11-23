"""Complete, losslessly compressed audit records for future learning episodes."""
import gzip,hashlib,json,os
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .storage import StorageGuard


def write_physics_audit(folder,audit):
    folder=Path(folder)
    if (folder/'physics_audit.json').exists():raise FileExistsError('Raw physics audit already exists')
    raw=json.dumps(audit,separators=(',',':'),allow_nan=False).encode()
    data=gzip.compress(raw,compresslevel=6,mtime=0)
    if json.loads(gzip.decompress(data))!=audit:raise RuntimeError('Audit serialization roundtrip differs')
    StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop').check(len(data)+4096)
    path=folder/'physics_audit.json.gz'
    with path.open('xb') as file:file.write(data);file.flush();os.fsync(file.fileno())
    record=dict(format='complete_physics_audit_gzip_v1',stored_name=path.name,
        json_bytes=len(raw),json_sha256=hashlib.sha256(raw).hexdigest(),stored_bytes=len(data),
        stored_sha256=hashlib.sha256(data).hexdigest(),no_actor_or_field_omitted=True)
    with (folder/'physics_audit_storage.json').open('x') as file:json.dump(record,file,indent=2);file.write('\n')
    return record


def read_physics_audit(folder):
    folder=Path(folder);raw=folder/'physics_audit.json';packed=folder/'physics_audit.json.gz'
    if raw.exists() and packed.exists():raise ValueError('Ambiguous physics audit formats')
    if raw.exists():return json.loads(raw.read_text())
    data=packed.read_bytes();record=json.loads((folder/'physics_audit_storage.json').read_text())
    if hashlib.sha256(data).hexdigest()!=record['stored_sha256']:raise ValueError('Compressed audit SHA differs')
    expanded=gzip.decompress(data)
    if len(expanded)!=record['json_bytes'] or hashlib.sha256(expanded).hexdigest()!=record['json_sha256']:
        raise ValueError('Expanded audit identity differs')
    return json.loads(expanded)
