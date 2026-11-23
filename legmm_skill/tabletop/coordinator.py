"""Read-only instructions plus the executor's separate acknowledgement file."""
import datetime,hashlib,json,re


def acknowledge(root,phase,**fields):
    source=root/'COORDINATOR_UPDATES.md';raw=source.read_bytes();text=raw.decode()
    match=re.search(r'当前有效版本[：:]\s*(\d+)',text)
    if not match:raise ValueError('Cannot identify coordinator instruction version')
    path=root/'coordinator_update_ack.json'
    record=json.loads(path.read_text()) if path.exists() else {}
    digest=hashlib.sha256(raw).hexdigest()
    if record and (record.get('version')!=int(match.group(1)) or record.get('coordinator_updates_sha256')!=digest):
        from .lifecycle_cleanup import SafeStop
        raise SafeStop('New coordinator version requires executor review at this stage boundary')
    record.update(fields);record.update(version=int(match.group(1)),phase=phase,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        coordinator_updates_sha256=digest)
    temporary=root/'coordinator_update_ack.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)
    print(text,flush=True)
    return record
