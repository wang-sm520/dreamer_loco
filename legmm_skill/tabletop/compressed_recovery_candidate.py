"""Prepare only: lossless full-state transport under bounded rolling storage."""
import ast,difflib

def recovery_source(source):
    source=source.replace('import copy,hashlib,io,json,os,random,resource,time','import copy,gzip,hashlib,io,json,os,random,resource,time',1)
    source=source.replace('complete_round_boundary=True):','complete_round_boundary=True,compress=False):',1)
    needle='    serialized=memory_snapshot();guard.check(len(data))'
    assert needle in source;source=source.replace(needle,"    uncompressed_bytes=len(data);uncompressed_sha256=hashlib.sha256(data).hexdigest()\n    if compress:data=gzip.compress(data,compresslevel=3,mtime=0)\n"+needle,1)
    needle='record=dict(path=str(target),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),'
    assert needle in source;source=source.replace(needle,needle+"\n        encoding='gzip' if compress else 'torch',uncompressed_bytes=uncompressed_bytes,uncompressed_sha256=uncompressed_sha256,",1)
    needle="        payload=torch.load(file,map_location='cpu',weights_only=False)"
    assert needle in source;source=source.replace(needle,"        signature=file.read(2);file.seek(0)\n        reader=gzip.GzipFile(fileobj=file,mode='rb') if signature==b'\\x1f\\x8b' else file\n        try:payload=torch.load(reader,map_location='cpu',weights_only=False)\n        finally:\n            if reader is not file:reader.close()",1)
    ast.parse(source);return source

def patch(source):
    new=recovery_source(source)
    return ''.join(difflib.unified_diff(source.splitlines(True),new.splitlines(True),fromfile='a/legmm_skill/tabletop/dagger_recovery.py',tofile='b/legmm_skill/tabletop/dagger_recovery.py'))
