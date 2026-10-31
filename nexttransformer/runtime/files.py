"""Pinned resource identifiers, file hashes and atomic JSON writes."""
import hashlib
import json
from pathlib import Path

DATA_REPO = 'karpathy/climbmix-400b-shuffle'
CORE_SHA256 = '90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200'
DATA_REVISION = '915333b4f8b8684f39aeaafea600fea6f43fb703'
VALIDATION_SHARD = 'shard_06542.parquet'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)
