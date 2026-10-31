"""Pinned dataset metadata and CORE preparation shared by native entry points."""
import json
from pathlib import Path
import subprocess
import zipfile
from nexttransformer.runtime.files import DATA_REPO, DATA_REVISION, CORE_SHA256, atomic_json, sha256

CORE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def verify_upstream_sources(root, files):
    """Verify original upstream bytes at their current implementation paths.

    Keep the original manifest keys and digests so existing prepared resources
    remain usable. The backbone moved byte-for-byte; its import alias is not
    the implementation being verified. Checkpoint source checks stay separate.
    """
    relocated = {"nanochat/gpt.py": "nexttransformer/models/backbone.py",
                 "scripts/base_eval.py": "nanochat/base_eval.py", "scripts/tok_train.py": "nanochat/tok_train.py"}
    actual = {}
    for name, expected in files.items():
        digest = sha256(Path(root) / relocated.get(name, name))
        if digest != expected:
            raise ValueError(f'pinned upstream implementation changed: {name}')
        actual[name] = digest
    return actual

def prepare_core(resources, archive=None, url=None):
    url = url or CORE_URL
    resources.mkdir(parents=True,exist_ok=True)
    bundle = resources/'eval_bundle'
    lock_path = resources/'core_manifest.json'
    if lock_path.exists():
        pinned = json.loads(lock_path.read_text())
        if pinned['archive_sha256'] != CORE_SHA256:
            raise ValueError('CORE archive differs from the experiment protocol')
        for name, expected in pinned['files'].items():
            if sha256(bundle/name) != expected:
                raise ValueError(f'CORE bundle changed: {name}')
        return pinned
    archive = Path(archive) if archive is not None else resources/'eval_bundle.zip'
    if not archive.exists():
        if archive != resources/'eval_bundle.zip':
            raise FileNotFoundError(f'local CORE archive missing: {archive}')
        temporary = resources/'eval_bundle.zip.incomplete'
        subprocess.run(['curl','-fL','--retry','5','--retry-all-errors','--connect-timeout','30','-C','-',
                        '-o',str(temporary),url],check=True)
        temporary.replace(archive)
    if sha256(archive) != CORE_SHA256:
        raise ValueError('CORE archive SHA-256 mismatch; use the pinned bundle')
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (resources/member.filename).resolve()
            if not target.is_relative_to(resources.resolve()):
                raise ValueError('unsafe CORE archive path')
        z.extractall(resources)
    import yaml
    tasks = yaml.safe_load((bundle/'core.yaml').read_text())['icl_tasks']
    if len(tasks) != 22:
        raise ValueError(f'expected 22 CORE entries, got {len(tasks)}')
    pinned = {'url':url,'archive_sha256':sha256(archive),'entries':len(tasks),
              'files':{str(p.relative_to(bundle)):sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file()}}
    atomic_json(lock_path,pinned)
    return pinned

def source_manifest_path(data_dir, resources, explicit=None):
    """Accept standard hf downloads; cache official metadata for offline reuse."""
    legacy = data_dir/'.downloads'/DATA_REVISION[:12]/'manifest.json'
    path = Path(explicit) if explicit else (legacy if legacy.exists() else resources/'source_manifest.json')
    if not path.exists():
        if explicit:
            raise FileNotFoundError(f'source manifest missing: {path}')
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(DATA_REPO, revision=DATA_REVISION, files_metadata=True)
        if info.sha != DATA_REVISION:
            raise ValueError('unexpected Hub revision')
        files = []
        for item in info.siblings:
            if item.rfilename.endswith('.parquet'):
                if item.lfs is None:
                    raise ValueError(f'missing official SHA-256: {item.rfilename}')
                files.append({'rfilename':item.rfilename,'size':item.size,'lfs':{'sha256':item.lfs.sha256}})
        atomic_json(path, {'repo':DATA_REPO,'revision':DATA_REVISION,'files':files})
    official = json.loads(path.read_text())
    if official['revision'] != DATA_REVISION or official.get('repo',DATA_REPO) != DATA_REPO:
        raise ValueError('unexpected source dataset revision/repository')
    return path
