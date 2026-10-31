"""Prepare native nanochat tokenizer and resource provenance in persistent storage."""
from nexttransformer.runtime.site import bootstrap_runtime, add_runtime_arguments
if __name__ == '__main__': bootstrap_runtime()

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from filelock import FileLock

from nexttransformer.runtime.files import DATA_REVISION, VALIDATION_SHARD, atomic_json, sha256
from nexttransformer.evaluation.metrics import verify_core
from nexttransformer.models.native.config import UPSTREAM_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_runtime_arguments(parser)
    parser.add_argument('--existing-core-resources', type=Path, help='reuse a verified CORE bundle; otherwise prepare the pinned bundle here')
    args = parser.parse_args()
    if args.data_dir is None: parser.error('--data-dir is required')
    root = Path(__file__).resolve().parents[1]
    args.resources.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.resources/'prepare.lock')):
        from nexttransformer.runtime.resources import source_manifest_path
        manifest_path = source_manifest_path(args.data_dir,args.resources,args.source_manifest)
        official = json.loads(manifest_path.read_text())
        if official['revision'] != DATA_REVISION: raise ValueError('dataset revision mismatch')
        from nexttransformer.runtime.data import data_inventory, inventory_fingerprint
        expected = sorted((f['rfilename'],f['size']) for f in official['files'] if f['rfilename'].endswith('.parquet'))
        if data_inventory(args.data_dir) != expected:
            raise ValueError('dataset files/sizes differ from the complete pinned manifest')
        val_record = next(f for f in official['files'] if f['rfilename']==VALIDATION_SHARD)
        if sha256(args.data_dir/VALIDATION_SHARD) != val_record['lfs']['sha256']:
            raise ValueError('validation shard checksum differs from the Hub manifest')
        core_resources=args.existing_core_resources or args.resources
        if args.existing_core_resources is None:
            from nexttransformer.runtime.resources import prepare_core
            prepare_core(args.resources,args.core_archive,args.core_url)
        verify_core(core_resources)
        for name, target in [('base_data_climbmix', args.data_dir),
                             ('eval_bundle', core_resources/'eval_bundle'),
                             ('core_manifest.json', core_resources/'core_manifest.json')]:
            link = args.resources/name
            if link.exists():
                if link.resolve() != target.resolve(): raise ValueError(f'resource target changed: {link}')
            else: link.symlink_to(target.resolve(), target_is_directory=target.is_dir())
        os.environ['NANOCHAT_BASE_DIR'] = str(args.resources)
        marker = args.resources/'tokenizer_complete.json'
        if not marker.exists():
            # Exactly the upstream defaults: 2B chars, 10k/doc, 32768 vocabulary.
            if (args.resources/'tokenizer').exists():
                raise ValueError('incomplete tokenizer exists; inspect before removing/retraining it')
            subprocess.run([sys.executable, '-m', 'nanochat.tok_train'], cwd=root, check=True)
            atomic_json(marker, {'max_chars':2_000_000_000, 'doc_cap':10_000, 'vocab_size':32768,
                                 'upstream_commit':UPSTREAM_COMMIT})
        from nexttransformer.runtime.data import tokenizer_fingerprint
        from nanochat.tokenizer import RustBPETokenizer
        tok = RustBPETokenizer.from_directory(args.resources/'tokenizer')
        if tok.get_vocab_size() != 32768: raise ValueError('native tokenizer must have 32768 entries')
        pinned=json.loads((root/'configs/native/upstream_sources.json').read_text())
        files=pinned['files']
        from nexttransformer.runtime.resources import verify_upstream_sources
        upstream_sources = verify_upstream_sources(root, files)
        record = {'upstream_commit':UPSTREAM_COMMIT, 'data_revision':DATA_REVISION,
                  'source_manifest_sha256':sha256(manifest_path), 'tokenizer_sha256':tokenizer_fingerprint(args.resources),
                  'data_inventory_sha256':inventory_fingerprint(expected), 'validation_sha256':val_record['lfs']['sha256'],
                  'tokenizer_recipe':json.loads(marker.read_text()), 'packing':'BOS best-fit; buffer=1000; tokenizer_batch=128',
                  'upstream_sources':upstream_sources}
        atomic_json(args.resources/'native_manifest.json', record)
        print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
