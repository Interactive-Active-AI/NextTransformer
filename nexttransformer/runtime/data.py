"""Upstream BOS-bestfit rows, with exact rank-local checkpoint state.

The algorithm and tie order match nanochat.dataloader. Row generation never
depends on microbatch size; each rank consumes the same rows per global update.
"""
import copy
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from nexttransformer.runtime.files import sha256, VALIDATION_SHARD, DATA_REVISION
from nexttransformer.runtime.resources import verify_upstream_sources


def tokenizer_fingerprint(resources):
    root = Path(resources)/'tokenizer'
    return hashlib.sha256(''.join(sha256(root/name) for name in ('tokenizer.pkl', 'token_bytes.pt')).encode()).hexdigest()


def data_inventory(data_dir):
    return [(p.name, p.stat().st_size) for p in sorted(Path(data_dir).glob('*.parquet'))]


def inventory_fingerprint(inventory):
    return hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()


def verify_resources(resources, data_dir=None):
    resources = Path(resources)
    record = json.loads((resources/'native_manifest.json').read_text())
    if record['data_revision'] != DATA_REVISION or record['tokenizer_sha256'] != tokenizer_fingerprint(resources):
        raise ValueError('native resource provenance mismatch')
    data_dir = Path(data_dir) if data_dir is not None else resources/'base_data_climbmix'
    if inventory_fingerprint(data_inventory(data_dir)) != record['data_inventory_sha256']:
        raise ValueError('native dataset file inventory differs from the pinned manifest')
    if sha256(data_dir/VALIDATION_SHARD) != record['validation_sha256']:
        raise ValueError('native validation shard checksum differs')
    verify_upstream_sources(Path(__file__).resolve().parents[2], record['upstream_sources'])
    return record, sha256(resources/'native_manifest.json')


class BestFitStream:
    def __init__(self, data_dir, tokenizer, split, rank=0, world=1, sequence_len=2048,
                 buffer_size=1000, tokenizer_batch_size=128, tokenizer_threads=4):
        self.paths = sorted(Path(data_dir).glob('*.parquet'))
        if not self.paths or self.paths[-1].name != VALIDATION_SHARD:
            raise ValueError('native data requires the pinned final validation shard')
        self.paths = self.paths[:-1] if split == 'train' else self.paths[-1:]
        if split not in ('train', 'val') or not self.paths:
            raise ValueError('invalid split or no training shards')
        self.tokenizer, self.split = tokenizer, split
        self.rank, self.world = rank, world
        self.sequence_len, self.buffer_size = sequence_len, buffer_size
        self.tokenizer_batch_size, self.tokenizer_threads = tokenizer_batch_size, tokenizer_threads
        self.cursor = dict(pq_idx=0, rg_idx=rank, document_offset=0, epoch=1)
        self.buffer = []
        self.rows = 0
        self._loaded_key, self._texts = None, None
        self.identity = hashlib.sha256(json.dumps({
            'files': [(p.name, p.stat().st_size) for p in self.paths], 'rank':rank, 'world':world,
            'sequence_len': sequence_len, 'split':split, 'buffer':buffer_size,
            'tokenizer_batch_size':tokenizer_batch_size}, sort_keys=True).encode()).hexdigest()

    def _documents(self):
        c = self.cursor
        while True:
            if c['pq_idx'] >= len(self.paths):
                c.update(pq_idx=0, rg_idx=self.rank, document_offset=0, epoch=c['epoch']+1)
            key = (c['pq_idx'], c['rg_idx'], c['epoch'])
            pf = pq.ParquetFile(self.paths[c['pq_idx']]) if self._loaded_key != key else None
            if pf is not None and c['rg_idx'] >= pf.num_row_groups:
                c.update(pq_idx=c['pq_idx']+1, rg_idx=self.rank, document_offset=0)
                continue
            if self._loaded_key != key:
                self._texts = pf.read_row_group(c['rg_idx']).column('text').to_pylist()
                self._loaded_key = key
            start = c['document_offset']
            batch = self._texts[start:start+self.tokenizer_batch_size]
            c['document_offset'] += len(batch)
            if c['document_offset'] >= len(self._texts):
                c.update(rg_idx=c['rg_idx']+self.world, document_offset=0)
            if batch: return batch

    def _row(self):
        capacity = self.sequence_len+1
        row = []
        while len(row) < capacity:
            while len(self.buffer) < self.buffer_size:
                self.buffer.extend(self.tokenizer.encode(self._documents(), prepend=self.tokenizer.get_bos_token_id(),
                                                        num_threads=self.tokenizer_threads))
            remaining = capacity-len(row)
            best_idx, best_len = -1, 0
            for i, doc in enumerate(self.buffer):
                if best_len < len(doc) <= remaining: best_idx, best_len = i, len(doc)
            if best_idx < 0:
                best_idx = min(range(len(self.buffer)), key=lambda i:len(self.buffer[i]))
            row.extend(self.buffer.pop(best_idx)[:remaining])
        self.rows += 1
        return row

    def batch(self, batch_size, device):
        rows = torch.tensor([self._row() for _ in range(batch_size)], dtype=torch.long)
        x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
        if device.type == 'cuda': x, y = x.pin_memory(), y.pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def batches(self, batch_size, device):
        while True: yield self.batch(batch_size, device)

    def state_dict(self):
        return copy.deepcopy(dict(identity=self.identity, cursor=self.cursor, buffer=self.buffer, rows=self.rows))

    def load_state_dict(self, state):
        if state['identity'] != self.identity: raise ValueError('data stream configuration changed on resume')
        self.cursor, self.buffer, self.rows = copy.deepcopy((state['cursor'], state['buffer'], state['rows']))
        self._loaded_key, self._texts = None, None
