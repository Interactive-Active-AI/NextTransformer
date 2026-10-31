"""The frozen comparison protocol and protocol-bound configuration files."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from nexttransformer.models.config import HYBRID, ModelConfig
from nexttransformer.models.registry import FAMILY

REPO = Path(__file__).resolve().parents[2]
# SHA-256 of the shared tokenizer and byte table (tokenizer.pkl + token_bytes.pt) every formal run uses.
TOKENIZER_SHA256 = "06258dc0044b6198c6fa8ba75c5b3bbaeaeab44f4804b464d9ce5f5e3797ec54"


@dataclass(frozen=True)
class Protocol:
    version: str = "nextt-reference-50b-v1"
    data_revision: str = "915333b4f8b8684f39aeaafea600fea6f43fb703"
    validation_shard: str = "shard_06542.parquet"
    upstream_commit: str = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
    seed: int = 42
    total_tokens: int = 49_152_000_000
    global_batch_tokens: int = 1_048_576
    sequence_len: int = 2048
    vocab_size: int = 32768
    val_every: int = 250
    val_tokens: int = 41_943_040
    core_steps: tuple = (8000, 16000, 24000, 32000, 40000, 46875)
    core_tasks: int = 22
    final_bpb_tokens: int = 20_971_520
    data_parallel_size: int = 8
    optimizer: str = "muon_adamw"
    muon_lr: float = .02
    adamw_lr: float = .0006
    embedding_lr: float = .3
    unembedding_lr: float = .008
    scalar_lr: float = .5
    weight_decay: float = .28
    warmup_steps: int = 40
    warmdown_ratio: float = .65
    final_lr_fraction: float = .05
    grad_clip: float = 0.
    activation_checkpointing: bool = False
    packing_buffer: int = 1000
    tokenizer_batch_size: int = 128
    dtype: str = "bfloat16"
    preflight_tolerance: float = .01
    cumulative_tolerance: float = .02
    routing_warning: float = .03

    @property
    def steps(self):
        if self.total_tokens % self.global_batch_tokens:
            raise ValueError("token budget must contain complete optimizer updates")
        return self.total_tokens // self.global_batch_tokens

    @property
    def validation_steps(self):
        return (0, *range(self.val_every, self.steps, self.val_every), self.steps)

    def accumulation(self, micro_batch, world_size=8):
        if world_size != self.data_parallel_size:
            raise ValueError("formal protocol requires eight data-parallel ranks")
        denominator = world_size * micro_batch * self.sequence_len
        if denominator <= 0 or self.global_batch_tokens % denominator:
            raise ValueError("micro batch does not divide global batch")
        return self.global_batch_tokens // denominator

    def validate_train_config(self,train):
        for name in ("seed","total_tokens","global_batch_tokens","muon_lr","adamw_lr",
                     "embedding_lr","unembedding_lr","scalar_lr","weight_decay",
                     "warmup_steps","warmdown_ratio","final_lr_fraction","grad_clip",
                     "val_every","val_tokens","final_bpb_tokens"):
            if getattr(train,name)!=getattr(self,name):raise ValueError("controlled training field changed: "+name)
        if train.core_steps()!=list(self.core_steps) or train.core_max_per_task!=-1:
            raise ValueError("controlled CORE protocol changed")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


def config_file(directory, method):
    """<directory>/<family>/<method>.json for a configuration directory such as configs/."""
    return Path(directory)/FAMILY[method]/f"{method}.json"


def config_path(method, root=REPO):
    """Admitted configuration of a registered method: configs/<family>/<method>.json."""
    return config_file(Path(root)/"configs", method)


def load_config(path):
    data=json.loads(Path(path).read_text())
    if data["protocol"] != json.loads(json.dumps(asdict(Protocol()))):
        raise ValueError("configuration changes a controlled protocol variable")
    model=dict(data["model"]);layout=model.pop("layer_mixers",None)
    c=ModelConfig(**model)
    if (c.method in HYBRID or layout is not None) and layout!=list(c.layer_mixers):
        raise ValueError("serialized layer layout differs from the registered hybrid layout")
    if data.get("protocol_hash") != Protocol().fingerprint:
        raise ValueError("protocol hash mismatch")
    if data.get("model_hash") != c.fingerprint:
        raise ValueError("model hash mismatch")
    return c


def write_config(path, config, calibration=None):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(protocol=asdict(Protocol()),protocol_hash=Protocol().fingerprint,
                 model=config.serialized(),model_hash=config.fingerprint,
                 calibration=calibration or {"status":"candidate_not_calibrated"})
    p.write_text(json.dumps(payload,indent=2)+"\n")
