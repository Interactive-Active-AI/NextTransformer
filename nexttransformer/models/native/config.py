"""Scientific configuration for the native d24 comparison."""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

UPSTREAM_COMMIT = '92d63d4e8bb4df75c3b71618f31ddde2378b2bcd'


@dataclass(frozen=True)
class NativeModelConfig:
    architecture: str = 'dense'
    dense_layers: int = 24
    pre_layers: int = 3
    core_layers: int = 9
    post_layers: int = 3
    recursions: int | None = None
    activation_checkpointing: bool = False
    fourier_dim: int = 256
    loop_modulation_bound: float = .5
    loop_condition_lr_multiplier: float = .5
    loop_condition_grad_clip: float = 1.
    vocab_size: int = 32768
    sequence_len: int = 2048
    width: int = 1536
    heads: int = 12
    mlp_width: int = 6144
    window_pattern: str = 'SSSL'
    circle_max_steps: int = 27
    circle_target_steps: float = 17.5
    circle_method: str = 'sparse_gate'
    circle_budget_mode: str = 'mean_survival'
    circle_halter: str = 'delta_depth'
    circle_halter_eps: float = 1e-6
    circle_exit_interval: int = 1
    circle_threshold: float = .5
    circle_routing_seed: int = 1729
    circle_budget_beta: float = .9
    circle_random_warmup: int = 256
    circle_route_transition: int = 768
    circle_depth_rho: float = 1.0
    circle_shape_weight: float = 0.1
    circle_dual_lr: float = 0.05
    circle_dual_limit: float = 2.0
    circle_halter_lr_multiplier: float = 0.5
    circle_halter_grad_clip: float = 1.0
    mor_target_survival: tuple = (.64, .32)
    mor_budget_strength: float = 1.0
    mor_dual_lr: float = .05
    mor_dual_limit: float = 10.0

    def __post_init__(self):
        expected = {'dense':1,'mor':3,'loopformer':2,'circle':3}
        if self.architecture not in expected:
            raise ValueError('invalid native architecture')
        if self.recursions is None:
            object.__setattr__(self,'recursions',expected[self.architecture])
        if min(self.width,self.heads,self.sequence_len,self.pre_layers,self.core_layers,
               self.post_layers,self.dense_layers,self.fourier_dim) <= 0 or self.width % self.heads:
            raise ValueError('invalid native model dimensions')
        if self.recursions != expected[self.architecture]:
            raise ValueError('common-core protocol: MoR/Circle max3, LoopFormer fixed2')
        if not 0 < self.circle_halter_eps < 1:
            raise ValueError('circle_halter_eps must lie between 0 and 1')
        if self.activation_checkpointing:
            raise ValueError('native FLOP protocol currently requires activation_checkpointing=false')
        if self.mlp_width != 4*self.width or self.width < 24 or self.width//self.heads % 2:
            raise ValueError('native backbone requires MLP=4d, width>=24 and even head dimension')
        if not self.window_pattern or set(self.window_pattern)-set('SL'):
            raise ValueError('window_pattern must contain only S and L')
        p = (1.,*self.mor_target_survival,0.)
        if len(self.mor_target_survival) != 2 or not all(a>b for a,b in zip(p,p[1:])):
            raise ValueError('MoR requires two strictly decreasing survival targets')
        if min(self.mor_budget_strength, self.mor_dual_lr, self.mor_dual_limit) <= 0:
            raise ValueError('MoR budget settings must be positive')
        if self.architecture == 'loopformer':
            if not 0 < self.loop_modulation_bound < 1:
                raise ValueError('LoopFormer modulation bound must lie between zero and one')
            if not all(math.isfinite(v) and v > 0 for v in
                       (self.loop_condition_lr_multiplier,self.loop_condition_grad_clip)):
                raise ValueError('LoopFormer conditioning optimizer settings must be positive')
        if self.architecture == 'circle':
            if self.circle_halter != 'delta_depth':
                raise ValueError('circle_halter must be delta_depth; legacy Circle needs its frozen source')
            if (self.circle_method != 'sparse_gate'
                    or self.circle_budget_mode != 'mean_survival' or self.circle_exit_interval != 1):
                raise ValueError('native Circle requires sparse_gate, mean_survival and per-block exits')
            if self.circle_max_steps != self.core_layers*self.recursions:
                raise ValueError('Circle max steps must equal three complete rings')
            if not self.core_layers < self.circle_target_steps < self.circle_max_steps:
                raise ValueError('Circle target must be strictly inside its depth range')
            if self.circle_threshold != .5:
                raise ValueError('sparse Circle uses a fixed 0.5 threshold')
            values = (self.circle_target_steps,self.circle_depth_rho,self.circle_shape_weight,
                      self.circle_dual_lr,self.circle_dual_limit,self.circle_budget_beta,
                      self.circle_halter_lr_multiplier,self.circle_halter_grad_clip)
            if not all(math.isfinite(v) for v in values): raise ValueError('nonfinite Circle setting')
            if min(self.circle_depth_rho,self.circle_dual_lr,self.circle_dual_limit,
                   self.circle_halter_lr_multiplier,self.circle_halter_grad_clip) <= 0:
                raise ValueError('Circle loss/optimizer settings must be positive')
            if self.circle_shape_weight < 0 or not 0 <= self.circle_budget_beta < 1:
                raise ValueError('invalid Circle shape weight or smoothing')
            if min(self.circle_random_warmup,self.circle_route_transition) < 0:
                raise ValueError('Circle curriculum lengths must be nonnegative')

    @property
    def n_layer(self):
        return self.dense_layers if self.architecture == 'dense' else self.pre_layers+self.core_layers+self.post_layers

    @property
    def max_depth(self):
        if self.architecture == 'dense':return self.dense_layers
        middle = self.circle_max_steps if self.architecture == 'circle' else self.core_layers*self.recursions
        return self.pre_layers+middle+self.post_layers

    @property
    def routing_steps(self):
        return self.circle_max_steps if self.architecture == 'circle' else self.recursions

    @property
    def n_embd(self): return self.width
    @property
    def n_head(self): return self.heads
    @property
    def n_kv_head(self): return self.heads


@dataclass(frozen=True)
class NativeTrainConfig:
    seed: int = 42
    total_tokens: int = 49152000000
    checkpoint_every: int = 5000
    muon_lr: float = .02
    adamw_lr: float = .0006
    global_batch_tokens: int = 1048576
    val_tokens: int = 80*524288  # base_train, not base_eval's smaller final split
    val_every: int = 250
    final_bpb_tokens: int = 40*524288
    core_every: int = 8000
    core_max_per_task: int = -1
    warmup_steps: int = 40
    warmdown_ratio: float = .65
    final_lr_fraction: float = .05
    embedding_lr: float = .3
    unembedding_lr: float = .008
    scalar_lr: float = .5
    weight_decay: float = .28
    # Native base_train does not clip gradients. Optional explicit override.
    grad_clip: float = 0.0

    @property
    def steps(self):
        if self.total_tokens <= 0 or self.global_batch_tokens <= 0 or self.total_tokens % self.global_batch_tokens:
            raise ValueError('total_tokens must be a positive multiple of global_batch_tokens')
        return self.total_tokens//self.global_batch_tokens

    def accumulation(self, model, world_size, micro_batch):
        denominator = world_size*micro_batch*model.sequence_len
        if denominator <= 0 or self.global_batch_tokens % denominator:
            raise ValueError('global batch must be divisible by world_size * micro_batch * sequence_len')
        return self.global_batch_tokens//denominator

    def validation_steps(self):
        return sorted({0,self.steps,*range(self.val_every,self.steps,self.val_every)})

    def core_steps(self, include_final=True):
        # Final CORE has its own post-training path, even when it is a multiple
        # of core_every. Non-positive intervals disable intermediate CORE only.
        steps = list(range(self.core_every, self.steps, self.core_every)) if self.core_every > 0 else []
        return steps + ([self.steps] if include_final else [])


def load_config(path):
    raw = json.loads(Path(path).read_text())
    if set(raw) != {'model', 'train'}:
        raise ValueError('configuration requires model and train sections')
    if 'mor_target_survival' in raw['model']:
        raw['model']['mor_target_survival'] = tuple(raw['model']['mor_target_survival'])
    mc, tc = NativeModelConfig(**raw['model']), NativeTrainConfig(**raw['train'])
    tc.steps
    if tc.val_tokens % mc.sequence_len or tc.final_bpb_tokens % mc.sequence_len:
        raise ValueError('evaluation budgets must contain whole sequences')
    return mc, tc


def write_config(path, mc, tc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({'model': asdict(mc), 'train': asdict(tc)}, indent=2)+'\n')
