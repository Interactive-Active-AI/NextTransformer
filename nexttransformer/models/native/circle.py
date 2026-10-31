"""Sparse, absorbing Circle routes with gate-weighted selected updates."""
import torch
import torch.distributed as dist


class SparseCircleMixin:
    def set_circle_step(self, step):
        if step < 0:raise ValueError('Circle curriculum step must be nonnegative')
        self.circle_step = step

    def learned_route_fraction(self):
        if not self.training:return 1.
        c = self.config
        elapsed = getattr(self,'circle_step',0)-c.circle_random_warmup
        if elapsed < 0:return 0.
        return min(1.,(elapsed+1)/max(1,c.circle_route_transition))

    def circle_stage(self):
        fraction = self.learned_route_fraction()
        return 'random_warmup' if fraction == 0 else 'learned' if fraction == 1 else 'route_transition'

    def routing_generator(self, device):
        if getattr(self,'_routing_generator',None) is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            self._routing_generator = torch.Generator(device=device).manual_seed(self.config.circle_routing_seed+rank)
            pending = getattr(self,'_routing_state',None)
            if pending is not None:
                self._routing_generator.set_state(pending);self._routing_state=None
        return self._routing_generator

    def routing_rng_state(self):
        generator = getattr(self,'_routing_generator',None)
        return generator.get_state() if generator is not None else getattr(self,'_routing_state',None)

    def restore_routing_rng(self, state):
        self._routing_generator=None;self._routing_state=state

    def _circle_random_routes(self, idx):
        fraction = self.learned_route_fraction()
        if fraction == 1:return torch.zeros_like(idx,dtype=torch.bool),None
        generator = self.routing_generator(idx.device)
        forced = torch.rand(idx.shape,device=idx.device,generator=generator) >= fraction
        uniform = torch.rand(idx.shape,device=idx.device,generator=generator)
        depths = torch.searchsorted(self.circle_budget.probabilities.cumsum(0),uniform.double().contiguous())
        depths = depths.clamp_max(self.config.circle_max_steps-self.config.core_layers)+self.config.core_layers
        return forced,depths

    def _sparse_circle(self, x, idx, run):
        """run() returns one native block's candidate at the next virtual depth.

        The first ring is dense. Subsequent blocks gather Q/K/V and MLP for
        hard survivors. Halter scores use only states that already exist.
        """
        c = self.config
        active = torch.ones_like(idx,dtype=torch.bool)
        forced,depths = self._circle_random_routes(idx)
        gate = None
        survival = []
        pending = x.new_tensor(1.,dtype=torch.float32)
        budget_anchor = x.new_zeros((),dtype=torch.float32)
        backout = None
        for s in range(1,c.circle_max_steps+1):
            count = int(active.sum())
            self.last_stats['active_tokens'][s-1] = count
            survival.append(pending)
            before = x
            candidate = run(c.pre_layers+(s-1)%c.core_layers,x,
                            active=None if s <= c.core_layers else active)
            x = candidate if gate is None else before+gate.to(x.dtype)[...,None]*(candidate-before)
            # The owner must capture the gated state, never its unweighted candidate.
            if c.pre_layers+s-1 == c.max_depth//2:backout = x
            if s < c.core_layers:continue
            if s == c.circle_max_steps:break
            scores = self.halter(before[active],x[active],s-c.core_layers)
            q = torch.zeros_like(idx,dtype=torch.float32).masked_scatter(active,torch.sigmoid(-scores))
            self.last_stats['forward_flops'] += 2*count*c.width
            self.last_stats['halter_extra_forward_ops'] += count*(5*c.width+10)
            selected = q >= .5
            if depths is not None:selected = torch.where(forced,depths>s,selected)
            next_active = active & selected
            # Local ST: no product through earlier hard gates. Randomly forced
            # decisions do not send budget gradients to content scores.
            local_st = next_active.float()+(active & ~forced).float()*(q-q.detach())
            pending = local_st.mean()+budget_anchor
            # Keep every local gate graph reachable even after an empty tail.
            budget_anchor = budget_anchor+q.sum()*0
            active,gate = next_active,q
        penalty,depth_loss,shape_loss = self.circle_budget.penalty(torch.stack(survival))
        if not self.training:penalty = x.new_zeros((),dtype=torch.float32)
        self.last_stats['active_depth_sum'] = idx.numel()*(c.pre_layers+c.post_layers)+sum(self.last_stats['active_tokens'])
        self.last_stats['circle_depth_loss'] = depth_loss.detach()
        self.last_stats['circle_shape_loss'] = shape_loss.detach()
        return x,penalty,backout
