"""Population budgets updated once per complete distributed optimizer update."""
import math
import torch
from torch import nn
from nexttransformer.models.native.depth import depth_reference


class CircleBudget(nn.Module):
    """Signed mean-depth dual plus a weak, fixed population survival prior.

    Penalty inputs have actual hard counts in forward and local gate ST
    derivatives in backward. No long product of sigmoid probabilities.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        probabilities,survival = depth_reference(config.core_layers,config.circle_max_steps,
                                                 config.circle_target_steps)
        self._probabilities, self._survival = probabilities,survival
        self.register_buffer('probabilities',torch.tensor(probabilities,dtype=torch.float64))
        self.register_buffer('survival',torch.tensor(survival,dtype=torch.float64))
        for name in ('dual','ema'):
            self.register_buffer(name,torch.zeros((),dtype=torch.float64))
        self.register_buffer('updates',torch.zeros((),dtype=torch.long))
        self.register_buffer('last_step',torch.tensor(-1,dtype=torch.long))

    @torch.no_grad()
    def reset(self):
        # to_empty() does not preserve even initialized buffers.
        self.probabilities.copy_(self.probabilities.new_tensor(self._probabilities))
        self.survival.copy_(self.survival.new_tensor(self._survival))
        for name in ('dual','ema','updates'):getattr(self,name).zero_()
        self.last_step.fill_(-1)

    def penalty(self, survival):
        c = self.config
        error = survival.sum()/c.circle_target_steps-1
        depth = self.dual.detach().clone().to(error)*error+.5*c.circle_depth_rho*error.square()
        shape = c.circle_shape_weight*(survival[c.core_layers:]-self.survival[c.core_layers:].to(survival)).square().mean()
        return depth+shape, depth, shape

    @torch.no_grad()
    def update(self, counts, tokens, step):
        c = self.config
        if step <= int(self.last_step) or tokens <= 0:
            raise ValueError('update Circle budget once per complete optimizer step')
        if len(counts) != c.circle_max_steps:
            raise ValueError('Circle budget needs counts at every block')
        values = [float(x)/tokens for x in counts]
        if (not all(math.isfinite(x) and 0 <= x <= 1 for x in values)
                or any(a+1e-9 < b for a,b in zip(values,values[1:]))
                or any(abs(x-1)>1e-9 for x in values[:c.core_layers])):
            raise ValueError('invalid absorbing Circle survival counts')
        mean = sum(values); error = mean/c.circle_target_steps-1
        used = float(self.dual)
        ema = c.circle_budget_beta*float(self.ema)+(1-c.circle_budget_beta)*error
        if step >= c.circle_random_warmup:
            self.dual.add_(c.circle_dual_lr*ema).clamp_(-c.circle_dual_limit,c.circle_dual_limit)
        self.ema.fill_(ema);self.last_step.fill_(step);self.updates.add_(1)
        errors = [a-b for a,b in zip(values,self._survival)]
        return dict(budget_target_steps=c.circle_target_steps,budget_error=error,budget_error_ema=ema,
                    budget_weight_used=used,budget_weight_next=float(self.dual),
                    budget_survival=values,budget_survival_error=errors,
                    budget_shape_rmse=math.sqrt(sum(x*x for x in errors[c.core_layers:])/(c.circle_max_steps-c.core_layers)),
                    budget_updates=int(self.updates))


class SurvivalBudget(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('dual', torch.zeros(config.recursions-1, dtype=torch.float64))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('last_step', torch.tensor(-1, dtype=torch.long))

    def penalty(self, hard_st_means):
        # Forward measures hard survival; backward uses the route surrogate.
        target = hard_st_means.new_tensor(self.config.mor_target_survival)
        error = hard_st_means-target
        dual = self.dual.detach().clone().to(error)
        return (dual*error + .5*self.config.mor_budget_strength*error.square()).sum()

    @torch.no_grad()
    def update(self, counts, tokens, step):
        if step <= int(self.last_step) or tokens <= 0:
            raise ValueError('update budget once per complete optimizer step')
        mean = self.dual.new_tensor(counts)/tokens
        if not torch.isfinite(mean).all() or (mean < 0).any() or (mean > 1).any():
            raise ValueError('invalid survival counts')
        error = mean-self.dual.new_tensor(self.config.mor_target_survival)
        self.dual.add_(self.config.mor_dual_lr*error).clamp_(
            -self.config.mor_dual_limit, self.config.mor_dual_limit)
        self.last_step.fill_(step)
        self.updates.add_(1)
        return {'budget_survival': mean.tolist(), 'budget_error': error.tolist(),
                'budget_dual': self.dual.tolist(), 'budget_updates': int(self.updates)}
