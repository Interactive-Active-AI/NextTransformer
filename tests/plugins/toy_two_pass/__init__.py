"""Test plugin for the model and stages hooks: one pass through the stack first, then repeated passes."""
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.protocol.spec import Protocol


class TwoPass(ReferenceLM):
    """Shared weights; the stack runs once during the first stage and `passes` times afterwards."""

    def set_training_stage(self, step):
        stage = super().set_training_stage(step)
        self.passes = 1 if step < self.config.option("single_pass_updates") else self.config.option("passes")
        return stage

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        x0, layers = x, self.config.n_layer
        passes = getattr(self, "passes", self.config.option("passes"))
        for repeat in range(passes):
            for layer in range(layers):
                x = self._block(x, x0, idx, layer, repeat * layers + layer, passes * layers)
        return x, None


model = TwoPass
# Without a cached midpoint hidden state the shell's backout scalar is unused, so it stays frozen.
frozen = ("backout_lambda",)


def stages(config):
    first = config.option("single_pass_updates")
    return [("single", first), ("repeated", Protocol().steps - first)]
