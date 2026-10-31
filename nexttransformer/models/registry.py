"""Registered methods and their families.

`METHODS` keeps its historical order: it orders configuration audits and test ids. Built-in
methods are listed here and implemented in their family modules; contributed methods are plugins
under nexttransformer/methods/ (see plugins.py), discovered from their method.json and appended in
name order, with no change to this file. Configurations live under configs/<family>/<method>.json.
Families are display metadata: cost-proof fingerprints exclude this file, and the model rules derive
from the built-in sets in config.py. SSSL keeps its historical id `dense` and is the cost anchor, but
it is not a baseline; only full-attention `dense_full` is.
"""
from nexttransformer.models.plugins import discover

BUILTIN_FAMILIES = {
    "baseline": ("dense_full",),
    "looped": ("ut_act", "fixed_recursive", "huginn", "ouro", "deeploop"),
    "sparse": ("dsa", "qsa", "nsa", "seer", "moba"),
    "linear": ("deltanet", "gla", "kda", "retnet", "elu_linear", "elu_norm", "elu_qknorm", "gdn", "kda_no_ve"),
    "residual": ("residual_standard", "hc", "mhc", "attnres"),
    "hybrid": ("gdn_hybrid", "gdn_hybrid_swa", "deltanet_hybrid", "kda_hybrid", "qsa_hybrid", "moba_hybrid"),
    "others": ("dense", "mla_nope", "mha_no_ve"),
}
BUILTIN_METHODS = ("dense", "ut_act", "fixed_recursive", "huginn", "ouro", "deeploop",
                   "dsa", "qsa", "nsa", "seer", "moba", "deltanet", "gla",
                   "kda", "retnet", "elu_linear", "elu_norm", "elu_qknorm", "gdn",
                   "dense_full", "residual_standard", "hc", "mhc", "attnres",
                   "gdn_hybrid", "gdn_hybrid_swa", "deltanet_hybrid",
                   "kda_hybrid", "kda_no_ve", "mla_nope", "mha_no_ve", "qsa_hybrid", "moba_hybrid")
# Hybrid layer kinds that are not methods themselves; no plugin may take these names.
MIXER_KINDS = ("attention", "swa")
PLUGINS = discover(BUILTIN_METHODS + MIXER_KINDS)
PLUGIN_METHODS = frozenset(PLUGINS)
FAMILIES = {family: (*members_, *(name for name in sorted(PLUGINS) if PLUGINS[name]["family"] == family))
            for family, members_ in BUILTIN_FAMILIES.items()}
METHODS = BUILTIN_METHODS + tuple(sorted(PLUGINS))
FAMILY = {method: family for family, members_ in FAMILIES.items() for method in members_}
assert sorted(FAMILY) == sorted(METHODS) and len(METHODS) == len(set(METHODS))


def members(family):
    """Built-in methods of a family; the model rules never apply to plugins."""
    return frozenset(BUILTIN_FAMILIES[family])
