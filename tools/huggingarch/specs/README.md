# Fixed architecture specs

This directory contains 35 YAML specs: all 33 registered methods, `native-dense`
and `native-loopformer`. [catalog.json](catalog.json) records each spec's supported
configuration, checks and limitations. It also explains the unsupported dynamic
token compaction in `native-mor` and `native-circle`.

GPT-6.1 Sol agents authored these specs using HuggingArch's per-model `prompt.md`,
shared schema and source-review instructions. Parameter counts, tensor names,
shapes, multiplicities and graph geometry were checked against configured meta
models. These checks do not establish numerical equivalence or checkpoint validity.

When model source or configuration changes, review and update the corresponding
spec and catalog entry. See the [usage guide](../../../docs/src/content/docs/guides/huggingarch.md)
for commands, cache handling and validation scope.
