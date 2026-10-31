#!/usr/bin/env bash
# Source with optional --site-config/--runtime-dir; JSON is parsed as data, never sourced.
NEXTTRANSFORMER_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_nexttransformer_exports="$("${NEXTTRANSFORMER_BOOTSTRAP_PYTHON:-python3}" "$NEXTTRANSFORMER_REPO_ROOT/nexttransformer/runtime/site.py" --shell "$@")" || return $?
eval "$_nexttransformer_exports"
unset _nexttransformer_exports
