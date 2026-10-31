# Third-party notices

NextTransformer is released under the [MIT License](LICENSE). It includes or adapts the following
third-party material, which remains under its own license.

## nanochat

The `nanochat/` runtime and the backbone in `nexttransformer/models/backbone.py` derive from
[nanochat](https://github.com/karpathy/nanochat) at commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

```text
MIT License

Copyright (c) 2025 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`nanochat/optim.py` is adapted, as in nanochat, from the Muon implementation in
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT License).

## Method references

Method implementations are written from the papers and the pinned reference repositories listed in the
[source notes](https://interactive-active-ai.github.io/NextTransformer/docs/methods/sources/); no CUDA or
Triton kernel source is copied. Linear-attention training on GPUs calls the separately installed
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) package (MIT License).

## Website

- `website/public/fonts/` — Kalam, SIL Open Font License 1.1 (`website/public/fonts/Kalam-OFL.txt`).
- `website/vendor/` — shadcn/ui Tailwind styles, MIT License (license file alongside), when present.
