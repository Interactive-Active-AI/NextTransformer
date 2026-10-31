"""Sample text from a trained checkpoint.

    python -m nexttransformer.generate --checkpoint RUN/checkpoint_00046875 \\
        --tokenizer RESOURCES/tokenizer --prompt "The capital of France is"

Each new token reruns the full prefix (no KV cache), so generation works for every
registered architecture. Methods on FLA chunk backends need a CUDA device.
"""
import argparse
from pathlib import Path

import torch

from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.models import load_model


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=64, temperature=.8, top_k=50, seed=42):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    prefix = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    x = torch.tensor([prefix], dtype=torch.long, device=device)
    was_training = model.training; model.eval()
    try:
        for _ in range(max_new_tokens):
            out = model(x[:, -model.config.sequence_len:])
            logits = (out["logits"] if isinstance(out, dict) else out)[:, -1, :].float()
            if temperature <= 0:
                token = logits.argmax(-1, keepdim=True)
            else:
                logits = logits/temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                token = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)
            x = torch.cat([x, token], 1)
    finally:
        model.train(was_training)
    return tokenizer.decode(x[0, len(prefix):].tolist())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="a complete checkpoint_<step>/ directory")
    p.add_argument("--tokenizer", type=Path, required=True, help="prepared resources/tokenizer directory")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=.8, help="0: greedy")
    p.add_argument("--top-k", type=int, default=50, help="0: no truncation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    model, _ = load_model(args.checkpoint, args.device)
    tokenizer = RustBPETokenizer.from_directory(args.tokenizer)
    print(args.prompt + generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature,
                                 args.top_k, args.seed))


if __name__ == "__main__":
    main()
