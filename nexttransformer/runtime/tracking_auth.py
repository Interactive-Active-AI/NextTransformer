"""Bounded, non-interactive W&B authentication before creating a process group."""
import subprocess
import sys


def preflight(mode, timeout=60):
    if mode != 'online':
        return
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'nexttransformer.runtime.tracking_auth'],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'W&B authentication timed out after {timeout}s; check network access to your W&B server') from None
    reasons = {
        2: 'No API key configured in this training container',
        3: 'W&B credentials could not be verified (credentials or network)',
        4: 'W&B is not installed; install with setup_environment.sh --tracking',
    }
    if result.returncode:
        # Do not echo SDK output: credential/configuration errors can contain secrets.
        reason = reasons.get(result.returncode, 'W&B authentication subprocess failed')
        raise RuntimeError(reason + '; provide WANDB_API_KEY or a mounted NETRC file (--wandb-netrc). '
                           'Notebook login alone does not authenticate a separate training container.')
    print('W&B online authentication verified before distributed initialization.', flush=True)


def _verify():
    try:
        import wandb
    except ImportError:
        return 4
    try:
        return 0 if wandb.login(verify=True, prompt=False) else 2
    except Exception:
        return 3


if __name__ == '__main__':
    raise SystemExit(_verify())
