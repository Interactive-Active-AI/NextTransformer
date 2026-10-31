import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from nexttransformer.runtime import tracking, tracking_auth
from nexttransformer.runtime.site import bootstrap_runtime, resolve


def test_netrc_is_an_input_file(tmp_path, monkeypatch):
    site = tmp_path/'machine.local.json'
    site.write_text(json.dumps({'runtime_dir':str(tmp_path/'runtime'), 'wandb_netrc':'secrets/wandb.netrc'}))
    assert resolve(['--site-config',str(site)], {})['NETRC'] == str(tmp_path/'secrets/wandb.netrc')
    monkeypatch.setattr('os.environ', {})
    bootstrap_runtime(['--site-config',str(site)])
    assert not (tmp_path/'secrets').exists()


def test_disabled_and_offline_do_not_authenticate(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a,**k:pytest.fail('unexpected SDK startup'))
    tracking_auth.preflight('disabled')
    tracking_auth.preflight('offline')


@pytest.mark.parametrize('code,message', [(2,'No API key'),(3,'could not be verified'),(4,'not installed')])
def test_auth_failure_is_actionable_and_does_not_echo_sdk_secrets(monkeypatch, code, message):
    def run(*args, **kwargs):
        assert kwargs['stdin'] == subprocess.DEVNULL and kwargs['timeout'] == 60
        return SimpleNamespace(returncode=code, stderr=b'secret-key', stdout=b'secret-key')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError, match=message) as error:
        tracking_auth.preflight('online')
    assert 'secret-key' not in str(error.value)


def test_auth_timeout_is_bounded(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired('verification', 60, stderr=b'secret-key')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='timed out after 60s'):
        tracking_auth.preflight('online')


def test_rank_zero_init_failure_is_collective(monkeypatch):
    def init(*args):
        raise ValueError('initialization failed')
    monkeypatch.setattr(tracking, 'RunTracker', init)
    monkeypatch.setattr(tracking.dist, 'is_initialized', lambda:True)
    seen = []
    monkeypatch.setattr(tracking.dist, 'all_reduce', lambda failed,op:seen.append(failed.item()))
    with pytest.raises(RuntimeError, match='all ranks stopped') as error:
        tracking.start_tracker(None,0,None,None,None,None,torch.device('cpu'))
    assert seen == [1] and isinstance(error.value.__cause__, ValueError)


def test_peer_does_not_enter_validation_after_init_failure(monkeypatch):
    monkeypatch.setattr(tracking, 'RunTracker', lambda *args:object())
    monkeypatch.setattr(tracking.dist, 'is_initialized', lambda:True)
    monkeypatch.setattr(tracking.dist, 'all_reduce', lambda failed,op:failed.fill_(1))
    with pytest.raises(RuntimeError, match='all ranks stopped'):
        tracking.start_tracker(None,1,None,None,None,None,torch.device('cpu'))
