import torch
import pytest
import hashlib
import json
from miniwam import robotwin_text_cache as module


def test_exact_persistent_cache_and_strict_miss(tmp_path, monkeypatch):
    encoder=tmp_path/'encoder.pt';encoder.write_bytes(b'weights')
    calls=[]
    monkeypatch.setattr(module,'load_text_encoder',lambda *args: (object(),object()))
    def encode(*args):
        calls.append(args[2])
        return torch.ones(2,3,dtype=torch.bfloat16),torch.ones(2,dtype=torch.bool)
    monkeypatch.setattr(module,'encode_text',encode)
    cache=module.RoboTwinTextCache(tmp_path/'cache',encoder,'tokenizer',2,torch.bfloat16,miss_policy='build')
    a=cache.get('pick up bottle')
    second=module.RoboTwinTextCache(tmp_path/'cache',encoder,'tokenizer',2,torch.bfloat16,miss_policy='error')
    import hashlib
    lock = second.directory/(hashlib.sha256(b'pick up bottle').hexdigest()+'.lock')
    lock.unlink()  # Offline builder's lock is not part of a published bundle.
    b=second.get('pick up bottle')
    assert torch.equal(a[0],b[0]) and calls==['pick up bottle']
    assert second.encoder is None
    assert not lock.exists()
    with pytest.raises(KeyError,match='Uncached'):
        second.get('pick up cup')
    encoder.write_bytes(b'different weights')
    changed=module.RoboTwinTextCache(tmp_path/'cache',encoder,'tokenizer',2,torch.bfloat16,miss_policy='error')
    with pytest.raises(KeyError):
        changed.get('pick up bottle')


def test_no_runtime_or_cpu_fallback(tmp_path):
    encoder = tmp_path/'encoder.pt'
    encoder.write_bytes(b'weights')
    with pytest.raises(ValueError, match='policy'):
        module.RoboTwinTextCache(tmp_path/'cache', encoder, 'tokenizer', 2,
                                torch.bfloat16, miss_policy='cpu')
    with pytest.raises(ValueError, match='CUDA'):
        module.RoboTwinTextCache(tmp_path/'cache', encoder, 'tokenizer', 2,
                                torch.bfloat16, miss_policy='build', build_device='cpu')
    strict = module.RoboTwinTextCache(tmp_path/'cache', encoder, 'tokenizer', 2,
                                    torch.bfloat16)
    with pytest.raises(KeyError, match='Uncached'):
        strict.get('never precomputed')
    assert strict.encoder is None


def test_cache_only_mode_binds_to_recorded_encoder_hash(tmp_path):
    encoder = tmp_path/'encoder.pt'; encoder.write_bytes(b'weights')
    from_file = module.RoboTwinTextCache(tmp_path/'cache', encoder, 'tokenizer', 2, torch.bfloat16)
    recorded = module.RoboTwinTextCache(
        tmp_path/'cache', tmp_path/'absent.pt', 'tokenizer', 2, torch.bfloat16,
        encoder_sha256=hashlib.sha256(b'weights').hexdigest())
    assert recorded.contract == from_file.contract
    assert recorded.directory == from_file.directory
    with pytest.raises(ValueError, match='recorded encoder hash'):
        module.RoboTwinTextCache(tmp_path/'cache', encoder, 'tokenizer', 2, torch.bfloat16,
                                 miss_policy='build', encoder_sha256='0' * 64)


def test_online_cache_is_bound_to_training_tokenizer(tmp_path, monkeypatch):
    encoder = tmp_path/'encoder.pt'; encoder.write_bytes(b'weights')
    backend = {'model': {'type': 'Unigram'}, 'normalizer': None}
    expected = hashlib.sha256(json.dumps(
        backend, sort_keys=True, separators=(',', ':')
    ).encode()).hexdigest()

    class Backend:
        def to_str(self):
            return json.dumps(backend)

    class Inner:
        backend_tokenizer = Backend()

    class Tokenizer:
        tokenizer = Inner()

    monkeypatch.setattr(module, 'load_text_encoder', lambda *args: (object(), Tokenizer()))
    monkeypatch.setattr(module, 'encode_text', lambda *args: (
        torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(2, dtype=torch.bool)))
    cache = module.RoboTwinTextCache(
        tmp_path/'cache', encoder, 'tokenizer', 2, torch.bfloat16,
        miss_policy='build', expected_tokenizer_backend_sha256=expected,
    )
    cache.get('official instruction')

    payload = torch.load(next(cache.directory.glob('*.pt')), weights_only=True)
    assert payload['tokenizer_backend_sha256'] == expected

    wrong = module.RoboTwinTextCache(
        tmp_path/'cache', encoder, 'tokenizer', 2, torch.bfloat16,
        miss_policy='build', expected_tokenizer_backend_sha256='f'*64,
    )
    with pytest.raises(ValueError, match='Online tokenizer differs'):
        wrong.get('official instruction')


def test_online_encoder_can_be_preloaded_idempotently(tmp_path, monkeypatch):
    encoder = tmp_path/'encoder.pt'; encoder.write_bytes(b'weights')
    calls = []
    monkeypatch.setattr(module, 'load_text_encoder', lambda *args: (
        calls.append(args) or object(), object()))
    cache = module.RoboTwinTextCache(
        tmp_path/'cache', encoder, 'tokenizer', 2, torch.bfloat16,
        miss_policy='build',
    )

    cache.ensure_encoder_loaded()
    cache.ensure_encoder_loaded()

    assert len(calls) == 1
    assert cache.encoder is not None


def test_frozen_cache_preload_is_a_noop(tmp_path, monkeypatch):
    encoder = tmp_path/'encoder.pt'; encoder.write_bytes(b'weights')
    monkeypatch.setattr(module, 'load_text_encoder', lambda *args: pytest.fail(
        'frozen cache must not load the online encoder'))
    cache = module.RoboTwinTextCache(
        tmp_path/'cache', encoder, 'tokenizer', 2, torch.bfloat16,
        miss_policy='error',
    )

    cache.ensure_encoder_loaded()

    assert cache.encoder is None
