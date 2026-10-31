"""Exact-instruction disk cache shared by RoboTwin evaluation workers."""
import fcntl
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path

import torch

from miniwam.prompting import DEFAULT_PROMPT
from miniwam.text_encoding import encode_text, load_text_encoder


def comparable_contract(contract):
    """Drop recorded source-file hashes (encoding_sources); they are not compared."""
    return {key: value for key, value in contract.items() if key != 'encoding_sources'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def encoder_fingerprint(path, directory):
    """Avoid rereading the 11-GB encoder in every worker; invalidate on file changes."""
    path = Path(path).resolve()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(str(path).encode()).hexdigest()
    manifest = directory / f'encoder_{name}.json'
    with (directory / f'encoder_{name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stat = path.stat()
        signature = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved['signature'] == signature:
                return saved['sha256']
        fingerprint = file_sha256(path)
        after = path.stat()
        if signature != [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]:
            raise RuntimeError('Encoder changed while fingerprinting.')
        temporary = manifest.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(dict(signature=signature, sha256=fingerprint)))
        temporary.replace(manifest)
        return fingerprint


class RoboTwinTextCache:
    def __init__(self, directory, encoder_path, tokenizer_id, context_len, dtype,
                 miss_policy='error', build_device='cuda',
                 expected_tokenizer_backend_sha256=None, encoder_sha256=None):
        if miss_policy not in {'build', 'error'}:
            raise ValueError('Text cache policy must be build or error.')
        if encoder_sha256 is not None and miss_policy != 'error':
            raise ValueError('Only cache-only evaluation may use a recorded encoder hash.')
        if miss_policy == 'build' and not str(build_device).startswith('cuda'):
            raise ValueError('Canonical RoboTwin text extraction requires CUDA; CPU fallback is disabled.')
        self.encoder_path = Path(encoder_path) if encoder_path is not None else None
        self.tokenizer_id = tokenizer_id
        self.context_len = int(context_len)
        self.dtype = dtype
        self.miss_policy = miss_policy
        self.build_device = build_device
        self.encoder = self.tokenizer = None
        self.tokenizer_backend_sha256 = None
        self.expected_tokenizer_backend_sha256 = expected_tokenizer_backend_sha256
        # Cache-only evaluation never loads the encoder, so it can bind the cache to
        # the hash recorded at training time instead of reading the 11-GB file.
        if encoder_sha256 is None:
            encoder_sha256 = encoder_fingerprint(self.encoder_path, directory)
        self.contract = dict(schema_version=2, prompt_template=DEFAULT_PROMPT,
                             encoder_sha256=encoder_sha256,
                             tokenizer_id=str(tokenizer_id), context_len=self.context_len,
                             compute_dtype=str(dtype), storage_dtype='torch.bfloat16', encoding_backend='cuda')
        # Namespace by encoder contents and prompt/config, never checkpoint label.
        # Preserve the released/offline namespace byte-for-byte. Official online
        # caches add the tokenizer binding to prevent cross-tokenizer reuse.
        namespace_contract = self.contract
        if expected_tokenizer_backend_sha256 is not None:
            namespace_contract = dict(
                contract=self.contract,
                tokenizer_backend_sha256=expected_tokenizer_backend_sha256,
            )
        namespace = hashlib.sha256(json.dumps(namespace_contract, sort_keys=True).encode()).hexdigest()
        self.directory = Path(directory) / namespace
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frozen_entries = None
        frozen_index = os.environ.get('MINIWAM_FROZEN_TEXT_INDEX')
        if frozen_index and miss_policy == 'error':
            from miniwam.robotwin_manifest import load_manifest
            cases, provenance = load_manifest(os.environ['MINIWAM_EVAL_MANIFEST'])
            index_path = Path(frozen_index)
            index = json.loads(index_path.read_text())
            seal = json.loads(index_path.with_suffix('.sha256.json').read_text())
            if (file_sha256(index_path) != seal['text_cache_sha256'] or
                    comparable_contract(index['contract']) != self.contract or
                    index['eval_manifest_sha256'] != provenance['eval_manifest_sha256']):
                raise ValueError('Frozen text cache contract/provenance mismatch')
            self.directory = index_path.parent
            self.frozen_entries = {entry['instruction']: entry for entry in index['entries']}
            if (len(self.frozen_entries) != len(index['entries']) or
                    set(self.frozen_entries) != {case['instruction'] for case in cases}):
                raise ValueError('Frozen text cache does not exactly cover manifest instructions')
            for name, expected in index['tokenizer_files'].items():
                if file_sha256(self.directory/name) != expected:
                    raise ValueError('Frozen tokenizer asset hash mismatch')

    def ensure_encoder_loaded(self):
        """Load and validate the online encoder before accepting rollout calls."""
        if self.miss_policy != 'build':
            return
        if self.encoder is not None:
            return
        # Loading four UMT5 copies concurrently can exhaust host RAM. Keep the
        # existing cross-worker serialization, but perform it during policy
        # server startup where the launcher has a 20-minute readiness budget.
        with (self.directory / 'encoder.lock').open('a') as encoder_lock:
            fcntl.flock(encoder_lock, fcntl.LOCK_EX)
            if self.encoder is not None:
                return
            self.encoder, self.tokenizer = load_text_encoder(
                self.encoder_path, self.tokenizer_id, self.context_len,
                self.build_device, self.dtype)
            if self.expected_tokenizer_backend_sha256 is not None:
                backend = json.loads(
                    self.tokenizer.tokenizer.backend_tokenizer.to_str()
                )
                self.tokenizer_backend_sha256 = hashlib.sha256(json.dumps(
                    backend, sort_keys=True, separators=(',', ':')
                ).encode()).hexdigest()
                if (self.tokenizer_backend_sha256 !=
                        self.expected_tokenizer_backend_sha256):
                    raise ValueError(
                        'Online tokenizer differs from the tokenizer used for '
                        'training feature extraction.'
                    )

    def get(self, instruction):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Expected a nonempty exact instruction string.')
        key = instruction
        filename = hashlib.sha256(key.encode()).hexdigest()
        path = self.directory / f'{filename}.pt'
        if self.frozen_entries is not None:
            entry = self.frozen_entries[key]
            if entry['file'] != path.name or file_sha256(path) != entry['embedding_sha256']:
                raise ValueError('Frozen embedding SHA256 mismatch')
        # Frozen evaluation assets are read-only. Only offline builders need
        # per-entry lock files; creating them here invalidates the sealed bundle.
        lock_context = ((self.directory / f'{filename}.lock').open('a')
                        if self.miss_policy == 'build' else nullcontext(None))
        with lock_context as lock:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                payload = torch.load(path, map_location='cpu', weights_only=True)
            else:
                if self.miss_policy == 'error':
                    raise KeyError(f'Uncached RoboTwin instruction: {key!r}. '
                                   'Pre-extract it with build_robotwin_text_feature_cache.py.')
                self.ensure_encoder_loaded()
                print(f'[MiniWAM] Encoding uncached instruction on {self.build_device}: {key}', flush=True)
                context, mask = encode_text(self.encoder, self.tokenizer, key,
                                            self.build_device, self.dtype)
                payload = dict(contract=self.contract, instruction=key, context=context, mask=mask,
                               encoding_device=str(self.build_device))
                if self.expected_tokenizer_backend_sha256 is not None:
                    payload['tokenizer_backend_sha256'] = self.tokenizer_backend_sha256
                temporary = path.with_suffix(f'.{os.getpid()}.tmp')
                torch.save(payload, temporary)
                temporary.replace(path)
            if (comparable_contract(payload.get('contract') or {}) != self.contract or payload.get('instruction') != key
                    or not str(payload.get('encoding_device','')).startswith('cuda')):
                raise ValueError(f'Text cache provenance mismatch: {path}')
            if (self.expected_tokenizer_backend_sha256 is not None and
                    payload.get('tokenizer_backend_sha256') !=
                    self.expected_tokenizer_backend_sha256):
                raise ValueError(f'Text cache tokenizer provenance mismatch: {path}')
            context, mask = payload['context'], payload['mask']
            if (context.ndim != 2 or context.shape[0] != self.context_len or
                    mask.shape != (self.context_len,) or mask.dtype != torch.bool or
                    context.dtype != torch.bfloat16 or not torch.isfinite(context).all() or
                    torch.count_nonzero(context[~mask]).item()):
                raise ValueError(f'Invalid text cache tensors: {path}')
            return context, mask
