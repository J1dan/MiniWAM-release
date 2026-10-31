#!/usr/bin/env python3
"""Build or verify a relocatable frozen-case artifact directory (no cloud credentials)."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.validate_robotwin_instructions import validate_instructions
from scripts.eval.robotwin_split_contract import require_marked_contract
spec = importlib.util.spec_from_file_location('protocol', ROOT/'miniwam/robotwin_manifest.py')
protocol = importlib.util.module_from_spec(spec); spec.loader.exec_module(protocol)


def inside(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Invalid bundle path: {name}')
    return path


def validate_v2_certification(provenance):
    if provenance.get('schema_version') != 2:
        return
    validation = provenance.get('strict_replay_validation', {})
    if (provenance.get('release_status') != 'certified_exact_local_replay' or
            validation.get('exact_cases') != 10000 or
            validation.get('expected_cases') != 10000 or
            validation.get('scene_mismatch_cases') != 0 or
            validation.get('setup_error_cases') != 0):
        raise ValueError('V2 manifest lacks exact 10000/10000 replay certification')


def verify(directory, robotwin_root=ROOT/'RoboTwin'):
    index_path = directory/'bundle.json'
    seal = json.loads((directory/'bundle.sha256.json').read_text())
    if protocol.sha256(index_path) != seal['bundle_sha256']:
        raise ValueError('Bundle inventory SHA256 mismatch')
    inventory = json.loads(index_path.read_text())
    expected = set(inventory['files']) | {'bundle.json', 'bundle.sha256.json'}
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != expected or any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('Unexpected files or symlinks in cache-only bundle')
    fixed = {'robotwin_eval_manifest.jsonl', 'robotwin_eval_manifest.provenance.json',
             'text_cache/frozen_index.json', 'text_cache/frozen_index.sha256.json', 'text_cache/validation.json'}
    for name in inventory['files']:
        if name not in fixed and not re.fullmatch(
                r'text_cache/(?:[0-9a-f]{64}\.pt|tokenizer/(?:tokenizer\.json|tokenizer_config\.json|special_tokens_map\.json|spiece\.model|added_tokens\.json))', name):
            raise ValueError(f'Non-evaluation asset forbidden in cache-only bundle: {name}')
    for name, digest in inventory['files'].items():
        if protocol.sha256(inside(directory, name)) != digest:
            raise ValueError(f'Bundle file SHA256 mismatch: {name}')
    cases, provenance = protocol.load_manifest(directory/'robotwin_eval_manifest.jsonl')
    if (not re.fullmatch(r'[0-9a-f]{40}', provenance['prism_commit']) or
            not provenance['prism_source_hashes'] or
            provenance['instruction_validation']['valid_instructions'] != 10000):
        raise ValueError('Incomplete code/instruction validation provenance')
    validate_v2_certification(provenance)
    semantic = require_marked_contract(cases, provenance, robotwin_root)
    if semantic is not None:
        if (provenance.get('split_validation') != semantic or
                provenance.get('split_contract_sha256') != protocol.sha256(ROOT/'scripts/eval/robotwin_split_contract.py')):
            raise ValueError('Official split certification is absent or differs from current validator')
        validate_instructions(cases, robotwin_root)
    cache = json.loads((directory/'text_cache/frozen_index.json').read_text())
    validation = json.loads((directory/'text_cache/validation.json').read_text())
    seal = json.loads((directory/'text_cache/frozen_index.sha256.json').read_text())
    if protocol.sha256(directory/'text_cache/frozen_index.json') != seal['text_cache_sha256']:
        raise ValueError('Text-cache index SHA256 mismatch')
    if cache['eval_manifest_sha256'] != provenance['eval_manifest_sha256']:
        raise ValueError('Cache/manifest provenance mismatch')
    instructions = [entry['instruction'] for entry in cache['entries']]
    if len(instructions) != len(set(instructions)) or set(instructions) != {c['instruction'] for c in cases}:
        raise ValueError('Text cache does not exactly cover manifest')
    import hashlib
    instruction_hash = hashlib.sha256(json.dumps(sorted(instructions)).encode()).hexdigest()
    if (validation['total'] != len(instructions) or validation['exact_matches'] != len(instructions)
            or validation['instructions_sha256'] != instruction_hash or validation['contract'] != cache['contract']):
        raise ValueError('Missing/mismatched exact embedding validation report')
    for entry in cache['entries']:
        if protocol.sha256(inside(directory/'text_cache', entry['file'])) != entry['embedding_sha256']:
            raise ValueError('Embedding payload SHA256 mismatch')
    for name, digest in cache['tokenizer_files'].items():
        if protocol.sha256(inside(directory/'text_cache', name)) != digest:
            raise ValueError('Tokenizer SHA256 mismatch')
    print(f'Verified 10000 cases and {len(instructions)} exact text entries; bundle SHA256={protocol.sha256(index_path)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--text-index', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--verify', type=Path, metavar='BUNDLE_DIR')
    p.add_argument('--robotwin-root', type=Path, default=ROOT/'RoboTwin')
    args = p.parse_args()
    if args.verify:
        verify(args.verify, args.robotwin_root)
        return
    if not all((args.manifest, args.text_index, args.output_dir)):
        p.error('Building requires --manifest, --text-index, --output-dir')
    _, provenance = protocol.load_manifest(args.manifest, args.robotwin_root)
    index = json.loads(args.text_index.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {'robotwin_eval_manifest.jsonl': args.manifest,
        'robotwin_eval_manifest.provenance.json': args.manifest.with_suffix('.provenance.json'),
        'text_cache/frozen_index.json': args.text_index,
        'text_cache/frozen_index.sha256.json': args.text_index.with_suffix('.sha256.json'),
        'text_cache/validation.json': args.text_index.parent/'validation.json'}
    for entry in index['entries']:
        sources['text_cache/'+entry['file']] = inside(args.text_index.parent, entry['file'])
    for name in index['tokenizer_files']:
        sources['text_cache/'+name] = inside(args.text_index.parent, name)
    hashes = {}
    for name, source in sources.items():
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = protocol.sha256(target)
    protocol.atomic_json(output/'bundle.json', dict(schema_version=1, files=hashes))
    protocol.atomic_json(output/'bundle.sha256.json', dict(bundle_sha256=protocol.sha256(output/'bundle.json')))
    verify(output, args.robotwin_root)


if __name__ == '__main__':
    main()
