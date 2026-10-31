import json

import pytest
import torch

from miniwam.instruction_contract import (
    require_exact_instruction_match,
    require_instruction_string,
)
from miniwam.prompting import (
    DEFAULT_PROMPT,
    TEXT_FEATURE_CACHE_FILENAME,
    load_cached_instruction_features,
    resolve_text_feature_cache,
)
from scripts.data import build_libero_text_feature_cache as cache_builder


build_cache = cache_builder.build_cache


DATASET = "libero_10_no_noops_lerobot"
INSTRUCTION = "pick up the bowl"


def _write_episode(root, index, context, provenance=None, instruction=INSTRUCTION):
    episode = root / DATASET / f"episode_{index:06d}"
    episode.mkdir(parents=True)
    mask = torch.tensor([True, True, False, False])
    context = context.clone()
    context[~mask] = 0
    torch.save({"context": context, "mask": mask}, episode / "text_embed.pt")
    metadata = {
        "dataset_name": DATASET,
        "language_instruction": instruction,
    }
    if provenance is not None:
        metadata["text_provenance"] = provenance
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_build_and_load_libero_text_feature_cache(tmp_path):
    context = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    provenance = {
        "prompt_template": DEFAULT_PROMPT,
        "text_encoder_sha256": "abc",
        "tokenizer_id": "test-tokenizer",
        "context_len": 4,
    }
    _write_episode(tmp_path, 0, context, provenance)
    _write_episode(tmp_path, 1, context, provenance)
    output = tmp_path / TEXT_FEATURE_CACHE_FILENAME

    payload = build_cache(
        tmp_path, [DATASET], output,
        verify_all_duplicates=True, verify_benchmark=False,
    )
    loaded_context, loaded_mask = load_cached_instruction_features(
        output, DATASET, INSTRUCTION, expected_context_len=4, expected_text_dim=6,
    )

    assert payload["context_len"] == 4
    assert payload["text_dim"] == 6
    assert payload["source_provenance"][DATASET]["legacy_episodes_without_provenance"] == 0
    torch.testing.assert_close(loaded_context, payload["datasets"][DATASET][INSTRUCTION]["context"])
    assert loaded_mask.tolist() == [True, True, False, False]
    with pytest.raises(KeyError, match="exact instruction match"):
        load_cached_instruction_features(
            output, DATASET, f"  {INSTRUCTION}  ", expected_context_len=4, expected_text_dim=6,
        )
    # A cache stored in the checkpoint folder is found without an explicit path.
    assert resolve_text_feature_cache(None, tmp_path) == output
    assert resolve_text_feature_cache(None, tmp_path / DATASET) is None


def test_cache_rejects_duplicate_and_runtime_contract_mismatches(tmp_path):
    context = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    _write_episode(tmp_path, 0, context)
    _write_episode(tmp_path, 1, context + 1)
    output = tmp_path / TEXT_FEATURE_CACHE_FILENAME

    with pytest.raises(ValueError, match="different text embeddings"):
        build_cache(
            tmp_path, [DATASET], output,
            verify_all_duplicates=True, verify_benchmark=False,
        )

    build_cache(
        tmp_path, [DATASET], output,
        verify_all_duplicates=False, verify_benchmark=False,
    )
    with pytest.raises(ValueError, match="context_len"):
        load_cached_instruction_features(
            output, DATASET, INSTRUCTION, expected_context_len=8, expected_text_dim=6,
        )
    with pytest.raises(KeyError, match="no exact instruction match"):
        load_cached_instruction_features(
            output, DATASET, "unknown instruction", expected_context_len=4, expected_text_dim=6,
        )


@pytest.mark.parametrize(
    "different",
    [
        f" {INSTRUCTION}",
        f"{INSTRUCTION} ",
        f"\t{INSTRUCTION}",
        f"{INSTRUCTION}\n",
        f"{INSTRUCTION}\N{NO-BREAK SPACE}",
        "cafe\N{COMBINING ACUTE ACCENT}",
    ],
)
def test_instruction_matching_rejects_any_codepoint_difference(different):
    expected = "caf\N{LATIN SMALL LETTER E WITH ACUTE}" if different.startswith("cafe") else INSTRUCTION
    with pytest.raises(ValueError, match="Language instruction mismatch"):
        require_exact_instruction_match(
            different, expected, source="source metadata", current="current episode",
        )


def test_instruction_matching_requires_strings_and_preserves_json_unicode(tmp_path, monkeypatch):
    with pytest.raises(TypeError, match="must be a string"):
        require_instruction_string(123, "metadata")

    instruction = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    context = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    _write_episode(tmp_path, 0, context, instruction=instruction)
    metadata_text = (tmp_path / DATASET / "episode_000000" / "metadata.json").read_text()
    assert "\\u00e9" in metadata_text
    monkeypatch.setattr(
        cache_builder,
        "_benchmark_instruction_sets",
        lambda dataset_names: {name: {instruction} for name in dataset_names},
    )
    payload = build_cache(
        tmp_path, [DATASET], tmp_path / "unicode.pt",
        verify_all_duplicates=True, verify_benchmark=True,
    )
    assert instruction in payload["datasets"][DATASET]


def test_consolidated_cache_does_not_strip_metadata_instructions(tmp_path, monkeypatch):
    context = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    _write_episode(tmp_path, 0, context, instruction=f"{INSTRUCTION} ")
    monkeypatch.setattr(
        cache_builder,
        "_benchmark_instruction_sets",
        lambda dataset_names: {name: {INSTRUCTION} for name in dataset_names},
    )
    with pytest.raises(ValueError, match="Instruction mismatch"):
        build_cache(
            tmp_path, [DATASET], tmp_path / "mismatch.pt",
            verify_all_duplicates=True, verify_benchmark=True,
        )
