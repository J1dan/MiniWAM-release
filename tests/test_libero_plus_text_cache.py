import pytest
import torch

from miniwam.libero_plus_text_cache import (
    LIBERO_PLUS_INSTRUCTION_CONTRACT,
    LiberoPlusTextFeatureCache,
    VANILLA_LIBERO_REV,
    build_packed_payload,
    compare_to_vanilla_cache,
)
from miniwam.prompting import DEFAULT_PROMPT, encode_instruction_features
from scripts.eval.libero_plus_protocol import LiberoPlusInstructions, LiberoPlusTask


def _record(task_id, name):
    return LiberoPlusTask(
        suite="libero_spatial",
        task_id=task_id,
        official_id=task_id + 1,
        task_name=name,
        category="Camera Viewpoints",
        difficulty=2,
    )


def _instructions(raw, clean=None, base=None, source="vanilla_libero_task_language"):
    clean = raw if clean is None else clean
    base = clean if base is None else base
    return LiberoPlusInstructions(raw=raw, clean=clean, base=base, source=source)


def _provenance():
    return {
        "prompt_template": DEFAULT_PROMPT,
        "instruction_contract": LIBERO_PLUS_INSTRUCTION_CONTRACT,
        "vanilla_libero_revision": VANILLA_LIBERO_REV,
        "vanilla_cache_verified": True,
    }


def test_packed_cache_is_bitwise_equal_and_preserves_strange_language():
    record = _record(3, "pick_bowl_view_0_0_100_initstate_7")
    instructions = _instructions(
        "pick bowl view 0 0 100 initstate 7",
        clean="pick bowl",
    )
    context = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    mask = torch.tensor([True, True, False, False])
    context[~mask] = 0
    payload = build_packed_payload(
        task_features=[(record, instructions, context, mask)],
        benchmark_revision="revision",
        classification_sha256="classification",
        manifest_id="manifest",
        text_provenance=_provenance(),
    )
    cache = LiberoPlusTextFeatureCache(payload=payload, source="test")
    restored, restored_mask = cache.get(
        record, instructions, expected_context_len=4, expected_text_dim=8,
    )
    assert torch.equal(restored, context)
    assert torch.equal(restored_mask, mask)
    assert payload["entries"][record.key]["raw_instruction"] == instructions.raw
    assert payload["entries"][record.key]["clean_instruction"] == instructions.clean
    with pytest.raises(ValueError, match="exact clean_instruction mismatch"):
        cache.get(
            record,
            _instructions(instructions.raw, clean="silently changed"),
            expected_context_len=4,
            expected_text_dim=8,
        )


def test_cache_deduplicates_only_exact_instruction_strings():
    first = _record(0, "task_suffix_1")
    second = _record(1, "task_suffix_2")
    context = torch.ones((4, 3), dtype=torch.bfloat16)
    mask = torch.tensor([True, False, False, False])
    context[~mask] = 0
    payload = build_packed_payload(
        task_features=[
            (first, _instructions("dirty suffix 1", clean="same exact text"), context, mask),
            (second, _instructions("dirty suffix 2", clean="same exact text"), context, mask),
        ],
        benchmark_revision="r", classification_sha256="c", manifest_id="m",
        text_provenance=_provenance(),
    )
    assert payload["packed_context"].shape == (1, 3)
    assert payload["entries"][first.key]["start"] == payload["entries"][second.key]["start"]


def test_cache_requires_existing_miniwam_baseline_equivalence_certificate():
    record = _record(0, "task_suffix")
    context = torch.ones((2, 3), dtype=torch.bfloat16)
    mask = torch.tensor([True, True])
    provenance = _provenance()
    provenance.pop("vanilla_cache_verified")
    with pytest.raises(ValueError, match="existing MiniWAM vanilla LIBERO text cache"):
        build_packed_payload(
            task_features=[(record, _instructions("task"), context, mask)],
            benchmark_revision="r",
            classification_sha256="c",
            manifest_id="m",
            text_provenance=provenance,
        )


def _vanilla_pair(scale=0.0, mask=None):
    generator = torch.Generator().manual_seed(0)
    baseline = torch.randn((4, 8), generator=generator).to(torch.bfloat16)
    baseline_mask = torch.tensor([True, True, True, False])
    baseline[~baseline_mask] = 0
    context = (baseline.float() * (1 + scale)).to(torch.bfloat16)
    return ("task", context, baseline_mask if mask is None else mask, baseline, baseline_mask)


def test_vanilla_cache_check_tolerates_small_numerical_differences():
    stats = compare_to_vanilla_cache([_vanilla_pair(), _vanilla_pair(scale=0.02)])
    assert stats["instructions"] == 2
    assert stats["bitwise_equal"] == 1
    assert 0 < stats["mean_relative_error"] < 0.05
    # A rebuild on another GPU type or the CPU lands near 0.05 and must still pass.
    stats = compare_to_vanilla_cache([_vanilla_pair(scale=0.06), _vanilla_pair(scale=0.06)])
    assert 0.05 < stats["mean_relative_error"] < 0.1


def test_vanilla_cache_check_rejects_systematic_differences():
    with pytest.raises(ValueError, match="mean relative error"):
        compare_to_vanilla_cache([_vanilla_pair(scale=0.3), _vanilla_pair(scale=0.3)])
    with pytest.raises(ValueError, match="Token mask differs"):
        compare_to_vanilla_cache([_vanilla_pair(mask=torch.tensor([True, True, False, False]))])
    with pytest.raises(ValueError, match="No unperturbed instruction"):
        compare_to_vanilla_cache([])


class _Tokenizer:
    def __call__(self, prompts, return_mask, add_special_tokens):
        assert prompts == [DEFAULT_PROMPT.format(task="task table 17")]
        return torch.tensor([[1, 2, 0]]), torch.tensor([[True, True, False]])


class _Encoder(torch.nn.Module):
    def forward(self, ids, mask):
        values = ids.unsqueeze(-1).expand(-1, -1, 2).float()
        return values


def test_canonical_encoder_has_exact_bf16_and_zero_padding_contract():
    context, mask = encode_instruction_features(
        _Encoder(), _Tokenizer(), "task table 17", torch.device("cpu"), torch.float32,
    )
    assert context.dtype == torch.bfloat16
    assert mask.tolist() == [True, True, False]
    assert torch.equal(context[2], torch.zeros(2, dtype=torch.bfloat16))
