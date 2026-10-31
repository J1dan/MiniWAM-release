"""Files whose byte hashes the certified frozen release pins must not change.

The v3 bundle provenance records ``replay_state_sha256`` and
``split_contract_sha256``; the replay hook refuses evaluation when the working
copies differ, so even a comment edit breaks frozen evaluation.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFIED_RELEASE = "493d9b60ff501a9ff"
PINNED = {
    # provenance: replay_state_sha256
    "scripts/eval/robotwin_replay_state.py":
        "68b70f80ade0aa8b2972b9c42d4042d9f18469bdb837a581765c6df320ded6fa",
    # provenance: split_contract_sha256
    "scripts/eval/robotwin_split_contract.py":
        "30a441698911fd24d85a5a3bcfb84a376a076c50baa3a021cf1f8ef894cf4f87",
    # provenance: scene_protocol_sha256
    "miniwam/robotwin_manifest.py":
        "19ef394fcd33990cf47d2fbdcda092bc439781d8c72dc2b382ea3c7dcea251f8",
}


def test_release_pinned_sources_are_byte_identical():
    for name, expected in PINNED.items():
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert digest == expected, (
            f"{name} differs from the bytes pinned by frozen release {CERTIFIED_RELEASE}; "
            "frozen replay would refuse it. Revert the edit or certify a new release."
        )
