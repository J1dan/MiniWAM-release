"""Pinned LIBERO-Plus registry, import, sampling, and reporting contracts."""

from __future__ import annotations

import ast
import hashlib
import importlib
import io
import json
import math
import os
import subprocess
import sys
from contextlib import redirect_stdout
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LIBERO_PLUS_REV = "4976dc30028e805ff8094b55501d532c48fec182"
VANILLA_LIBERO_REV = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CLASSIFICATION_SHA256 = "faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_SIZES = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)
DIFFICULTIES = (1, 2, 3, 4, 5, None)
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
SELECTION_SCHEMA = "libero_plus_ablation_stratified_v1"
MANIFEST_COLUMNS = (
    "suite",
    "task_id",
    "official_id",
    "task_name",
    "category",
    "difficulty",
)


@dataclass(frozen=True)
class LiberoPlusTask:
    suite: str
    task_id: int
    official_id: int
    task_name: str
    category: str
    difficulty: int | None

    @property
    def key(self) -> str:
        return f"{self.suite}:{self.task_id}"


@dataclass(frozen=True)
class LiberoPlusInstructions:
    raw: str
    clean: str
    base: str
    source: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_paths(libero_plus_root: Path) -> tuple[Path, Path]:
    root = Path(libero_plus_root).expanduser().resolve()
    benchmark_dir = root / "libero" / "libero" / "benchmark"
    return (
        benchmark_dir / "task_classification.json",
        benchmark_dir / "libero_suite_task_map.py",
    )


def _load_task_map(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                break
            return value
    raise ValueError(f"Could not find a literal libero_task_map in {path}.")


def _vanilla_language_from_task_name(task_name: str) -> str:
    filename = task_name + ".bddl"
    if filename[0].isupper():
        offset = 8 if "SCENE10" in filename else 7
        language = " ".join(filename[filename.find("SCENE") + offset:].split("_"))
    else:
        language = " ".join(filename.split("_"))
    return language[:language.find(".bddl")]


def load_vanilla_instruction_registry(
    libero_root: Path, *, require_pinned: bool = True,
) -> dict[str, dict[str, str]]:
    """Load the exact task-language style used by MiniWAM's vanilla LIBERO extraction."""
    root = Path(libero_root).expanduser().resolve()
    task_map_path = root / "libero/libero/benchmark/libero_suite_task_map.py"
    if not task_map_path.is_file():
        raise FileNotFoundError(f"Vanilla LIBERO task map does not exist: {task_map_path}.")
    if require_pinned:
        if not (root / ".git").is_dir():
            raise FileNotFoundError(f"Vanilla LIBERO must be a Git checkout: {root}.")
        actual_rev = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual_rev != VANILLA_LIBERO_REV:
            raise RuntimeError(f"Vanilla LIBERO is at {actual_rev}, expected {VANILLA_LIBERO_REV}.")
    task_map = _load_task_map(task_map_path)
    registry = {}
    for suite in SUITES:
        names = task_map.get(suite)
        if not isinstance(names, list) or len(names) != 10:
            raise ValueError(f"Vanilla LIBERO {suite} must contain exactly 10 base tasks.")
        registry[suite] = {
            name: _vanilla_language_from_task_name(name)
            for name in names
        }
    return registry


def resolve_libero_plus_instructions(
    record: LiberoPlusTask,
    task,
    vanilla_registry: dict[str, dict[str, str]],
) -> LiberoPlusInstructions:
    """Separate the fork's filename-derived language from the model-facing instruction."""
    if task.name != record.task_name:
        raise ValueError(f"Runtime task name mismatch for {record.key}.")
    candidates = [
        (name, language)
        for name, language in vanilla_registry[record.suite].items()
        if task.name == name or task.name.startswith(name + "_")
    ]
    if not candidates:
        raise ValueError(f"Could not map LIBERO-Plus task {record.key} to a vanilla base task.")
    base_name, base_instruction = max(candidates, key=lambda item: len(item[0]))
    suffix = task.name[len(base_name):]
    expected_markers = {
        "Background Textures": ("_table_", "_tb_"),
        "Camera Viewpoints": ("_view_",),
        "Language Instructions": ("_language_",),
        "Light Conditions": ("_light_",),
        "Objects Layout": ("_add_", "_level"),
        "Robot Initial States": ("_view_",),
        "Sensor Noise": ("_view_", "_noise_"),
    }[record.category]
    if not any(marker in suffix for marker in expected_markers):
        raise ValueError(
            f"Unexpected {record.category} suffix for {record.key}: {suffix!r}."
        )

    raw = str(task.language)
    if record.category == "Language Instructions":
        clean = raw
        source = "libero_plus_language_perturbation"
    else:
        clean = base_instruction
        source = "vanilla_libero_task_language"
    if not raw or not clean or not base_instruction:
        raise ValueError(f"Empty instruction contract for {record.key}.")
    return LiberoPlusInstructions(
        raw=raw,
        clean=clean,
        base=base_instruction,
        source=source,
    )


def load_registry(libero_plus_root: Path, *, require_pinned: bool = True) -> list[LiberoPlusTask]:
    classification_path, task_map_path = benchmark_paths(libero_plus_root)
    if not classification_path.is_file() or not task_map_path.is_file():
        raise FileNotFoundError(
            f"LIBERO-Plus registry is incomplete under {Path(libero_plus_root).resolve()}."
        )
    actual_sha = sha256_file(classification_path)
    if require_pinned and actual_sha != CLASSIFICATION_SHA256:
        raise ValueError(
            f"LIBERO-Plus classification checksum mismatch: {actual_sha}, "
            f"expected {CLASSIFICATION_SHA256}."
        )
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    task_map = _load_task_map(task_map_path)
    records: list[LiberoPlusTask] = []
    for suite in SUITES:
        rows = classification.get(suite)
        names = task_map.get(suite)
        expected_size = SUITE_SIZES[suite]
        if not isinstance(rows, list) or not isinstance(names, list):
            raise KeyError(f"Pinned LIBERO-Plus has no complete {suite} registry.")
        if len(rows) != expected_size or len(names) != expected_size:
            raise ValueError(
                f"{suite} registry size mismatch: classification={len(rows)}, "
                f"task_map={len(names)}, expected={expected_size}."
            )
        for task_id, (row, task_name) in enumerate(zip(rows, names)):
            official_id = int(row["id"])
            if official_id != task_id + 1 or row["name"] != task_name:
                raise ValueError(
                    f"LIBERO-Plus task registry mismatch at {suite}:{task_id}: {row!r}."
                )
            category = str(row["category"])
            difficulty = row.get("difficulty_level")
            if category not in CATEGORIES or difficulty not in DIFFICULTIES:
                raise ValueError(
                    f"Unsupported classification at {suite}:{task_id}: "
                    f"category={category!r}, difficulty={difficulty!r}."
                )
            records.append(
                LiberoPlusTask(
                    suite=suite,
                    task_id=task_id,
                    official_id=official_id,
                    task_name=task_name,
                    category=category,
                    difficulty=difficulty,
                )
            )
    if len(records) != sum(SUITE_SIZES.values()):
        raise AssertionError("LIBERO-Plus registry did not produce 10,030 tasks.")
    return records


def _stratum_sort_key(key: tuple[str, str, int | None]) -> tuple[int, int, int]:
    suite, category, difficulty = key
    return SUITES.index(suite), CATEGORIES.index(category), DIFFICULTIES.index(difficulty)


def _selection_rank(record: LiberoPlusTask, seed: int) -> bytes:
    difficulty = "unclassified" if record.difficulty is None else str(record.difficulty)
    payload = "\0".join(
        (
            SELECTION_SCHEMA,
            str(int(seed)),
            record.suite,
            str(record.task_id),
            record.task_name,
            record.category,
            difficulty,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def select_stratified_tasks(
    records: Iterable[LiberoPlusTask], *, count: int, seed: int
) -> tuple[list[LiberoPlusTask], dict[tuple[str, str, int | None], int]]:
    """Select an exact lower-bounded proportional suite/category/difficulty sample."""
    groups: dict[tuple[str, str, int | None], list[LiberoPlusTask]] = defaultdict(list)
    records = list(records)
    for record in records:
        groups[(record.suite, record.category, record.difficulty)].append(record)
    if count < len(groups) or count > len(records):
        raise ValueError(
            f"Selection count {count} must be between the {len(groups)} nonempty strata "
            f"and {len(records)} total tasks."
        )

    total = len(records)
    ideal = {key: count * len(items) / total for key, items in groups.items()}
    quotas = {key: max(1, math.floor(ideal[key])) for key in groups}
    allocated = sum(quotas.values())
    if allocated > count:
        raise ValueError("Minimum-one stratum allocation exceeds requested sample count.")
    remaining = count - allocated
    candidates = sorted(
        groups,
        key=lambda key: (-(ideal[key] - math.floor(ideal[key])), _stratum_sort_key(key)),
    )
    while remaining:
        progressed = False
        for key in candidates:
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise AssertionError("Could not allocate all requested LIBERO-Plus samples.")

    selected: list[LiberoPlusTask] = []
    for key in sorted(groups, key=_stratum_sort_key):
        ranked = sorted(
            groups[key],
            key=lambda record: (_selection_rank(record, seed), record.task_id),
        )
        selected.extend(ranked[: quotas[key]])
    selected.sort(key=lambda record: (SUITES.index(record.suite), record.task_id))
    if len(selected) != count or len({record.key for record in selected}) != count:
        raise AssertionError("LIBERO-Plus selection is not exact and unique.")
    return selected, quotas


def _difficulty_text(value: int | None) -> str:
    return "unclassified" if value is None else str(value)


def manifest_data_bytes(records: Iterable[LiberoPlusTask]) -> bytes:
    lines = ["\t".join(MANIFEST_COLUMNS)]
    for record in records:
        lines.append(
            "\t".join(
                (
                    record.suite,
                    str(record.task_id),
                    str(record.official_id),
                    record.task_name,
                    record.category,
                    _difficulty_text(record.difficulty),
                )
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_manifest(
    records: list[LiberoPlusTask],
    quotas: dict[tuple[str, str, int | None], int],
    *,
    seed: int,
) -> str:
    data = manifest_data_bytes(records)
    manifest_id = hashlib.sha256(data).hexdigest()
    allocation = {
        f"{suite}|{category}|{_difficulty_text(difficulty)}": quotas[(suite, category, difficulty)]
        for suite, category, difficulty in sorted(quotas, key=_stratum_sort_key)
    }
    header = [
        f"# schema: {SELECTION_SCHEMA}",
        f"# seed: {int(seed)}",
        f"# count: {len(records)}",
        f"# libero_plus_revision: {LIBERO_PLUS_REV}",
        f"# classification_sha256: {CLASSIFICATION_SHA256}",
        f"# manifest_id: {manifest_id}",
        "# allocation: " + json.dumps(allocation, sort_keys=True, separators=(",", ":")),
    ]
    return "\n".join(header) + "\n" + data.decode("utf-8")


def load_task_manifest(path: Path) -> tuple[list[LiberoPlusTask], dict[str, str]]:
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if ":" in line:
                key, value = line[1:].split(":", 1)
                metadata[key.strip()] = value.strip()
        elif line.strip():
            data_lines.append(line)
    if not data_lines or tuple(data_lines[0].split("\t")) != MANIFEST_COLUMNS:
        raise ValueError(f"Invalid LIBERO-Plus manifest columns in {path}.")
    expected_metadata = {
        "schema": SELECTION_SCHEMA,
        "libero_plus_revision": LIBERO_PLUS_REV,
        "classification_sha256": CLASSIFICATION_SHA256,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"LIBERO-Plus manifest {key}={metadata.get(key)!r}, expected {value!r}."
            )
    try:
        int(metadata["seed"])
        declared_count = int(metadata["count"])
        declared_allocation = json.loads(metadata["allocation"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid LIBERO-Plus manifest metadata in {path}.") from exc
    records = []
    for line in data_lines[1:]:
        values = line.split("\t")
        if len(values) != len(MANIFEST_COLUMNS):
            raise ValueError(f"Invalid LIBERO-Plus manifest row: {line!r}.")
        suite, task_id, official_id, task_name, category, difficulty = values
        records.append(
            LiberoPlusTask(
                suite=suite,
                task_id=int(task_id),
                official_id=int(official_id),
                task_name=task_name,
                category=category,
                difficulty=None if difficulty == "unclassified" else int(difficulty),
            )
        )
    expected_id = metadata.get("manifest_id")
    actual_id = hashlib.sha256(manifest_data_bytes(records)).hexdigest()
    if expected_id != actual_id:
        raise ValueError(
            f"LIBERO-Plus manifest digest mismatch in {path}: {actual_id}, expected {expected_id}."
        )
    if len(records) != len({record.key for record in records}):
        raise ValueError(f"LIBERO-Plus manifest {path} contains duplicate task IDs.")
    if declared_count != len(records):
        raise ValueError(
            f"LIBERO-Plus manifest declares {declared_count} tasks but contains {len(records)}."
        )
    actual_allocation = Counter(
        f"{record.suite}|{record.category}|{_difficulty_text(record.difficulty)}"
        for record in records
    )
    if declared_allocation != dict(actual_allocation):
        raise ValueError(f"LIBERO-Plus manifest allocation metadata is stale in {path}.")
    return records, metadata


def validate_manifest_against_registry(
    selected: Iterable[LiberoPlusTask], registry: Iterable[LiberoPlusTask]
) -> None:
    expected = {record.key: record for record in registry}
    for record in selected:
        if expected.get(record.key) != record:
            raise ValueError(
                f"Manifest task {record.key} does not exactly match the pinned registry."
            )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def assert_libero_modules_from(libero_plus_root: Path) -> None:
    root = Path(libero_plus_root).expanduser().resolve()
    checked = []
    for module_name in ("libero", "libero.libero.benchmark", "libero.libero.envs.env_wrapper"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file:
            origins = [Path(module_file).resolve()]
        else:
            origins = [Path(value).resolve() for value in getattr(module, "__path__", ())]
            if not origins:
                raise RuntimeError(f"Imported module {module_name} has no filesystem origin.")
        for resolved in origins:
            checked.append((module_name, resolved))
            if not _is_relative_to(resolved, root):
                raise RuntimeError(
                    f"Imported {module_name} from {resolved}, not intended LIBERO-Plus checkout "
                    f"{root}. Start a fresh process with LIBERO-Plus first on PYTHONPATH."
                )
    if "libero" in sys.modules and not checked:
        raise RuntimeError("Could not validate the already-imported libero module origin.")


def configure_libero_plus(
    libero_plus_root: Path, *, config_dir: Path, require_assets: bool = True
):
    """Configure and import the pinned fork, failing on namespace shadowing."""
    root = Path(libero_plus_root).expanduser().resolve()
    if not (root / ".git").is_dir():
        raise FileNotFoundError(f"LIBERO-Plus must be a Git checkout: {root}.")
    actual_rev = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_rev != LIBERO_PLUS_REV:
        raise RuntimeError(f"LIBERO-Plus is at {actual_rev}, expected {LIBERO_PLUS_REV}.")
    load_registry(root, require_pinned=True)
    benchmark_root = root / "libero" / "libero"
    if require_assets and not (benchmark_root / "assets" / "textures").is_dir():
        raise FileNotFoundError(
            f"LIBERO-Plus perturbation assets are missing under {benchmark_root / 'assets'}."
        )

    assert_libero_modules_from(root)
    config_dir = Path(config_dir).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                f"assets: {benchmark_root / 'assets'}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"benchmark_root: {benchmark_root}",
                f"datasets: {root / 'libero' / 'datasets'}",
                f"init_states: {benchmark_root / 'init_files'}",
                "",
            )
        ),
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != root]
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    benchmark = importlib.import_module("libero.libero.benchmark")
    importlib.import_module("libero.libero.envs.env_wrapper")
    assert_libero_modules_from(root)
    return benchmark


def get_libero_plus_env(task, resolution: int, seed: int):
    """Construct a fork environment with the string path its variant parser requires."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_path = str(
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=int(resolution),
        camera_widths=int(resolution),
    )
    env.seed(int(seed))
    return env, task.language


def instantiate_suite(benchmark_module, suite_name: str):
    """Instantiate a suite without the fork printing thousands of task IDs."""
    if suite_name not in SUITES:
        raise KeyError(f"Unsupported official LIBERO-Plus suite: {suite_name}.")
    with redirect_stdout(io.StringIO()):
        suite = benchmark_module.get_benchmark_dict()[suite_name]()
    if int(suite.n_tasks) != SUITE_SIZES[suite_name]:
        raise ValueError(
            f"Runtime suite {suite_name} has {suite.n_tasks} tasks, expected {SUITE_SIZES[suite_name]}."
        )
    return suite


def get_libero_plus_init_states(task_suite, task_id: int):
    """Use the official variant resolver with PyTorch 2.6+ trusted-local loading."""
    import torch

    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        return task_suite.get_task_init_states(int(task_id))
    finally:
        torch.load = original_load


def aggregate_results(results: Iterable[dict]) -> dict:
    """Aggregate completed policy episodes; execution failures are not policy failures."""
    buckets: dict[str, Counter] = defaultdict(Counter)
    execution_failures = []
    for result in results:
        if result.get("status") != "completed":
            execution_failures.append(result)
            continue
        success = int(bool(result["success"]))
        difficulty = result.get("difficulty")
        difficulty_text = "unclassified" if difficulty is None else str(difficulty)
        labels = {
            "overall": "all",
            "by_suite": result["suite"],
            "by_category": result["category"],
            "by_difficulty": difficulty_text,
            "by_stratum": "|".join((result["suite"], result["category"], difficulty_text)),
        }
        for group, label in labels.items():
            buckets[f"{group}\0{label}"]["episodes"] += 1
            buckets[f"{group}\0{label}"]["successes"] += success

    output: dict[str, dict] = defaultdict(dict)
    for composite, counts in sorted(buckets.items()):
        group, label = composite.split("\0", 1)
        episodes = counts["episodes"]
        output[group][label] = {
            "episodes": episodes,
            "successes": counts["successes"],
            "success_rate": counts["successes"] / episodes if episodes else 0.0,
        }
    output["execution_failures"] = execution_failures
    return dict(output)
