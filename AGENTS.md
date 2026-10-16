This file provides guidance to AI agents when working with code in this repository.

> **中文版 → [`AGENTS.zh.md`](./AGENTS.zh.md)**
> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools (pytest, pre-commit, mypy)
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
pre-commit install                          # Run hooks on every commit
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                          # All tests (needs --extra test; many suites also need policy/env extras)
uv run pytest tests/scripts/test_lerobot_eval.py -svv          # Single file
uv run pytest tests/scripts/test_lerobot_eval.py::test_name    # Single test
DEVICE=cuda make test-end-to-end                               # E2E train/eval smoke tests → tests/outputs/
pre-commit run --all-files                                     # Lint + format (ruff, mypy, typos, bandit, prettier, gitleaks)
make doctest                                                   # Doctest examples listed in utils/documentation_tests.txt
make check-docstrings                                          # Docstring standard checks (fix variants: make fix-docstrings, make fix-doctest-list)
```

- `LEROBOT_TEST_DEVICE` env var overrides the device tests use (defaults to auto-detected).
- Hardware/GPU tests skip themselves via decorators in `tests/utils.py`: `require_cuda`, `require_cpu`, `require_hf_token`, `require_env`, `require_x86_64_kernel`, `skip_if_package_missing(...)`.
- Pytest markers `multigpu` / `multigpu_heavy` select distributed tests needing 2+ GPUs; they never run in regular CI lanes.

## Architecture (`src/lerobot/`)

Core flow: **dataset → processor → policy → training/rollout**, all wired through draccus configs and Hub persistence.

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-rollout`, `lerobot-policy-server`, `lerobot-annotate`, …), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`. `policies/act/` is the canonical simple example.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`; optional Lance backend in `lance_backend.py`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`inference/`, `remote_inference/`, `rollout/`** — Policy execution contracts and sync/async backends; async policy serving over Zenoh; robot rollout orchestration with pluggable strategies.
- **`rl/`, `rewards/`, `distributed/`, `optim/`** — RL training loop; reward classifiers (`robometer`, `sarm`, `topreward`); distributed-training runtime; optimizer/scheduler factories.
- **`annotations/`, `jobs/`** — Steerable annotation pipeline (`lerobot-annotate`) and Hugging Face Jobs helpers.
- **`streaming/`, `transport/`, `transforms/`** — Training-time dataset streaming primitives; transport backends (gRPC/Zenoh; generated `*_pb2` protobuf modules — never hand-edit or reformat them); image transforms.
- **`model/`, `common/`, `data_processing/`, `templates/`** — Kinematics; cross-cutting bridges; dataset transformation tools; model-card templates.
- **`lerobot_types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. E2E tests via `Makefile` write to `tests/outputs/`. The root `conftest.py` only customizes `--doctest-modules` collection (see `make doctest`); the suite itself is configured by `tests/conftest.py`.
- **`utils/`** — Docstring/doctest CI helper scripts (run via Makefile targets).
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit + docstring gates), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — `Dockerfile.user` (users), `Dockerfile.internal` (CI), and per-benchmark eval images (`Dockerfile.benchmark.*`) — benchmark env dependencies are painful to install locally, so prefer these images for benchmark eval.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E + docstring targets), `uv.lock`, `CONTRIBUTING.md` (PR workflow), `AI_POLICY.md` (AI usage policy), `README.md`.

## Notes

- **Docstrings**: public docstrings in `src/lerobot/` are parsed into the API reference and checked in CI. Follow the standard in `docs/source/writing_docstrings.mdx`. The coverage gate (`[tool.interrogate]` in `pyproject.toml`, `fail-under` ratchet) is raised in the same PR that documents a module — never set it to a value that fails main.
- **Mypy**: one ruleset (`[tool.mypy]` in `pyproject.toml`) checks the paths in its `files` setting, except `lerobot.rl`, the vendored `molmoact2_hf_model` and the generated `*_pb2` modules. Run it with `pre-commit run mypy --all-files`. Add type annotations when modifying code.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g. `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
- **AI contributions**: per `AI_POLICY.md`, disclose significant AI assistance in the PR description and be able to explain every change — human maintainers review everything. `CONTRIBUTING.md` also asks contributors to peer-review at least one other open PR before their own gets attention.
