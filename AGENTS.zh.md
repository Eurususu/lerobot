本文件为 AI 智能体在此仓库中工作时提供指引。

> **English version → [`AGENTS.md`](./AGENTS.md)**
> **面向用户的帮助 → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)**（SO-101 装配、录制、策略选型、训练时长、评测 —— 附可直接复制的命令）。

## 项目概览

LeRobot 是一个基于 PyTorch 的真实世界机器人库，提供数据集、预训练策略，以及训练、评测、数据采集和机器人控制工具。它与 Hugging Face Hub 集成，用于模型/数据集共享。

## 技术栈

Python 3.12+ · PyTorch · Hugging Face（datasets、Hub、accelerate）· draccus（配置/CLI）· Gymnasium（环境）· uv（包管理）

## 开发环境搭建

```bash
uv sync --locked                            # 基础依赖
uv sync --locked --extra test --extra dev   # 测试 + 开发工具（pytest、pre-commit、mypy）
uv sync --locked --extra all                # 全部依赖
git lfs install && git lfs pull             # 测试工件
pre-commit install                          # 每次提交时自动运行钩子
```

## 常用命令

```bash
uv run pytest tests -svv --maxfail=10                          # 全部测试（需要 --extra test；许多测试套件还需要策略/环境的 extras）
uv run pytest tests/scripts/test_lerobot_eval.py -svv          # 单个文件
uv run pytest tests/scripts/test_lerobot_eval.py::test_name    # 单个测试
DEVICE=cuda make test-end-to-end                               # 端到端训练/评测冒烟测试 → tests/outputs/
pre-commit run --all-files                                     # 代码检查 + 格式化（ruff、mypy、typos、bandit、prettier、gitleaks）
make doctest                                                   # 运行 utils/documentation_tests.txt 中列出的文档字符串示例
make check-docstrings                                          # 文档字符串规范检查（修复变体：make fix-docstrings、make fix-doctest-list）
```

- 环境变量 `LEROBOT_TEST_DEVICE` 可覆盖测试使用的设备（默认自动检测）。
- 硬件/GPU 测试通过 `tests/utils.py` 中的装饰器自动跳过：`require_cuda`、`require_cpu`、`require_hf_token`、`require_env`、`require_x86_64_kernel`、`skip_if_package_missing(...)`。
- Pytest 标记 `multigpu` / `multigpu_heavy` 用于筛选需要 2 块以上 GPU 的分布式测试；它们不会在常规 CI 通道中运行。

## 架构（`src/lerobot/`）

核心流程：**数据集 → 处理器（processor）→ 策略（policy）→ 训练/rollout**，全部通过 draccus 配置和 Hub 持久化机制串联。

- **`scripts/`** — CLI 入口（`lerobot-train`、`lerobot-eval`、`lerobot-record`、`lerobot-rollout`、`lerobot-policy-server`、`lerobot-annotate` 等），映射定义在 `pyproject.toml [project.scripts]`。
- **`configs/`** — 由 draccus 解析的 dataclass 配置。`train.py` 中的 `TrainPipelineConfig` 是顶层配置。`policies.py` 中是 `PreTrainedConfig` 基类。多态通过 `draccus.ChoiceRegistry` 与 `@register_subclass("name")` 装饰器实现。
- **`policies/`** — 每个策略独占一个子目录。全部继承 `pretrained.py` 中的 `PreTrainedPolicy`（`nn.Module` + `HubMixin`）。`factory.py` 提供懒加载工厂。`policies/act/` 是最典型的入门示例。
- **`processor/`** — 数据变换流水线。`ProcessorStep` 为带注册机制的基类。`DataProcessorPipeline` / `PolicyProcessorPipeline` 负责串联各步骤。
- **`datasets/`** — `LeRobotDataset`（episode 感知采样 + 视频解码）和 `LeRobotDatasetMetadata`；可选的 Lance 后端位于 `lance_backend.py`。
- **`envs/`** — `EnvConfig` 基类位于 `configs.py`，工厂位于 `factory.py`。每个环境子类需定义 `gym_kwargs` 和 `create_envs()`。
- **`robots/`、`motors/`、`cameras/`、`teleoperators/`** — 硬件抽象层。
- **`inference/`、`remote_inference/`、`rollout/`** — 策略执行契约与同步/异步后端；基于 Zenoh 的异步策略服务；带可插拔策略的机器人 rollout 编排。
- **`rl/`、`rewards/`、`distributed/`、`optim/`** — 强化学习训练循环；奖励分类器（`robometer`、`sarm`、`topreward`）；分布式训练运行时；优化器/调度器工厂。
- **`annotations/`、`jobs/`** — 可引导的标注流水线（`lerobot-annotate`）和 Hugging Face Jobs 辅助工具。
- **`streaming/`、`transport/`、`transforms/`** — 训练时数据集流式传输原语；传输后端（gRPC/Zenoh；生成的 `*_pb2` protobuf 模块 —— 切勿手动编辑或重新格式化）；图像变换。
- **`model/`、`common/`、`data_processing/`、`templates/`** — 运动学；跨模块通用桥接；数据集变换工具；模型卡片模板。
- **`lerobot_types.py`** 与 **`configs/types.py`** — 核心类型别名与特征类型定义。

## 仓库结构（`src/` 之外）

- **`tests/`** — 按模块组织的 pytest 测试套件。fixture 位于 `tests/fixtures/`，mock 位于 `tests/mocks/`。通过 Makefile 运行的端到端测试写入 `tests/outputs/`。根目录的 `conftest.py` 仅定制 `--doctest-modules` 的收集行为（见 `make doctest`）；测试套件本身由 `tests/conftest.py` 配置。
- **`utils/`** — 文档字符串/文档测试的 CI 辅助脚本（通过 Makefile 目标运行）。
- **`.github/workflows/`** — CI：`quality.yml`（pre-commit + 文档字符串门禁）、`fast_tests.yml`（基础依赖，每个 PR 必跑）、`full_tests.yml`（全部 extras + 端到端 + GPU，合并后运行）、`latest_deps_tests.yml`（每日锁定文件升级）、`security.yml`（TruffleHog）、`release.yml`（打 tag 时发布到 PyPI）。
- **`docs/source/`** — Hugging Face 文档（`.mdx` 文件）。含各策略 README、硬件指南、教程。通过 `docs-requirements.txt` 和 CI 工作流单独构建。
- **`examples/`** — 按用例组织的端用户教程与脚本（数据集创建、训练、硬件搭建）。
- **`docker/`** — `Dockerfile.user`（用户）、`Dockerfile.internal`（CI），以及各评测基准镜像（`Dockerfile.benchmark.*`）—— 基准环境的依赖在本地安装非常麻烦，做基准评测时优先使用这些镜像。
- **根目录文件**：`pyproject.toml`（依赖、构建、工具配置的唯一事实来源）、`Makefile`（端到端 + 文档字符串目标）、`uv.lock`、`CONTRIBUTING.md`（PR 流程）、`AI_POLICY.md`（AI 使用政策）、`README.md`。

## 注意事项

- **文档字符串**：`src/lerobot/` 中的公开文档字符串会被解析生成 API 参考文档，并在 CI 中检查。请遵循 `docs/source/writing_docstrings.mdx` 中的规范。覆盖率门禁（`pyproject.toml` 中的 `[tool.interrogate]`，`fail-under` 棘轮值）必须在为模块补文档的同一个 PR 中上调 —— 切勿设为会让 main 分支失败的值。
- **Mypy**：单一规则集（`pyproject.toml` 中的 `[tool.mypy]`）检查其 `files` 设置中的路径，但 `lerobot.rl`、内嵌的 `molmoact2_hf_model` 和生成的 `*_pb2` 模块除外。用 `pre-commit run mypy --all-files` 运行。修改代码时请添加类型标注。
- **导入**：优先使用顶层导入；同一模块内的相邻文件用相对导入（`from .sibling import X`），跨模块用绝对导入（`from lerobot.module import X`）。
- **可选依赖**：许多策略、环境和机器人位于 extras 之后（如 `lerobot[aloha]`，见 `pyproject.toml`）。可选导入的写法：模块顶部用 `TYPE_CHECKING or _foo_available` 守卫 + 使用处调用 `require_package(...)` 检查。复用 `utils/import_utils.py` 中的 `_foo_available` 标志；不要自行调用 `is_package_available`。
- **视频解码**：数据集可将观测存储为视频文件。`LeRobotDataset` 负责帧提取，但运行测试需要安装 ffmpeg。
- **优先使用 `uv run`** 执行 Python 命令（不要用裸 `python` 或 `pip`）。
- **AI 贡献**：根据 `AI_POLICY.md`，若 AI 工具提供了实质性帮助，须在 PR 描述中声明，且能够解释每一处改动 —— 人类维护者会逐行审查。`CONTRIBUTING.md` 还要求贡献者在自己的 PR 获得关注之前，至少参与评审另一个开放的 PR。
