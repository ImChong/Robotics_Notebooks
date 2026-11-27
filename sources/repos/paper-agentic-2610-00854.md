# embodied-agent-arena/embodied-agent-arena

- **官方仓库：** [https://github.com/embodied-agent-arena/embodied-agent-arena](https://github.com/embodied-agent-arena/embodied-agent-arena)
- **项目页：** [https://embodied-agent-arena.github.io/embodied-agent-arena/](https://embodied-agent-arena.github.io/embodied-agent-arena/)
- **配套数据集：** [uuu-Quant/Embodied-Agent-Arena](https://huggingface.co/datasets/uuu-Quant/Embodied-Agent-Arena)（README 固定修订版 `5f729375eb35163a024625dd8ea72ac57a427a24`）
- **开源协议：** MIT License
- **默认分支：** `main`（快照时 Stars：119）
- **关联论文：** [Are Frontier VLM Agents Ready to Be Robot Generalists? An Empirical Study with the Embodied Agent Arena](../papers/paper-agentic-2610-00854.md)（arXiv:2610.00854v3）
- **核对边界：** 静态代码与文档核对（基于远程仓库文件样本与 `README.md`），未本地运行

## 仓库定位

`embodied-agent-arena/embodied-agent-arena` 是论文 *Embodied Agent Arena* 的官方代码库与评测发布仓库，提供统一极简评测框架（`arena` CLI）、1,000 个案例的基准配置与环境锁文件、Hugging Face 数据集拉取与哈希校验脚本，以及 10 款 VLM 智能体的评测结果 CSV 与分析工具。

## 已核对目录与入口文件（来自仓库文件样本与 README）

### 1. 文档与核心配置
- `README.md`：包含十大模型交叉对比表、五大域分布说明与 Quick Start 运行指南。
- `DESIGN.md`、`CITATION.cff`、`NOTICE`、`LICENSE`（MIT）。
- `.env.example`：API 凭据环境变量模板，配合 `--env-file .env` 使用。
- `configs/batch.example.json`：多模型批量评测配置文件示例。
- `configs/operation_environment_lock.json`、`configs/operation_runtime_matrix.json`：操作类交互环境的依赖锁定与运行时矩阵配置。
- `configs/environments/*.json` / `*.txt`：各环境配置（覆盖 `behavior1k`、`calvin`、`capx`、`capx_libero_pro`、`cliport`、`humanclaw`、`maniskill`、`pukun`、`rlbench`、`robocasa`、`robotwin2`、`robowits`、`vimabench`、`vlabench`）。
- `configs/patches/`：环境兼容性补丁（如 `behavior1k/archived-asset-compatibility.patch`、`capx/base-inplace-rotation.patch`、`robodojo/isaaclab-offline-urdf-importer.patch`、`robodojo/curobo-submodule-version-fallback.patch` 等）。
- `configs/capx/`、`configs/capx_minimal_closure.json`、`configs/capx_runtime_config.yaml`、`configs/robodojo_case_closure.json`、`configs/nvidia-egl-vendor.json`。

### 2. 基准任务清单（`benchmarks/`）
- **非操作类（`benchmarks/non_operation/`）**：`alfworld`、`esi`（含 `primitives/primitive_cards.yaml`）、`mmsi_bench`、`scienceworld`、`spatialclaw`，均含 `benchmark.yaml`、`cases.yaml`（部分含 `environment.json`）。
- **操作类（`benchmarks/operation/`）**：`behavior1k`、`calvin`、`capx`、`cliport`、`maniskill`、`rlbench`、`robocasa`、`robocasa365`、`robodojo`、`robotwin2`、`robowits`、`vimabench`、`vlabench`，均含 `benchmark.yaml`、`cases.yaml`、`environment.json`（及部分 `environment.yml`）。
- **策略契约测试（`benchmarks/policy_contract/`）**：`lerobot`、`octo`、`openpi`、`openvla` 的 `benchmark.yaml` 与 `cases.yaml`。

### 3. 已记录的运行入口（来自官方 README Quick Start）
- **依赖安装（Linux, Python ≥ 3.10）**：
  `python -m pip install -e '.[offline,test,hub]'`
- **数据集拉取与校验**：
  `python scripts/fetch_data.py --repo-id uuu-Quant/Embodied-Agent-Arena --revision 5f729375eb35163a024625dd8ea72ac57a427a24 --output ../data`
  `arena validate --data-root ../data --hashes`
  `arena list --data-root ../data`
- **单基准评测**：
  `arena run --data-root ../data --benchmark w1_vlm_depth -- --provider openai-compatible --model MODEL_ID --env-file .env --workers 1 --retry-http --max-http-rounds 3 --output-dir outputs/evaluation`
- **多模型批量评测**：
  `arena batch --config configs/batch.example.json --output-dir outputs/batch --execute`
- **公开结果与数据**：`public/data/model-comparison.csv`、`docs/INSTALL.md`、`docs/RESULTS.md`。

## 对知识库的映射

- 实体节点：[Are Frontier VLM Agents Ready to Be Robot Generalists? An Empirical Study with the Embodied Agent Arena](../../wiki/entities/paper-agentic-2610-00854.md)
