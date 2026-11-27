# DorianAtSchool/robotalk（RoboTalk 官方代码、仿真扩展与复现仓库）

- **官方仓库：** https://github.com/DorianAtSchool/robotalk
- **关联论文：** [RoboTalk: Learning Multi-Robot Communication and Coordination from Multimodal Demonstrations](../../sources/papers/paper-agentic-2609-23997.md)（`arXiv:2609.23997`）
- **开源协议：** Apache-2.0（RoboTalk 代码、数据集与微调模型）；仓库内 `robocasa/` 修改自 RoboCasa365（MIT）；RoboCasa365 资产单独下载（CC BY 4.0）
- **默认分支：** `main`
- **核对范围：** 仓库根目录文件清单、`configs/` 配置清单、`docs/` 技术文档清单、`robocasa/` 场景任务扩展、以及完整 `README.md` 内的安装、快速回放、数据生成、LoRA 训练与闭环评测命令（静态代码与文档核对，未本地运行）
- **复现边界：** 静态代码核对，未本地运行；完整数据生成依赖 Google Gemini API（`google-genai`），开放权重模型闭环评测依赖 vLLM `v0.27.1` 容器（CUDA 13.0 / 驱动 580+），RoboCasa365 资产需约 10 GB 下载 / 23 GB 磁盘空间

## 仓库结构与核心模块核对

### 1. 主包与核心目录（`robotalk/`、`robocasa/`、`configs/`、`scripts/`）

- `robotalk/tasks/`：53 个双机器人厨房任务规格定义、符号世界状态（symbolic state）与并发 FSM 校验器
- `robotalk/tools/`：数据生成、LoRA 训练与闭环评测共用的工具调用接口定义（感知、导航、操作、通信、同步与让位）
- `robotalk/generation/`：Gemini-3-Flash 轨迹生成（`robotalk.generation.raw.production_cascade`）、`get_image` 观测插入与重校验（`robotalk.generation.image.cli`）以及场景重放渲染（`robotalk.generation.sweep_trajectories`）
- `robotalk/training/`：训练样本预处理（`robotalk.training.preprocess`）、数据规模子集抽取（`robotalk.training.materialize_scale_artifacts`）与多卡 LoRA 微调主入口（`robotalk.training.main`）
- `robotalk/evaluation/`：闭环仿真评测引擎（`robotalk.evaluation.live_sim_eval`，支持 `vllm`、`gemini` 与 `oracle` 回放三种后端）及自定义任务划分工具（`robotalk.evaluation.resplit_cohort`）
- `robotalk/release/`：Hugging Face 数据集导出（`robotalk.release.export_robotalk`）、本地渲染目录解包重建（`robotalk.release.materialize_training_layout`）与 Explorer Space
- `robotalk/analysis/`：论文图表分析与绘图逻辑
- `robocasa/`：支持双移动操作机器人同场及技能执行器（skill executor）的 RoboCasa365 分支修改（说明见 `docs/FORK_CHANGES.md`），配套底层修改版 [DorianAtSchool/robosuite](https://github.com/DorianAtSchool/robosuite)
- `configs/`：`configs/experiments.yaml`、`configs/splits/43_train_10_heldout.json`、`configs/splits/selection_{30,60,90,120,150}.json`、`configs/train/accelerate_multigpu.yaml`、`configs/eval/fixed_live_sim_cohort.json`、`configs/eval/oracle_smoke.json`、`configs/eval/scene_compatibility_cache_v2.json`
- `scripts/`：端到端图表复现脚本 `scripts/reproduce.py` 与 SLURM 调度脚本 `scripts/slurm/reproduce.sbatch`
- `docs/`：`docs/paper/RoboTalk_arxiv.pdf` 及技术规范文档（`docs/concurrency_semantics.md`、`docs/tool_interface.md`、`docs/data_format.md`、`docs/evaluation.md`、`docs/FORK_CHANGES.md`）

### 2. 官方验证过的 CLI 运行入口（来自 `README.md`）

- **环境初始化与资产下载（Linux + Python 3.11 + `uv`）：**
  - `uv sync --all-extras`
  - `python -m robocasa.scripts.setup_macros`
  - `python -m robocasa.scripts.download_kitchen_assets`
- **数据集解包与 Oracle 仿真冒烟回放：**
  - `python -m robotalk.release.materialize_training_layout --hf-dir data/hf --output data/robotalk_rendered --tasks prepare_coffee`
  - `python -m robotalk.evaluation.live_sim_eval --backend oracle --manifest configs/eval/oracle_smoke.json --tasks prepare_coffee --dataset-root data/robotalk_rendered --output-dir outputs/oracle`
- **三阶段合成新轨迹：**
  - `python -m robotalk.generation.raw.production_cascade --task PrepareCoffee --num-runs 150 --output-root data/my_raw`
  - `python -m robotalk.generation.image.cli --revalidate --dataset data/my_raw/prepare_coffee/summary.json --output-dataset data/my_image/prepare_coffee/summary.json`
  - `python -m robotalk.generation.sweep_trajectories --input-dir data/my_image --output-dir data/my_rendered`
- **预处理与 LoRA 微调：**
  - `python -m robotalk.training.preprocess --output-dir outputs/preprocessed/my_run`
  - `accelerate launch --config_file configs/train/accelerate_multigpu.yaml --num_processes 4 -m robotalk.training.main --preprocessed-data-dir outputs/preprocessed/my_run --model-name-or-path Qwen/Qwen3-VL-8B-Instruct --output-dir outputs/train/my_run`（加 `--train-reasoning` 开启 `<think>...</think>` 推理监督）
- **论文图表复现与新模型闭环评测：**
  - `python scripts/reproduce.py fig3|fig5|fig6|fig7`
  - `python scripts/reproduce.py evaluate --adapter outputs/train/my_run [--base Qwen/Qwen3-VL-8B-Thinking --profile thinking]`

### 3. 关联外部资产

- **数据集（v1.1）：** [DorianAtSchool/RoboTalk](https://huggingface.co/datasets/DorianAtSchool/RoboTalk)（包含 `trajectories` 与 `ticks` 两张表，v1.1 替换了 v1.0 中 95 条仿真重放失败的轨迹，占 1.2%；已发布模型权重基于 v1.0 训练）
- **模型权重与可视化：** [Hugging Face 模型合集](https://huggingface.co/DorianAtSchool) · [RoboTalk Explorer](https://huggingface.co/spaces/DorianAtSchool/RoboTalk-Explorer)
