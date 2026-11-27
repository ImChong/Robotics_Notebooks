# Ropedia/egotools（EgoTools 官方源码仓库）

- **仓库地址：** https://github.com/Ropedia/egotools （论文内写作 `github.com/Ropedia/EgoTools`，GitHub 解析全名 `Ropedia/egotools`）
- **默认分支：** `main`
- **开源许可：** GitHub API 未标明单一仓级许可证；README 标注「The project license is pending」，第三方依赖许可记录在 `docs/THIRD_PARTY.md`（含 `third_party/LICENSE-VLMEvalKit`）
- **关注度（核查快照）：** 16 stars
- **对应论文 / 项目：** [arXiv:2609.39378](https://arxiv.org/abs/2609.39378) · [项目页](https://ropedia.github.io/egotools) · [数据集](https://huggingface.co/datasets/ropedia-ai/egotools-data) · [EgoTools-8B 权重](https://huggingface.co/ropedia-ai/egotools-8b)
- **核查边界：** 静态代码核对（基于 GitHub 目录样本 `files_sample` 与 `README.md` 预览），未本地安装依赖或运行；仓库 README 明确标注完整论文实验尚未以当前公开发布资源完全复现。

## 仓库结构与可运行入口（已核实）

- **包定义与环境配置：**
  - `pyproject.toml`：定义 Python 包 `egotools` 及命令行入口（`egotools-prepare-benchmark`、`egotools-evaluate`、`egotools-prepare-sft`）。
  - `scripts/setup_eval.sh`、`configs/evaluation/environment.yml`、`configs/evaluation/constraints.txt`：构建 `egotools_eval` conda 评测环境；`scripts/verify_eval.py` 用于评测环境自检。
  - `configs/training/requirements-cu124.txt`、`scripts/train.sh`：基于 MS-Swift 的 8 卡 CUDA 12.4 训练脚本与依赖。
  - `configs/resources.yaml`、`configs/resources.preview.yaml`：数据与模型资源配置；相关说明文档在 `docs/DATA.md`、`docs/DATA_PROCESSING.md`、`docs/EVALUATION.md`、`docs/TRAINING.md`。
- **数据构建与源视频切分（`src/egotools/data/`）：**
  - `prepare.py` / `manifest.py` / `resources.py`：根据 `manifest.tsv`（相对视频路径、问题、A–H 选项、答案、稳定问题/源视频 ID）与媒体目录生成本地基准目录。
  - `data/prepare_sft.py`：将 `train.jsonl` 转为 MS-Swift 所需的 `train.swift.jsonl`。
  - `data/splits/build_train_temporal_split.py`、`data/splits/filter_train_eval_overlap.py`：训练时间切分与源视频防泄露过滤。
  - `data/benchmark/build_benchmark.py`、`extract_final_reviewed_qa.py`、`normalize_to_8choice.py`、`assign_benchmark_research_tracks.py`、`qa_common.py`、`_resolvers.py`：基准 8 选 1 规整、赛道分配与 QA 构建工具。
- **评测与打分管线（`src/egotools/evaluation/`）：**
  - `runner.py`、`inference.py`、`_torchrun_entry.py`：多卡或单卡推理调度入口；默认 `Qwen3-VL-8B-Instruct` 与 `EgoTools-8B` 使用 `--nframe 64`，Thinking 模型支持 512 帧，Gemini 按 1 FPS 评测。
  - `adapters/vlmevalkit.py`、`datasets/egotools.py`、`data.py`、`models.py`：对接 VLMEvalKit 评测适配器与模型加载。
  - `metrics.py`、`answers.py`、`score.py`：答案抽取、覆盖率与四赛道/总准确率计算，输出 `results.json`。
- **单元测试（`tests/`）：**
  - 包含 `test_data_processing.py`、`test_evaluation_protocol.py`、`test_prepare_benchmark.py`、`test_prepare_sft.py`、`test_public_utilities.py`、`test_run_eval_model_download.py`、`test_scoring.py`、`test_training.py`、`test_vlmeval_register.py`。
- **项目站静态产物（`website/`）：**
  - Next.js 导出的项目网站静态文件（含 `.github/workflows/pages.yml`）。

## 关联页面

- [EgoTools 实体页](../../wiki/entities/paper-agentic-2609-39378.md)
- [EgoTools 原文核对归档](../papers/paper-agentic-2609-39378.md)
