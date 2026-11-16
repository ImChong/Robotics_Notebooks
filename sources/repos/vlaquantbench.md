# VLAQuantBench 官方仓库

> 来源归档（repo）

- **标题：** VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models
- **类型：** repo
- **链接：** <https://github.com/jiuyixu25/VLAQuantBench>
- **arXiv：** <https://arxiv.org/abs/2609.25376>
- **协议：** MIT License
- **入库日期：** 2026-10-18
- **一句话说明：** 面向 VLA 闭环训练后量化的统一评测 harness 与实验结果存档，覆盖 7 个 VLA 适配器与 4 个仿真基准，附带实际硬件 kernel 延迟/显存分析、动作保真度重放及作用域完整性校验脚本。
- **沉淀到 wiki：** [`wiki/entities/paper-vlaquantbench.md`](../../wiki/entities/paper-vlaquantbench.md)
- **通读：** 已完成
- **核对范围：** 通读官方仓库根目录 `_tree.txt`、`README.md`（L1-200），并核对运行入口与核心模块 `vlaquantbench/cli.py`（L1-477 主体与 `cmd_run` / `cmd_inspect` / `cmd_matrix` / `cmd_summarize` / `apply_quantization` / `_collect_act_calib`）、`vlaquantbench/components.py`（L120-178）、`vlaquantbench/quant/apply.py`（L1-263）、`vlaquantbench/methods/__init__.py`（L50-104）、`vlaquantbench/benchmarks/base.py`（L39-138）、`vlaquantbench/benchmarks/libero.py`（L80-278）以及复现脚本索引（`scripts/verify_cells.py`、`scripts/profile_kernels.py`、`scripts/action_fidelity.py`、`scripts/replay_pi05_subsets.py`）。

## 开源状态

- **已开源**（2026-11-14 源码核查）：仓库以 MIT License 完整开源可运行代码，不仅包含论文主分析的 409 runs（94,574 episodes）记录，还公开了扩展后的 **417 个配置单元、105,868 条 JSONL episode 记录**（含 `results/summary.csv`、`results/latency/`、`results/fidelity/`、`results/replay/`、`diagnostics/v2/`）。

## 架构与可运行模块核查

- **CLI 入口（`vlaquantbench/cli.py`）：** 提供 `vqb inspect`、`vqb run`、`vqb summarize`、`vqb matrix`、`vqb presets`、`vqb serve` 等子命令；`cmd_run` 解析模型配置（`configs/models/<model>.yaml`）、初始化 `BenchmarkRunner` 并调用 `runner.configure_for(adapter)` 继承模型官方评测协议。
- **组件解耦与作用域路由（`vlaquantbench/components.py`）：** 定义 `ComponentMap` 并在 `quantize_scope()` 中执行 `cmap.check_disjoint()`，防止同一线性层被多组件重复量化；支持 `e2e`、`ve`、`mp`、`llm`、`ah`、`llm+ah` 及 `--only` / `--exclude` 子层消融。
- **均匀 PTQ 与激活校准执行器（`vlaquantbench/quant/apply.py`）：** 默认对 `nn.Linear` 原地写入伪量化权重（`fake_quant_weight`），对激活安装 `register_forward_pre_hook`（确保每层选中权重与激活 hook 严格对齐）；当指定 `--act-calib` 时，先在不量化策略上跑校准 episode（默认 `episodes_from=20` 与评测态 `0-19` 隔离），将平滑因子 `calib.smooth` 折入权重并通过 `_make_calibrated_act_hook` 做逐层平滑与裁剪（`clamp`）。
- **外部 LLM PTQ 适配器（`vlaquantbench/methods/`）：** `apply_method()` 支持仅 LLM 作用域下的 `rtn`、`awq`（含 `apply_awq_fake` 与 `apply_awq_real`）、`nf4`/`int8`（`bitsandbytes`）、`smoothquant`（含 `real_w8a8` 与 `int8_gemm.W8A8Linear` 真实 kernel 测量）。
- **已实现模型与基准：** 包内 `models/` 已实现 `pi05`/`pi0`（`pi0_lerobot.py`）、`openvla_oft.py`、`openvla.py`、`xvla.py`、`cogact.py`、`internvla_m1.py`；`benchmarks/` 已实现 `libero.py`（须锁定 `mujoco==3.3.2`）、`calvin.py`、`simpler.py`、`vlabench.py`。复现实验需自行准备各模型上游依赖环境（`scripts/env/setup_*.sh`）及对应模型权重。
