# X-Planner 官方仓库

> 来源归档（repo）

- **标题：** X-Planner: Event-Structured Task Planning for Embodied Intelligence
- **类型：** repo
- **组织 / 作者：** X-Square-Robot
- **代码：** <https://github.com/X-Square-Robot/Xplanner>
- **论文：** <https://arxiv.org/abs/2609.25187>
- **项目页：** <https://x-square-robot.github.io/Xplanner/>
- **权重 / 数据集：** <https://huggingface.co/x-square-robot/X-Planner-9B-0916>（Apache-2.0）；<https://huggingface.co/datasets/x-square-robot/xplanner-benchmark>
- **许可证：** MIT（代码）
- **入库日期：** 2026-10-18
- **核对范围：** 全仓通读 `_tree.txt`、`README.md`、`scripts/train/train_qwen35_sft.sh`、`scripts/train/train_event_planner.sh`、`scripts/inference/run_event_planner.py`、`x_planner/trainer/launcher.py`、`x_planner/data/event_states/{schema.py,inference.py,dataset.py}`、`benchmarks/real_robot/README.md`，并检索全目录 `x_planner/`
- **通读：** 已完成
- **一句话说明：** 包含基于 Qwen3.5-VL（支持 DINOv3 视觉塔及 Cut Cross-Entropy）的分布式 SFT 训练框架、V5 事件状态协议数据/校验与事件推理入口，以及 lmms-eval/具身基准评测脚本；公开版本含多项外部依赖与未开放模块边界。

## 开源状态与开放边界（2026-11-14 源码核查）

- **结论：** **部分开源 — 核心 SFT/事件规划训练、V5 事件推理与 HF 9B 检查点已开放，存在外部数据集后端断层与下游执行模块缺失**

| 模块 | 开放状态 | 核查证据 |
|------|----------|----------|
| Qwen3.5-VL SFT 训练栈 | **已开源** | `scripts/train/train_qwen35_sft.sh` → `x_planner.trainer.launcher`（DeepSpeed ZeRO-1、DINOv3 可选视觉塔、CCE 省显存交叉熵） |
| V5 事件状态协议与推理 | **已开源（需外部库配合）** | `x_planner/data/event_states/schema.py`（5 类事件状态、15 类失效编码 `1.1`–`8.1`）、`inference.py`（`Generator` + `_InitialPlanStructuralStoppingCriteria`）、`scripts/inference/run_event_planner.py` |
| 事件规划端到端训练脚本 | **存在外部后端阻断与门控** | README 明示 2026-09-10 公开的 `xDataset/main` 缺少 JSONL/Qwen3.5 processor（导致 `validate_backend.py` 提前退出）；`scripts/train/train_event_planner.sh train` 直接返回退出码 4（仅放开 `unit` / `prepare` / `smoke-*` / `overfit-ddp3` / `infer` 等有界验证） |
| Staircase 潜推理与下游世界–动作策略 | **未在仓库体现** | 检索全部 `x_planner/` 未见 Staircase Decoding MoT 分支、`L_CoT` 潜推理或 `M_T P_Q` 下游策略代码 |
| 真机评估脚本 | **仅纸面记录，无代码** | `benchmarks/real_robot/` 仅存放与论文对应的结果 JSON，未发布机器人硬件驱动或闭环评估器 |

## 仓库关键脚本与入口对照

| 路径 | 职责 |
|------|------|
| `scripts/train/train_qwen35_sft.sh` | 通过 `torchrun` 调用 `x_planner.trainer.launcher` 启动分布式 SFT |
| `scripts/train/train_event_planner.sh` | 封装 V5 事件数据的 `prepare`、`data-smoke`、`smoke-single`、`smoke-ddp3/4`、`infer` 等子命令 |
| `scripts/inference/run_event_planner.py` | 调用 `x_planner.data.event_states.inference` 读取快照并推理事件 JSON |
| `x_planner/data/event_states/schema.py` | 定义 V5 样本规范、15 类 failure tag、JSON 解析与校验契约 |
| `scripts/evaluation/run_lmms_eval.sh` / `run_embodied_benchmarks.sh` | 通用 VLM 与具身推理评测入口 |

## 对 wiki 的映射

- [paper-x-planner-event-structured-task-planning](../../wiki/entities/paper-x-planner-event-structured-task-planning.md)
- [论文摘录](../papers/x_planner_event_structured_task_planning_arxiv_2609_25187.md)
