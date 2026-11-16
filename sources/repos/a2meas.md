# A²-MEAS 官方代码：推理与基准评测

> 来源归档（repo）；核查日：2026-11-11；**部分开源**（相对于论文所述完整系统及训练方案）

- **仓库：** <https://github.com/AutoLab-SAI-SJTU/A2MEAS>
- **默认分支：** `evaluation`（核查时仓库包含一个初始发布提交）
- **论文：** <https://arxiv.org/abs/2611.09124v1>
- **项目页：** <https://a2-meas.github.io/>
- **模型：** <https://huggingface.co/shibohuang/policy-weights>；仓库 README 给出固定版本的模型权重链接
- **许可：** 核心项目代码、参考 episode 列表和 A²-WLA 推理权重采用 Apache-2.0，第三方依赖与仿真资产各有许可
- **核对范围：** README 的 Overview、Setup、Evaluation、Results、License；逐一核对 `evaluate.py`、`serve.py`、`benchmarks/behavior/run.py`、`benchmarks/behavior/evaluate.py`、`a2meas/behavior/server.py`、`a2meas/behavior/policy.py`、`a2meas/behavior/service.py`、`a2meas/behavior/correction/controller.py`、`benchmarks/robotwin/run.py`、`a2meas/robotwin/server.py` 的入口与关键调用（不是全仓逐文件审计）。
- **通读：** 已完成

## 开源范围与关键入口

| 路径 | 职责 |
|---|---|
| `evaluate.py` | 顶层 CLI：分发到 `benchmarks.behavior.evaluate.main` 或 `benchmarks.robotwin.run.main` |
| `serve.py` | 服务 CLI：分发到 BEHAVIOR / RoboTwin 对应服务入口 |
| `benchmarks/behavior/run.py` | 多任务多 GPU 编排，按 task/instance 分配评测，支持 `--adapter`，汇总结果 |
| `benchmarks/behavior/evaluate.py` | 单任务评测，启动官方 BEHAVIOR simulator GPU evaluator，连接服务 endpoint |
| `a2meas/behavior/server.py` | WebSocket/MessagePack 网关、`RouteManager` 路由与子进程 `route_worker` |
| `a2meas/behavior/policy.py` | `VLAPolicy.step`：累积观测、动作 chunk 队列、必要时生成新 chunk，执行单步输出 |
| `a2meas/behavior/service.py` | `ModelService` 装载模型和 checkpoint，调用 A²-WLA 推理 |
| `a2meas/behavior/correction/controller.py` | `CorrectionController.predict`：基础动作提案→诊断/grounding→编码纠错上下文→Adapter 注入→动作安全门 |
| `benchmarks/robotwin/run.py` | RoboTwin 评测调度与外部 vla-evaluation-harness 集成 |
| `a2meas/robotwin/server.py` | `RoboTwinModelServer` 接收多视角/语言/关节观测并返回策略动作 |

## README 已给出的复现入口

```bash
# 安装完成且 assets、权重已准备后；示例默认八卡
runtime/model/bin/python evaluate.py robotwin \
  --gpus 0 1 2 3 4 5 6 7 --workers-per-gpu 8 \
  --episodes 100 --output-dir results/robotwin-run

runtime/model/bin/python -m benchmarks.behavior.run \
  --gpus 0 1 2 3 4 5 6 7 --workers-per-gpu 4 \
  --num-inference-steps 2 --instance $(seq 0 9) --adapter \
  --output-dir results/behavior-run
```

README 要求 Linux x86-64、原生 BF16 NVIDIA GPU、CUDA 12.8 兼容驱动；开启 Adapter Judge 时默认要求 CUDA 13 兼容驱动及 96 GiB GPU 显存；BEHAVIOR 资源至少预留 250 GiB 及模拟器 assets。BEHAVIOR 的 `--adapter` 仅对 `configs/behavior/adapter.json` 指定的任务生效。真实运行受资产访问授权、模型权重、硬件和仿真版本约束，本次**只验证了入口及静态调用链，没有执行 GPU benchmark**。

## 源码证据与运行时流程

1. `benchmarks/behavior/run.py` 编排任务并启动评测/服务；`benchmarks/behavior/evaluate.py` 调用官方 GPU evaluator。
2. `a2meas/behavior/server.py` 接收 WebSocket 二进制 MessagePack 观测，经 `RouteManager.forward()` 投送路由 worker，要求动作输出为有限数值的 23D float32。
3. `route_worker` 调用 `VLAPolicy.step()`；没有待执行 chunk 时 `ModelService.predict()`，若配置中该任务启用 Adapter，则通过 `CorrectionController.predict()` 先产生基础动作、完成 need/progress 审查与空间 grounding，再尝试 residual hidden-state correction。可用性/安全门不通过时回退基础提案。
4. `VLAPolicy.step()` 从动作队列弹出一个 action，经网关发给模拟器；日志和汇总由评测编排与结果模块输出。

## 已发布 vs 未完整发布

- **已发布：** 两套 benchmark 的评测/推理源码、BEHAVIOR 任务路由、一个 Adapter、RoboTwin RSI 组件、权重/安装入口、测试样例。
- **未完整发布：** README 明说完整 Adapter 配置及选定任务的执行策略（含导航逻辑）被省略；原文承诺的完整训练脚本、训练数据处理及优化基础设施不属于本次公开复现路径，不能宣称训练端已全面开源。
- **论文完整系统报告：** BEHAVIOR Q **0.6411** / SR **45.40%**。
- **仓库最新复现表：** BEHAVIOR Q **0.5287** / SR **32.40%**；RoboTwin 完整系统平均 **95.05%**。论文结果与发布版复现表必须区分。

## 映射

- [原论文归档](../papers/a2-meas-agentic-embodied-intelligence_arxiv_2611_09124.md)
- [项目主页归档](../sites/a2-meas.md)
- [唯一主 Wiki 节点](../../wiki/entities/paper-a2-meas-agentic-embodied-intelligence.md)
