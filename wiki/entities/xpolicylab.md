---
type: entity
tags: [vla, policy, evaluation, serving, open-source, infrastructure, robodojo, benchmark]
status: complete
updated: 2026-07-27
related:
  - ./robodojo.md
  - ../methods/vla.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ./vla-sota-leaderboard.md
  - ./xiaomi-robotics-1.md
  - ../methods/star-vla.md
sources:
  - ../../sources/repos/xpolicylab.md
  - ../../sources/sites/robodojo-benchmark.md
  - ../../sources/repos/robodojo.md
  - ../../sources/blogs/robodojo_open_longterm_eval_2026-07.md
summary: "XPolicyLab：机器人策略开发与部署的统一标准与基建——policy/<NAME> 自管依赖/权重/训练，框架统一 serving 与观测–动作契约；对接 RoboDojo/RoboTwin；约 40+ 模型适配；官方榜 verified 公布须经本仓开源训推与 checkpoint。"
---

# XPolicyLab（统一策略训推与评测适配层）

**XPolicyLab**（[GitHub](https://github.com/XPolicyLab/XPolicyLab)，Apache-2.0；文档站 [xpolicylab.github.io](https://xpolicylab.github.io)）是策略代码与评测环境之间的 **共享层**：每个模型把依赖、checkpoint、训练配方留在 `policy/<POLICY>/`，框架负责 **serving、观测/动作契约、与基准 eval 接线**。它是 [RoboDojo](./robodojo.md) 的官方策略集成口，也服务于 RoboTwin 等榜单；社区公告强调其已集成 **40+ 前沿模型复现**。

## 一句话定义

把异构 VLA / 扩散策略 / 基线接到 **统一 websocket policy server 与评测客户端契约**，使「适配一次 → 本地 debug / 仿真 / 真机云评测 / 开源上榜」走同一套目录与脚本约定。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 仓内大量适配对象（π、GR00T、InternVLA、Xiaomi 等） |
| WS | WebSocket | 默认 policy-server 协议（`protocol: ws`） |
| HDF5 | Hierarchical Data Format 5 | RoboDojo 等导出数据的一种格式 |
| SFT | Supervised Fine-Tuning | 适配 README 常见训练入口语义 |
| PR | Pull Request | 社区模型接入与官方上榜的提交形态 |
| CKPT | Checkpoint | 评测与 verified 公布必须对齐的权重快照 |

## 为什么重要

- **评测可复现的瓶颈常在「模型侧」：** 基准开源后，若每个实验室私有 serving/动作键不一致，分数不可比。XPolicyLab 把契约标准化。
- **对接 RoboDojo 公正性门槛：** verified 上榜要求在分数公开前经本仓释放 **训推代码 + evaluated checkpoint + 配置与推理说明**——本仓既是工程基建，也是 **社区监督入口**。
- **降低「集成税」：** `scripts/create_policy.sh`、`demo_policy` 参考实现，以及 Cursor skills（`xpolicylab-model-integration` 等）缩短新模型接入路径。

## 核心原理

### 边界划分

```text
Policy environment                         Evaluation / benchmark environment
------------------                         ----------------------------------
policy/<POLICY>/model.py     <---ws--->    env client / simulator / robot
policy server                              environment client
deploy.yml runtime config                  benchmark task and observation API
```

### 适配目录骨架

```text
policy/<POLICY>/
├── README.md
├── install.sh
├── process_data.sh      # optional
├── train.sh             # optional；暂不可开源训练时可先 eval-only 并告知维护者时间线
├── eval.sh
├── setup_eval_policy_server.sh
├── setup_eval_env_client.sh
├── deploy.yml
├── deploy.py
└── model.py             # Model: __init__ / update_obs / get_action / reset (+ batch)
```

### 已集成策略（核查日口径）

截至 **2026-07-27**，上游 `policy/` 约 **41** 个目录（含 `demo_policy`、ACT、DP 等基线），覆盖例如：π₀ / π₀.₅ / π₀-FAST、GR00T_N17、InternVLA_A1(_5)、starVLA、Xiaomi_Robotics_0/1、GO1、OpenVLA_OFT、RDT_1B、SmolVLA、MolmoACT2、A1、GalaxeaVLA、GigaWorldPolicy 等。论文 RoboDojo 摘要写集成 **30** 策略——**引用「多少模型」时以仓内当日目录为准**。

## 工程实践

| 步骤 | 做法 |
|------|------|
| 1. 学参考实现 | 读 `policy/demo_policy` 的 `model.py` / `deploy.py` / `deploy.yml` / `eval.sh` |
| 2. 建骨架 | `bash scripts/create_policy.sh <POLICY_NAME>` |
| 3. 无仿真验契约 | `EVAL_ENV_TYPE=debug` 跑 `eval.sh` |
| 4. 仿真 / 远程 | `EVAL_ENV_TYPE=sim` 或拆分 server 与 env client |
| 5. 数据 | `scripts/RoboDojo/download_robodojo_data.sh`（demo / hdf5 / lerobot / real） |
| 6. 上榜 | 按 CONTRIBUTING 开 PR；描述中附 HF/ModelScope checkpoint 下载脚本 |

与 [RoboDojo](./robodojo.md) 联调时：`eval.sh` 拉起 server 后回调 RoboDojo `scripts/eval_policy.sh`。

## 局限与风险

- **适配质量不齐：** 目录存在 ≠ 该模型在 RoboDojo 全任务可跑通；以各 `policy/*/README.md` 与官方榜为准。
- **训练开源可延期、评测不可糊弄：** 规则允许先 eval-only 接入，但 **verified 公布**仍要可复现的 evaluated artifact。
- **依赖地狱：** 各 policy 自带环境；统一的是契约而非单一 conda。
- **勿与摘录榜混淆：** 本仓服务 **重跑/官方评测接线**；[VLA SOTA Leaderboard](./vla-sota-leaderboard.md) 是论文分数导航。

## 关联页面

- [RoboDojo](./robodojo.md) — 统一 sim-and-real 基准与公益上榜规则
- [VLA](../methods/vla.md) — 方法总览与仓内大量适配对象
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md) — 闭环评测方法论
- [具身评测选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 策略成功率层工程入口
- [Xiaomi-Robotics-1](./xiaomi-robotics-1.md) — 已出现在适配目录与 RoboDojo 分数叙事中的案例
- [starVLA](../methods/star-vla.md) — 仓内适配之一

## 参考来源

- [仓库归档 XPolicyLab](../../sources/repos/xpolicylab.md)
- [RoboDojo 官网归档](../../sources/sites/robodojo-benchmark.md)
- [RoboDojo 仓库归档](../../sources/repos/robodojo.md)
- [开放长期公益评测公告](../../sources/blogs/robodojo_open_longterm_eval_2026-07.md)

## 推荐继续阅读

- [XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab) — 框架总览与 Common Workflow
- [RoboDojo 文档 · XPolicyLab](https://robodojo-benchmark.com/doc/usage/xpolicylab/) — 与基准联调说明
- [RoboDojo Protocol](https://robodojo-benchmark.com/leaderboard/protocol) — verified 开源产物要求全文
