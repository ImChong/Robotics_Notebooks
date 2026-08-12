---
type: entity
tags: [vla, policy, evaluation, serving, open-source, infrastructure, robodojo, benchmark, hku, tsinghua]
status: complete
updated: 2026-08-12
related:
  - ./paper-xpolicylab.md
  - ./robodojo.md
  - ../methods/vla.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ./vla-sota-leaderboard.md
  - ./xiaomi-robotics-1.md
  - ../methods/star-vla.md
sources:
  - ../../sources/repos/xpolicylab.md
  - ../../sources/sites/xpolicylab-github-io.md
  - ../../sources/papers/xpolicylab_arxiv_2608_09892.md
  - ../../sources/sites/robodojo-benchmark.md
  - ../../sources/repos/robodojo.md
  - ../../sources/blogs/robodojo_open_longterm_eval_2026-07.md
summary: "XPolicyLab：机器人策略评测与部署的统一标准与基建（arXiv:2608.09892）——policy/<NAME> 自管依赖/权重/训练，框架统一 serving 与观测–动作契约；对接 RoboTwin/RoboDojo；论文日 42 策略；官方榜 verified 须经本仓开源训推与 checkpoint。"
---

# XPolicyLab（统一策略训推与评测适配层）

**XPolicyLab**（[GitHub](https://github.com/XPolicyLab/XPolicyLab)，Apache-2.0；文档站 [xpolicylab.github.io](https://xpolicylab.github.io)；技术报告 [arXiv:2608.09892](https://arxiv.org/abs/2608.09892)）是策略代码与评测环境之间的 **共享层**：每个模型把依赖、checkpoint、训练配方留在 `policy/<POLICY>/`，框架负责 **serving、观测/动作契约、与基准 eval 接线**。它是 [RoboDojo](./robodojo.md) 的官方策略集成口，也服务于 RoboTwin 等榜单。论文级机制与结论见 [XPolicyLab 论文实体](./paper-xpolicylab.md)。

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
- **降低「集成税」：** `scripts/create_policy.sh`、`demo_policy` 参考实现，以及 Cursor skills（`xpolicylab-model-integration` 等）缩短新模型接入路径；论文受控研究称代表策略 **>5 h → ~2 h**，agent skills 再降至约 **30 min**。

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

截至论文/项目页 **2026-08-08**，官方口径为 **42** 个策略适配（Table I / 项目站列表），覆盖 π₀ / π₀.₅、GR00T-N1.7、InternVLA-A1(_5)、StarVLA、Xiaomi-Robotics-0/1、GO-1、OpenVLA-OFT、RDT-1B、SmolVLA、MolmoAct2、GigaWorld-Policy、DreamZero、Mem-0 等。更早 RoboDojo 摘要写 **30**、社区公告写 **40+**——**引用「多少模型」时以仓内当日 `policy/` 目录为准并注明核查日**。

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

- [XPolicyLab 论文实体](./paper-xpolicylab.md) — arXiv:2608.09892 机制、时序图与结论
- [RoboDojo](./robodojo.md) — 统一 sim-and-real 基准与公益上榜规则
- [VLA](../methods/vla.md) — 方法总览与仓内大量适配对象
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md) — 闭环评测方法论
- [具身评测选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 策略成功率层工程入口
- [Xiaomi-Robotics-1](./xiaomi-robotics-1.md) — 已出现在适配目录与 RoboDojo 分数叙事中的案例
- [starVLA](../methods/star-vla.md) — 仓内适配之一

## 参考来源

- [仓库归档 XPolicyLab](../../sources/repos/xpolicylab.md)
- [项目页归档](../../sources/sites/xpolicylab-github-io.md)
- [论文归档（arXiv:2608.09892）](../../sources/papers/xpolicylab_arxiv_2608_09892.md)
- [RoboDojo 官网归档](../../sources/sites/robodojo-benchmark.md)
- [RoboDojo 仓库归档](../../sources/repos/robodojo.md)
- [开放长期公益评测公告](../../sources/blogs/robodojo_open_longterm_eval_2026-07.md)

## 推荐继续阅读

- [XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab) — 框架总览与 Common Workflow
- [项目页](https://xpolicylab.github.io/) — 42 策略与贡献指南
- [RoboDojo 文档 · XPolicyLab](https://robodojo-benchmark.com/doc/usage/xpolicylab/) — 与基准联调说明
- [RoboDojo Protocol](https://robodojo-benchmark.com/leaderboard/protocol) — verified 开源产物要求全文
