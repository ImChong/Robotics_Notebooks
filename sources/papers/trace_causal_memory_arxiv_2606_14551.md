# TRACE（arXiv:2606.14551）

> 来源归档（ingest）

- **标题：** TRACE：轨迹路由因果记忆用于延迟证据视觉运动模仿
- **英文标题：** TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation
- **类型：** paper / imitation-learning / memory / visuomotor / act / diffusion-policy
- **arXiv：** <https://arxiv.org/abs/2606.14551>（v3 2026-08-31；PDF：<https://arxiv.org/pdf/2606.14551>）
- **项目页：** <https://jeong-zju.github.io/trace/>
- **代码：** <https://github.com/Jeong-zju/corl-trace>（步骤 2.5 已开源，见 [`sources/repos/corl-trace.md`](../repos/corl-trace.md)）
- **机构：** 芝诺机器人（Zeno AI）；浙江大学（ZJU）；浙江工业大学（ZJUT）；悉尼大学（USYD）
- **作者：** Zihao Li, Ranpeng Qiu, Yincong Chen, Guoqiang Ren, Weiming Zhi
- **平台：** 真机 5 项延迟证据操作任务；每项 25 rollouts
- **开源：** **已开源**（训练 / 部署 / `streaming_act` 策略扩展）
- **入库日期：** 2026-09-15

## 核心论文摘录

### 1) 延迟证据问题

- 早期线索（物体来源、衣物侧面、托盘位置等）在**分支决策点**前离开视野；当前帧观测不足以区分应执行的动作。
- **对 wiki 的映射：** [paper-trace-causal-memory](../../wiki/entities/paper-trace-causal-memory.md)

### 2) 固定槽因果记忆 + 路径签名路由

- **Write：** 线索可见时将视觉-本体证据写入固定大小 latent 槽。
- **Route：** 用深度-3 **path signature**（机器人状态轨迹的序敏感紧凑特征）作地址，而非原始时间或人工任务标签。
- **Read / Adapt：** 轻量 adapter 将读出的记忆条件化到 ACT / Diffusion 等骨干，**不改**动作头与 IL 目标。
- **对 wiki 的映射：** 同上

### 3) 真机评测

- **平均阶段进度：** TRACE Regression **69.23%** vs ACT **25.50%**；TRACE Diffusion **59.53%** vs Diffusion Policy **25.00%**。
- 任务例：Book 双路线各 **83%**；Laundry **81%**；Cable **51%**；Medicine 左右托盘各 **76.67%**。
- **对 wiki 的映射：** 同上；与 [paper-zeno-1-collaborative-intelligence](../../wiki/entities/paper-zeno-1-collaborative-intelligence.md) 的持久交互记忆叙事可对照

## 步骤 2.5 开源核查（2026-09-15）

- 项目页列 **Code** → `Jeong-zju/corl-trace`。
- 仓库含 `scripts/`、`policy/`（含 `streaming_act`）、`deploy/`（ROS1 真机适配）、`environment.yml`、`bash/train_policy.sh`。
- **结论：** **已开源**；真机部署见 `deploy/README.md` 与 `deploy/configs/deploy_zeno_*.yaml`。

## 当前提炼状态

- [x] 项目页 + 仓库步骤 2.5 核查
- [x] wiki 实体页 + 源码运行时序图
- [x] `sources/repos/corl-trace.md`
