# PredActor: Predictive Action Diffusion for Steerable Onboard Humanoid Control（arXiv:2609.24840）

> 来源归档（ingest）

- **标题：** PredActor: Predictative Action Diffusion for Steerable Onboard Humanoid Control
- **类型：** paper / humanoid / diffusion-policy / onboard-control
- **arXiv abs：** <https://arxiv.org/abs/2609.24840>
- **PDF：** <https://arxiv.org/pdf/2609.24840>
- **项目页：** <https://masteryip.github.io/predactor.github.io/> — 归档见 [`sources/sites/predactor-masteryip-github-io.md`](../sites/predactor-masteryip-github-io.md)
- **代码：** **待发布** — 官方占位仓 <https://github.com/MasterYip/PredActor>（MIT；README 标 Code Coming Soon，尚无训练/部署脚本）；归档见 [`sources/repos/predactor.md`](../repos/predactor.md)
- **机构：** 哈尔滨工业大学（HIT）、上海创智学院（Shanghai Innovation Institute）、RoboParty Lab、清华大学（Tsinghua）、上海交通大学（SJTU）等
- **入库日期：** 2026-09-24
- **一句话说明：** 联合状态–动作扩散：仅用本体感知历史 + 可选任务上下文，内部预测未来状态供 CG/CFG 引导，直接输出可执行动作；G1 Jetson Orin NX 机载 50 Hz（中位 16.79 ms / p95 19.38 ms）。

## 核心摘录（面向 wiki 编译）

### 1) 范式定位

- **相对 generator–tracker：** 未来状态留在策略内部，不经独立 motion-reference tracker。
- **相对 action-only diffusion：** 显式 future-state 轨迹支持 test-time CG。
- **相对其他 joint diffusion（Diffuse-CLoC、SCDP、SCRIPT 等）：** 同时支持 **CFG + CG**，且 **仅 proprio** 输入、**G1 机载 50 Hz** 完整闭环。

### 2) headline 数字

| 指标 | PredActor | 对照（文内） |
|------|-----------|--------------|
| 15 目标点导航 | **15/15** | — |
| 文本检索分 | **0.580** | conditional action diffusion **0.373** |
| 推扰存活率 | **0.535** | **0.564**（相近） |
| Orin NX 完整 callback p50/p95 | **16.790 / 19.383 ms** | < 20 ms 控制周期 |
| 50 Hz 达标率 | **593/600** | — |

### 3) 训练管线要点

- 动作库 + 自动任务标签；异步扰动 teacher rollout；DAgger 式在 learner 访问状态聚合 teacher 标签。
- **Rolling denoising** + 计算保留优化 + 延迟补偿 → 机载实时。

### 4) 开源状态（项目页 + GitHub，2026-09-24）

| 组件 | 状态 |
|------|------|
| 项目页 / demo | 公开 |
| GitHub MasterYip/PredActor | **占位**（overview + demo 链；代码/checkpoint **Coming Soon**） |
| 权重 | **未发布** |

## 对 wiki 的映射

- 新建：[paper-predactor](../../wiki/entities/paper-predactor.md)
- 交叉：[diffusion-policy](../../wiki/methods/diffusion-policy.md)、[sonic-motion-tracking](../../wiki/methods/sonic-motion-tracking.md)、[whole-body-control](../../wiki/concepts/whole-body-control.md)、[locomotion](../../wiki/tasks/locomotion.md)
