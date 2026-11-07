# PredActor: Predictive Action Diffusion for Steerable Onboard Humanoid Control（arXiv:2609.24840）

> 来源归档（ingest）

- **标题：** PredActor: Predictive Action Diffusion for Steerable Onboard Humanoid Control
- **类型：** paper / humanoid / diffusion-policy / onboard-control
- **arXiv abs：** <https://arxiv.org/abs/2609.24840>
- **PDF：** <https://arxiv.org/pdf/2609.24840>
- **项目页：** <https://masteryip.github.io/predactor.github.io/> — 归档见 [`sources/sites/predactor-masteryip-github-io.md`](../sites/predactor-masteryip-github-io.md)
- **代码：** **部分开放** — <https://github.com/MasterYip/PredActor> 提供带 PDP051 checkpoint 的 MuJoCo evaluator；训练/数据采集/DAgger/真机部署仍未发布；评测权重：[PredActor_Artifacts](https://huggingface.co/MasterYip/PredActor_Artifacts)，见 [`sources/repos/predactor.md`](../repos/predactor.md) 与 [`sources/repos/predactor-artifacts.md`](../repos/predactor-artifacts.md)
- **机构：** 哈尔滨工业大学（Harbin Institute of Technology）、上海创新研究院（Shanghai Innovation Institute）、RoboParty Lab、清华大学（Tsinghua）、上海交通大学（SJTU）等
- **入库日期：** 2026-09-24
- **复核日期：** 2026-11-07（重新取得 arXiv 当前版 PDF，31 页）
- **核对范围：** 摘要、引言要点、Table 1 的部署栏、第 4 章实验设置、4.2 推理加速与机载评测（Table 4）、附录 B.3 任务与部署接口、B.4 评测部署配置、附录 E 中与 TensorRT 基准有关的一处引用；其余方法推导、仿真对比表和附录**未读**。注意当前版摘要的仿真数字（44/45 目标、文本检索分 0.539）与本归档早先摘录（15/15、0.580）不同，本归档的仿真数字未更新。
- **通读：** 未完成
- **最近复核：** 2026-10-08
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

### 4) 首次入库时开源状态（项目页 + GitHub，2026-10-03）

| 组件 | 状态 |
|------|------|
| 项目页 / demo | 公开 |
| GitHub MasterYip/PredActor | **占位**（overview + demo 链；代码/checkpoint **Coming Soon**） |
| 权重 | **未发布** |

## 最新开源状态（2026-10-08）

- 官方 [MuJoCo evaluator](https://github.com/MasterYip/PredActor) 和 [PDP051 / MotionCLIP checkpoint](https://huggingface.co/MasterYip/PredActor_Artifacts) 已公开；quick evaluation 使用锁定的 Python 3.10 环境。
- 训练、数据采集/标注、DAgger 策略精炼和硬件部署仍未发布；不能据评测包声称完整训练/真机复现已开放。
- 项目页团队入口：[masteryip.github.io/predactor.github.io/#people](https://masteryip.github.io/predactor.github.io/#people)。

## 对 wiki 的映射

- 复用并更新：[paper-predactor](../../wiki/entities/paper-predactor.md)
- 交叉：[diffusion-policy](../../wiki/methods/diffusion-policy.md)、[sonic-motion-tracking](../../wiki/methods/sonic-motion-tracking.md)、[whole-body-control](../../wiki/concepts/whole-body-control.md)、[locomotion](../../wiki/tasks/locomotion.md)

## 2026-11-07 部署相关核对（当前版 PDF）

- 评测平台是 Unitree G1 板载 Jetson Orin NX；**策略导出为 TorchScript 主干 + C++ 运行时（LibTorch inference mode），batch-one FP32，两步确定性 DDIM**，并非 ONNX 或 TensorRT。完整控制回调（含观测处理、网络前向、下发）**中位 16.790 ms / p95 19.383 ms，593/600 次在 20 ms 预算内**；仅 actor 前向 p50 13.175 ms。时延逐项优化（Table 4）：结构基线 21.461 ms → 堆叠 WBG 18.315 → DDIM 预计算 17.442 → QKV 打包 16.331 → 条件缓存 15.930 → inference mode 13.175（actor p50）。
- 控制为 29 个关节、50 Hz、预测 20 步状态-动作视野；物理 G1 试验每种设置 5 次（Table 3），如 Stand→Squat 5/5、Squat→Stand 4/5、站立被推 4/5。
- 文中 Table 1 同时列出了其他系统的部署栏（如 RTX 4090 离板、Orin 机载等）；附录 B 还提到 DiffuseLoco 的 TensorRT 基准 116.5 Hz（RTX 4060 mini-PC）。
