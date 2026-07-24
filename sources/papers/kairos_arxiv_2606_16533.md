# Kairos: A Regret-Aware Native World-Action Model Stack for Physical AI（arXiv:2606.16533）

> 来源归档（ingest）；**v3（2026-07-03）** 将标题与叙事升格为 **regret-aware / control-sufficient**；代码仓更名为 `kairos-agi/kairos`。

- **标题：** Kairos: A Regret-Aware Native World-Action Model Stack for Physical AI — Learning, Maintaining, and Deploying Control-Sufficient World States
- **历史标题（v1）：** Kairos: A Native World Model Stack for Physical AI
- **品牌名：** **Kairos** / **Kairos 3.0 / 3.1**（kairos-agi 世界–动作模型栈；与 **Kairos · HomeWorld**（arXiv:2606.06390 全屋场景生成）为**不同项目**）
- **类型：** paper / technical report
- **arXiv：** <https://arxiv.org/abs/2606.16533>（PDF：<https://arxiv.org/pdf/2606.16533.pdf>；HTML v3：<https://arxiv.org/html/2606.16533v3>；当前版本 **v3**，2026-07-03）
- **Hugging Face Papers：** <https://huggingface.co/papers/2606.16533>
- **平台页：** <https://kairos.acerobotics.com>
- **代码：** <https://github.com/kairos-agi/kairos>（旧名 `kairos-agi/kairos-sensenova` **301 重定向**至此仓；Apache-2.0）
- **权重：** 组织入口 <https://huggingface.co/kairos-agi>；集合 [Kairos3.0](https://huggingface.co/collections/kairos-agi/kairos30)；ModelScope：<https://modelscope.cn/collections/kairos-team/kairos30>。README 模型卡常写作 `kairos-agi/...`，实际解析可落到 **`ACERobotics/...`** 镜像（如 `Kairos3.1-4B-robot-480P`、`kairos-4B-robot-RoboTwin2.0`、`kairos-4B-robot-LIBERO-plus`、`kairos-sensenova-4B-720P`）。
- **团队：** Kairos Team（Project Lead：Fei Wang, Shan You, Qiming Zhang；Core：Tao Huang, Zuoyi Fu；平台归属 **Ace Robotics / ACERobotics**）
- **入库日期：** 2026-06-18；**修订：** 2026-07-24（v3 regret-aware 叙事 + 官方仓更名 + Kairos3.1 权重）
- **一句话说明：** 面向 **Physical AI** 的 **regret-aware 原生世界–动作模型栈**：不以「复现全部未来像素」为目标，而以 **control-sufficient state** \(Z_t\) 压缩控制相关信息（物体状态、接触、任务进度、动作后果、失败边界与部署不确定性）；以 **CEDC** 按干预强度组织开放视频→人类行为→机器人数据，以 **理解/生成/预测统一 MoT + SWA/DSWA/GLA** 维持多时间尺度状态，并以 **部署协同设计** 支撑低延迟闭环；**Kairos-4B / 3.1** 在 WM 与 WAM 基准报告强竞争力结果，并开源推理与 LIBERO-Plus / RoboTwin 评测入口。

## 摘要级要点（v3）

- **第一性原理：** 具身世界模型应是 **控制充分状态** 的学习与部署系统，而非全像素仿真器；机器人付出的代价是碰撞、任务失败、干预与安全越界——报告用 **horizon-level representation-induced regret** \(\operatorname{Reg}_H(f;g)\) 度量「从 \(Z_t\) 规划」相对「从全历史 \(H_t\) 规划」的超额物理代价。
- **三大支柱（不变骨架，叙事重写）：**
  1. **Learn — CEDC：** 按 **干预强度**（被动物理观察 → 有意图行为 → 具身动作接地）组织异构数据，拒绝「先训通用 T2V 再后训策略」割裂范式。
  2. **Maintain — 统一 Understanding / Generation / Prediction + Hybrid Linear Temporal Attention：** 共享世界–动作状态 \(Z_t\)；**SWA / DSWA / GLA** 分别覆盖局部、中程与全局因果记忆（含误差界理论）。
  3. **Run — Deployment-Aware Co-Design：** 延迟、显存、硬件兼容性为一等约束；**DMD + Consistency Distillation** 等蒸馏与 kernel/量化/streaming。
- **评测定位：** 现有 WM/WAM/效率数字被视为 **regret 相关能力的 proxy evidence**；摘要明确写出：**真机闭环 regret 降低**（rollout 相关、失败预测、安全过滤、恢复学习、想象经验改进策略）仍是未来方向。
- **WAM：** Video DiT + Action DiT（MoT）联合 flow matching；**action-only** 与 **Kairos-joint**；LIBERO-Plus **89.0 / joint 90.8**；RoboTwin 2.0 Average **96.1%**（Clean **96.9** / Randomized **95.2**）。
- **开源（截至 2026-07-02 README）：** 官方仓含 `examples/inference.sh`、`kairos/pipelines/*`、`benchmarks/{libero_plus,robotwin}`；**Kairos3.1** 发布具身生成权重与可执行动作预测权重（RoboTwin / LIBERO-Plus）。

## 核心论文摘录（MVP）

### 1) Regret-aware 目标与 control-sufficient state

- **链接：** Abstract；§1 Introduction；Conclusion（v3）
- **摘录要点：** 世界模型不应追求复现桌面纹理/窗外云等无关像素；应保留杯子位姿、接触、抓取 affordance、失败风险与动作后果。给定历史 \(H_t\)、目标 \(g\) 与候选动作序列，模型维护 \(Z_t=f(H_t)\)，使基于 \(Z_t\) 的规划期望物理代价逼近基于全历史的代价——\(\operatorname{Reg}_H(f;g)\) 非负，度量压缩诱导的超额成本。视觉真实感仍有用，但对 Physical AI **不充分**。
- **对 wiki 的映射：**
  - [Kairos（原生世界–动作模型栈）](../../wiki/entities/paper-kairos-native-world-model-stack.md) — 总判与 regret 叙事。
  - [World Action Models](../../wiki/concepts/world-action-models.md) — Joint WAM 与「控制相关监督」坐标。

### 2) 三支柱：CEDC + 统一架构 + 部署协同

- **链接：** Abstract；Fig. 1–2；§2–§5
- **摘录要点：** **CEDC** 以干预强度递进；**Understanding** 用 Qwen 系 VLM 构造 \(Z_t\)；**Generation** 用未来想象正则化物理一致性；**Prediction** 用 Video/Action DiT 提供世界–动作接口；**Hybrid Linear Temporal Attention** 维持多尺度状态；部署侧把效率写进建模目标，以支撑未来 observation–action–feedback 闭环与 proxy rollout–evaluation–refinement。
- **对 wiki 的映射：**
  - [Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md) — 流程总览 Mermaid。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 「基础设施级 / regret-aware」叙事更新。

### 3) Hybrid Linear Temporal Attention（SWA / DSWA / GLA）

- **链接：** §2.2–2.3；Theorem 1–2
- **摘录要点：** 替代全序列 Softmax 二次复杂度；**GLA（Gated DeltaNet）** 为全局路径；**SWA** 局部运动/接触，**DSWA（dilation 6/12）** 中程依赖；纯局部窗口对超窗依赖任务有不可消除 excess risk；混合分解在收缩全局记忆下有界累积误差。
- **对 wiki 的映射：**
  - [Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md) — 时序因子化。

### 4) CEDC 与三阶段原生预训练

- **链接：** §3；Fig. 7–8
- **摘录要点：** Phase I 开放视频物理；Phase II 人类中心任务组织；Phase III 机器人轨迹（AgiBotWorld-Beta、DROID 等）；**Stage I–II 仅 VideoDiT → Stage III 联合 ActionDiT**；后接域 SFT、merging、Video DPO。数据价值应按 **失败/接触/恢复/边界案例** 的控制信息密度衡量，而非仅看视觉干净度（Limitations）。
- **对 wiki 的映射：**
  - [Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md) — CEDC 表。
  - [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) — 异构数据混合坐标。

### 5) 评测、效率与开源复现入口

- **链接：** §6；Table 12–15；仓库 README（2026-07-02）
- **摘录要点：** WorldModelBench-robot **9.30**；DreamGen AVG **0.618**；PAI-Bench TI2V Overall **82.57**；WAM 微调后 LIBERO-Plus **89.0**（joint **90.8**）、RoboTwin 2.0 **96.1%**；人类中心预训练 **+6.0**、联合生成+预测相对仅 ActionDiT **+23.2**；蒸馏模型 A800 480P 实时推理表（1×GPU **11.7 s** / 4× **3.0 s**，README 5.2.1）。官方仓提供 `examples/inference.sh` 与 `benchmarks/libero_plus`、`benchmarks/robotwin`。
- **对 wiki 的映射：**
  - [Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md) — 评测表、源码时序图、工程实践。
  - [τ₀-World Model](../../wiki/entities/tau0-world-model.md)、[Cosmos 3](../../wiki/entities/cosmos-3.md) — 同生态对照。

## BibTeX（官方 README / arXiv v3）

```bibtex
@misc{kairosteam2026kairosregretawarenativeworldaction,
  title         = {Kairos: A Regret-Aware Native World-Action Model Stack for Physical AI},
  author        = {Kairos Team and Fei Wang and Shan You and Qiming Zhang and Tao Huang and Zuoyi Fu and Zhisheng Zheng and Yunlong Xi and Feng Lv and Xiaoming Wu and Zeyu Liu and Cong Wan and Pu Li and Ruiqing Yang and Xiaoou Li and Wei Wang and Kangkang Zhu and Yuwei Zhang and Shi Fu and Zheng Zhang and Xiaoning Wu and Xuzeng Fan and Dacheng Tao and Xiaogang Wang},
  year          = {2026},
  eprint        = {2606.16533},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2606.16533}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-kairos-native-world-model-stack.md`](../../wiki/entities/paper-kairos-native-world-model-stack.md)
- 代码归档：[`sources/repos/kairos.md`](../repos/kairos.md)（旧索引 [`kairos_sensenova.md`](../repos/kairos_sensenova.md) 保留重定向说明）
- 平台页：[`sources/sites/kairos-acerobotics.md`](../sites/kairos-acerobotics.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[HomeWorld（品牌区分）](../../wiki/entities/paper-homeworld-whole-home-scene-generation.md)
