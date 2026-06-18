# Kairos: A Native World Model Stack for Physical AI（arXiv:2606.16533）

> 来源归档（ingest）

- **标题：** Kairos: A Native World Model Stack for Physical AI — Learning, Maintaining, and Running Worlds for Future Self-Evolving Agents
- **品牌名：** **Kairos** / **Kairos 3.0**（kairos-agi 世界模型栈；与 **Kairos · HomeWorld**（arXiv:2606.06390 全屋场景生成）为**不同项目**）
- **类型：** paper / technical report
- **arXiv：** <https://arxiv.org/abs/2606.16533>（PDF：<https://arxiv.org/pdf/2606.16533.pdf>；HTML：<https://arxiv.org/html/2606.16533>）
- **Hugging Face Papers：** <https://huggingface.co/papers/2606.16533>
- **代码：** <https://github.com/kairos-agi/kairos-sensenova>
- **权重：** <https://huggingface.co/kairos-agi>；ModelScope：<https://modelscope.cn/collections/kairos-team/kairos30>
- **团队：** Kairos Team（Project Lead：Fei Wang, Shan You, Qiming Zhang；Core：Tao Huang, Zuoyi Fu）
- **入库日期：** 2026-06-18
- **一句话说明：** 面向 **Physical AI** 的 **原生世界模型栈**：以 **跨具身数据课程（CEDC）** 从开放视频 → 人类行为 → 机器人交互渐进注入物理知识，以 **理解–生成–预测统一 MoT + 混合线性时序注意力（SWA/DSWA/GLA）** 维持长程世界状态，并以 **部署感知系统协同设计** 在服务器与消费级硬件上支撑低延迟闭环；**Kairos-4B** 在 WorldModelBench / DreamGen / PAI-Bench 与 **LIBERO-Plus / RoboTwin 2.0** 等 WAM 基准上报告 SOTA 级表现与线性可扩展推理效率。

## 摘要级要点

- **定位转变：** 世界模型从「被动视觉生成器」转向 **Physical AI 可操作基础设施**——须 **原生习得** 异构经验、**持久维护** 长程状态、并在 **真实部署约束** 下 **高效运行**。
- **三大支柱：**
  1. **Learn — Native Pre-training + CEDC：** 拒绝「先训通用视频生成器再后训策略」的割裂范式；按 **物理常识 → 人类任务组织 → 机器人具身** 金字塔组织开放视频、人类行为与机器人数据。
  2. **Maintain — 统一理解/生成/预测 + Hybrid Linear Temporal Memory：** **Qwen 系 VLM** 做 World Understanding；**Video DiT**（flow matching）做 World Generation；**Video DiT + Action DiT（MoT）** 做 World Prediction（WAM）；时序骨干 **SWA（局部）+ DSWA（中程）+ GLA（全局因果记忆）**，并给出 **误差累积上界** 理论。
  3. **Run — Deployment-Aware Co-Design：** 硬件感知 kernel、量化、token streaming；**DMD + Consistency Distillation** 将 480P 具身 WM 蒸馏为 **4 步** 生成器；**Kairos-4B** 在 A800 上报告 **23.5 GB** 显存、**2.3 PFlops**、单卡 **43 s / 四卡 9 s**（480P 5s 蒸馏模型），DiT 单步延迟随分辨率/时长 **近线性** 扩展。
- **WAM 机制：** 历史/未来视频 token 与动作 token 三组序列 + **混合注意力掩码**；动作分支可 **不生成未来视频** 单独推理（action-only mode）；**Kairos-joint** 在推理时联合去噪未来视频与动作，LIBERO-Plus 平均 **89.0 → 90.8**。
- **主要数字（论文报告）：** WorldModelBench-robot **9.30**；DreamGen AVG_Score **0.618**；PAI-Bench TI2V Overall **82.57**；LIBERO-Plus **89.0**（joint **90.8**）；RoboTwin 2.0 Average **96.1%**。

## 核心论文摘录（MVP）

### 1) 三支柱总览与原生预训练主张

- **链接：** <https://arxiv.org/abs/2606.16533> Abstract；§1 Introduction
- **摘录要点：** 四大瓶颈——异构经验碎片化、长程状态维护、理解–控制鸿沟、部署/闭环延迟——被 **联合** 而非分项工程化处理；强调 **物理规律、行为语义与具身接地必须在 scaling 起点原生合成**，而非对开放域视频生成器做后训微调。
- **对 wiki 的映射：**
  - [Kairos（原生世界模型栈）](../../wiki/entities/paper-kairos-native-world-model-stack.md) — 总览与 Mermaid。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 从「演示级生成」到「基础设施级 WM」叙事。

### 2) 统一架构：Understanding / Generation / Prediction

- **链接：** §2.1；Fig. 2–4
- **摘录要点：** **World Understanding** 用 **Qwen2.5-VL / Qwen3.5** 等 VLM 编码多模态条件；**World Generation** 为条件扩散 **LinearDiT**（高压缩 VAE + 跨注意力条件）；**World Prediction** 为 **WAM**：**Video DiT**（自预训练生成模型初始化）+ **Action DiT**（约 1/5 规模）联合 flow matching；历史视频 token 仅看历史，未来视频/动作可看全历史，动作分支 **full attention**、未来视频 **sparse spatiotemporal attention**。
- **对 wiki 的映射：**
  - [Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md) — MoT 与 WAM 掩码表。
  - [World Action Models](../../wiki/concepts/world-action-models.md) — Joint WAM 族对照。

### 3) Hybrid Linear Temporal Attention（SWA / DSWA / GLA）

- **链接：** §2.2；§2.3 理论；Fig. 5–6
- **摘录要点：** 替代全序列 Softmax 二次复杂度；**GLA（Gated DeltaNet）** 为唯一全局路径，delta 更新 + 遗忘门维持 **物体恒常性** 与延迟效应；**SWA** 捕获帧内/短程运动，**DSWA（dilation 6/12）** 捕获约秒级中程依赖；**Theorem 1** 证明纯局部窗口对超窗依赖任务有 **不可消除 excess risk**；**Theorem 2** 证明混合分解在收缩全局记忆下 **有界累积误差**。
- **对 wiki 的映射：**
  - [Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md) — 时序因子化与理论动机节。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 长程视频 WM 效率–一致性折中。

### 4) CEDC 与三阶段原生预训练

- **链接：** §3；Fig. 7–8；Tab. 1
- **摘录要点：** **Phase I 物理知识**（百万小时级开放视频 + 物理 CoT）；**Phase II 人类中心行为**（>10 万小时，任务组织与干预因果）；**Phase III 机器人动作**（AgiBotWorld-Beta、DROID 等接地）；训练流水线：**Stage I–II 仅优化 VideoDiT** → **Stage III 联合 ActionDiT**；分辨率 **256P→720P**、时长至 **241 帧（~15 s）**；后接域 SFT + model merging + **Video DPO**。
- **对 wiki 的映射：**
  - [Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md) — CEDC 金字塔表。
  - [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) — 异构数据混合与训练闭环坐标。

### 5) 部署蒸馏与 WAM 评测

- **链接：** §5 Inference；§6 Evaluation；Tab. 4–5、11–12、13–14
- **摘录要点：** **DMD + CM** 混合蒸馏至 **4 步**；**Kairos-4B** 相对 Cosmos-Predict2.5-14B **28×–85×** 延迟优势；WAM 微调后 **LIBERO-Plus 89.0**、**RoboTwin 2.0 96.1%**；消融：人类中心预训练 **+6.0** LIBERO-Plus；联合生成+预测训练相对仅 ActionDiT **+23.2**；joint 去噪 **+1.8**。
- **对 wiki 的映射：**
  - [Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md) — 效率与 benchmark 表。
  - [τ₀-World Model](../../wiki/entities/tau0-world-model.md)、[Cosmos 3](../../wiki/entities/cosmos-3.md) — 同生态 WAM / 全模态平台对照。

## BibTeX（arXiv）

```bibtex
@misc{kairos2026native,
  title         = {Kairos: A Native World Model Stack for Physical AI},
  author        = {{Kairos Team}},
  year          = {2026},
  eprint        = {2606.16533},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2606.16533}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-kairos-native-world-model-stack.md`](../../wiki/entities/paper-kairos-native-world-model-stack.md)
- 代码归档：[`sources/repos/kairos_sensenova.md`](../repos/kairos_sensenova.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[HomeWorld（品牌区分）](../../wiki/entities/paper-homeworld-whole-home-scene-generation.md)
