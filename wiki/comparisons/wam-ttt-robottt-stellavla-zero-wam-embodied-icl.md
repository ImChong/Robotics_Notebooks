---
type: comparison
tags: [in-context-learning, icl, vla, wam, test-time-training, manipulation, taxonomy]
status: complete
updated: 2026-11-19
related:
  - ../../roadmap/depth-icl.md
  - ../queries/embodied-fm-taxonomy-loop.md
  - ../concepts/robot-in-context-learning.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../entities/paper-wam-ttt-human-video-test-time-steering.md
  - ../entities/paper-robottt-test-time-training-vla-context.md
  - ../entities/paper-stellavla-structured-icl-vla.md
  - ../entities/paper-ici-vla-spatiotemporal-icl.md
  - ../entities/paper-zero-wam.md
  - ../entities/generalist-gen15-one-shot.md
  - ../entities/skild-s1.md
  - ../entities/paper-host-one-shot-human-video.md
  - ../entities/paper-imitator-game.md
sources:
  - ../../sources/papers/wam_ttt_arxiv_2607_06988.md
  - ../../sources/papers/robottt_arxiv_2607_15275.md
  - ../../sources/papers/stellavla_arxiv_2608_11671.md
  - ../../sources/papers/zero_wam_arxiv_2608_26103.md
  - ../../sources/repos/zero-wam.md
  - ../../sources/papers/in_context_learning_robots_arxiv_2609_36012.md
  - ../../sources/blogs/wechat_meiri_zhineng_embodied_icl_four_papers_2026-08-31.md
  - ../../sources/blogs/wechat_embodied_heart_robot_icl_gen15_survey_2026-08-25.md
  - ../../sources/blogs/wechat_mbot_embodied_icl_survey_2026-09.md
original:
  - ../../sources/papers/wam_ttt_arxiv_2607_06988.md
  - ../../sources/papers/robottt_arxiv_2607_15275.md
  - ../../sources/papers/stellavla_arxiv_2608_11671.md
  - ../../sources/papers/zero_wam_arxiv_2608_26103.md
summary: "四篇可核对论文把「演示当提示」拆成快权重记忆（WAM-TTT/RoboTTT）与纯上下文（StellaVLA/Zero-WAM）两族；跨篇数字不可直接排序，应按漂移轴（场景/构型/扰动/任务/时长）与开源条件（Zero-WAM 代码权重与公开子集已开源，其余三篇未放完整栈）选型。"
---

# WAM-TTT × RoboTTT × StellaVLA × Zero-WAM：具身 ICL 四路线对比

## 一句话定义

**2026 年学术侧最接近产业「演示当提示」叙事的四条可核对路线**：两条 **快权重 TTT**（感知侧技能记忆 vs 动作侧工作记忆）与两条 **零梯度上下文**（结构化语言示范 vs 合成人类视频任务规格）；**横向表只做定性定位，不做跨协议性能排序**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ICL | In-Context Learning | 部署期适应；本页四路线中仅 StellaVLA 与 Zero-WAM 为完全零梯度 |
| TTT | Test-Time Training | 测试时对快权重做梯度更新，主干冻结 |
| KVM | Key-Value Memory | WAM-TTT 记忆重建损失；附录证明等价于线性注意力 |
| IFP | In-context Future chunk Prediction | Zero-WAM 反捷径训练目标（开源代码中对应 MCP 模块） |
| ICIL | In-Context Imitation Learning | 检索示范作前缀的模仿学习框架 |

## 为什么重要

- **名词过载：** 2026 年「上下文」同时指 π0.7 metadata、[RoboTTT](../entities/paper-robottt-test-time-training-vla-context.md) 8K fast weights、[StellaVLA](../entities/paper-stellavla-structured-icl-vla.md) 结构化前缀与 [Zero-WAM](../entities/paper-zero-wam.md) 人视频规格——混用会误判机制与部署代价。
- **产业空白：** [GEN-1.5](../entities/generalist-gen15-one-shot.md) / [S1](../entities/skild-s1.md) 宣称涌现式 one-shot（闭源）；四篇学术论文共同反证在可负担算力规模上 **需要显式反捷径与记忆机制**。
- **零直接对比：** 四篇各自独立发表且 **无同 backbone 并排实验**；本页建立可核对的多维坐标系，辅助按漂移轴选型而非跨协议刷榜。

## 核心原理

### 坐标系：演示改什么 × 何时适应

```mermaid
flowchart TB
  subgraph weight["改快权重（TTT，主干冻结）"]
    WAM["WAM-TTT：感知侧 · 部署前吸收"]
    RT["RoboTTT：动作侧 · 每步递推"]
  end
  subgraph ctx["改输入上下文（完全零梯度 ICL）"]
    ST["StellaVLA：结构化语言示范"]
    ZW["Zero-WAM：合成人类视频规格"]
  end
  WAM -->|"技能记忆，rollout 前冻结"| freeze1["部署执行"]
  RT -->|"工作记忆，每步更新"| stream["长程 rollout"]
  ST -->|"KV-cache 前缀"| freeze2["部署执行"]
  ZW -->|"视频分支前缀 + ΔH RoPE"| freeze3["部署执行"]
```

| 论文 | 演示落点 | 适应时机 | 梯度更新 | 实体页 |
|------|----------|----------|----------|--------|
| [WAM-TTT](../entities/paper-wam-ttt-human-video-test-time-steering.md) | 冻结 WAM **video 专家** fast weights | 部署前批次 TTT（默认 1 步 SGD） | 零**主干**梯度 | arXiv:2607.06988 |
| [RoboTTT](../entities/paper-robottt-test-time-training-vla-context.md) | **DiT 动作头** fast weights | 每步 visuomotor 递推 | 零**主干**梯度 | arXiv:2607.15275 |
| [StellaVLA](../entities/paper-stellavla-structured-icl-vla.md) | 检索 **结构化语言** 示范前缀 | 无 | **完全零梯度** | arXiv:2608.11671 |
| [Zero-WAM](../entities/paper-zero-wam.md) | **人类视频** 作视频分支上下文 | 无 | **完全零梯度** | arXiv:2608.26103 |
| [HOST](../entities/paper-host-one-shot-human-video.md) | **真人视频** + 进度流形 / 自接地未来观测 | 无 | **完全零梯度** | arXiv:2607.20033 |

### 十二维对照（定性，非排名）

| 维度 | WAM-TTT | RoboTTT | StellaVLA | Zero-WAM |
|------|---------|---------|-----------|----------|
| **ICL 口径** | 测试时训练；反对纯上下文堆叠 | 上下文长度 scaling；ICL 为长上下文副产品 | 标准 ICIL；创新在示范**表示** | 零样本跨任务 = 任务规格问题 |
| **上下文模态** | 真实 GoPro 人视频（无姿态） | 人视频 / 自身 rollout / DAgger 失败（掩码统一） | 子目标 + 2D/3D 运动文本 + 关键帧 | **合成**人类操作视频 |
| **泛化主轴** | **未见场景**（家庭/厨房/办公室） | **时长/阶段**（~5 min 十阶段） | **扰动轴**（LIBERO-Plus、VLA-Arena L1/L2） | **未见任务**（RoboTwin 留出） |
| **代表数字** | New 9 任务 progress **46.2%** | 长程装配完成分 **~79 vs ~42** | LIBERO **98.8%**；VLA-Arena **0.63** | 未见 7 任务 **46.95%** |
| **指标注意** | **progress 部分给分** | rubric 完成分 + 部分二元 SR | 饱和仿真 SR | 3 种子 × 100 rollout SR |
| **开源状态（2026-11 实核）** | 无代码/无独立页 | 仅项目页 + arXiv | 无官方代码（对照用 StarVLA） | **已开源（Apache-2.0 代码、pretrain/posttrain 权重与公开 HumanGen 潜变量）** |

> **读表规则：** 46.2、79、98.8、46.95 **不可并排排名**——分别对应未见场景 progress、固定场景 rubric、饱和仿真 SR、未见任务 SR。

### 漂移轴地图（互补失败）

四篇各守一条漂移轴，尚无单一方法同时覆盖全部维度：

| 漂移轴 | 强项 | 典型盲区 |
|--------|------|----------|
| **场景/视觉** | WAM-TTT（真实野外人视频） | RoboTTT 固定场景相机 |
| **时长/阶段** | RoboTTT（8K 预训练上下文） | StellaVLA Long Horizon L1/L2 ≈ 0 |
| **扰动/规格** | StellaVLA（LIBERO-Plus + 三向干预） | 物体布局增益仅 +0.1 |
| **未见任务** | Zero-WAM（任务级切分） | 未见环境外观与强接触纠偏较弱（Stack three blocks 9.00%） |
| **秒级单视频 + 不遗忘** | [HOST](../entities/paper-host-one-shot-human-video.md)（开源管线） | 单平台 ARX；不按 L0–L3 报 |

**结构性事实：** 四篇里 **仅 WAM-TTT** 用真实、跨域人类视频；StellaVLA 与 Zero-WAM 均用 VLM **把机器人轨迹转换** 为另一模态（文本结构 vs 合成视频）。

### 跨篇可带走的三条经验

1. **人类信息只改感知/生成侧** — WAM-TTT（video 专家）、Zero-WAM（视频分支，且在 `ICLAttentionBackend` 中屏蔽动作流直连人类 KV）；RoboTTT 的 TTT 在动作头但上下文是**机器人自身**历史，不违反此条。
2. **训练期挂辅助、推理期剥离** — StellaVLA spatial-language expert（剥离后 91 ms vs 3177 ms）；Zero-WAM IFP/MCP（[3,11,19,29] 中间层预测，推理剥离）；辅助目标须逼主分支学 in-context 信号。
3. **上下文会走捷径** — RoboTTT +1 历史帧反而更差；Zero-WAM 无 IFP 时人视频为净负（28.55% < 39.44%）；StellaVLA Image-only 分内好但 OOD 差（75.7% vs 84.4%）。

### 后续补充：ICI-VLA（2026-09）

[ICI-VLA](../entities/paper-ici-vla-spatiotemporal-icl.md)（[arXiv:2609.07581](https://arxiv.org/abs/2609.07581)）与 StellaVLA 同属 **零梯度纯 ICL**，但示范形态为 **子任务 micro-demo + DTW 相位检索**（text-action VLA-0 接口），而非结构化语言 verbalization。RoboTwin 2.0 平均 **60.4%**（朴素 ICL 仅 **10.7%**）表明 **检索对齐** 是核心杠杆；与上表四路线 **无同 backbone 并排**，作为重要对照点补充参考。

## 工程实践

| 你的瓶颈 | 优先看 | 原因与工程落地注意 |
|----------|--------|--------------------|
| 新家庭/厨房 OOD，有真实 egocentric 人视频 | [WAM-TTT](../entities/paper-wam-ttt-human-video-test-time-steering.md) | 唯一直面真实跨域人视频；需 meta-training 配对，且官方未开源 |
| 5 min 级多阶段装配，需部署后自纠偏 | [RoboTTT](../entities/paper-robottt-test-time-training-vla-context.md) | 8K 预训练上下文 scaling；DAgger Distillation 可拆用，目前无开源仓库 |
| 桌面 OOD 扰动，可检索同任务示范 | [StellaVLA](../entities/paper-stellavla-structured-icl-vla.md) | 三向干预证明真用上下文；重 OOD 可试 λ=0；无官方代码，可参考 StarVLA |
| 仿真/真机 **未见任务**，需可直接运行的基线 | [Zero-WAM](../entities/paper-zero-wam.md) | **唯一已开源完整训练/RoboTwin 评测代码的四路线节点**（Apache-2.0），注意 `INSTALL.md` 两项任务判定更新及 `IFP` 不可省 |
| 要开源单视频、不改权重、可下权重 | [HOST](../entities/paper-host-one-shot-human-video.md) | 代码与权重公开，注意真机包与开环评测口径差异 |

## 局限与风险

- **开源完整度不均（2026-11-19 再核）：** [Zero-WAM](../entities/paper-zero-wam.md) 现已开源 Apache-2.0 代码、`zero-wam-pretrain`/`zero-wam-posttrain-robotwin` 权重与 RoboTwin/公开外部 `HumanGen` 潜变量，但 **HumanGen 视频合成脚本与内部/真机数据子集仍未公开**；WAM-TTT、RoboTTT、StellaVLA 均无完整公开代码与权重，三篇数字仍只能采信论文自报。
- **零同骨干并排实验：** 截至目前尚无同一 VLA/WAM 骨干下把这四种上下文接口做统一基准对比的公开消融研究。
- **产业规模未覆盖：** 四篇证明的是「可负担规模需显式机制」，并不能否定 GEN-1.5 / S1 级超大预训练规模下的涌现假说。

## 与其他页面的关联

- [ICL 纵深路线](../../roadmap/depth-icl.md) — 本页坐标系对应 Stage 4「零梯度上下文 vs 快权重 TTT vs 记忆增强」的选型环节
- [Query：具身大模型分类学选型闭环](../queries/embodied-fm-taxonomy-loop.md) — 本页四路线都落在闭环的「执行（VLA）× 推演（WM）」两层
- [机器人 In-Context Learning](../concepts/robot-in-context-learning.md) — 真 ICL vs TTT vs 映射选择 taxonomy
- [In-Context Learning for Robots 综述](../entities/paper-in-context-learning-robots-survey.md) — arXiv:2609.36012 四类接口分类总图
- [VLA](../methods/vla.md) — 长程记忆与部署期适应
- [Manipulation](../tasks/manipulation.md) — 四实体索引入口
- [GEN-1.5](../entities/generalist-gen15-one-shot.md) / [S1](../entities/skild-s1.md) — 产业闭源对照
- [HOST](../entities/paper-host-one-shot-human-video.md) — 开源零梯度单视频；进度对齐 + 自接地
- [The Imitator Game](../entities/paper-imitator-game.md) — 意图级模仿基准，不是方法路线

## 参考来源

- [WAM-TTT 论文归档（arXiv:2607.06988）](../../sources/papers/wam_ttt_arxiv_2607_06988.md)
- [RoboTTT 论文归档（arXiv:2607.15275）](../../sources/papers/robottt_arxiv_2607_15275.md)
- [StellaVLA 论文归档（arXiv:2608.11671）](../../sources/papers/stellavla_arxiv_2608_11671.md)
- [Zero-WAM 论文归档（arXiv:2608.26103）](../../sources/papers/zero_wam_arxiv_2608_26103.md)
- [Zero-WAM 开源仓库实核](../../sources/repos/zero-wam.md)
- [In-Context Learning for Robots 综述（arXiv:2609.36012）](../../sources/papers/in_context_learning_robots_arxiv_2609_36012.md)
- [每日智能四篇 ICL 纵横向解读（2026-08-31）](../../sources/blogs/wechat_meiri_zhineng_embodied_icl_four_papers_2026-08-31.md)
- [具身智能之心 ICL 综述（2026-08-25）](../../sources/blogs/wechat_embodied_heart_robot_icl_gen15_survey_2026-08-25.md)
- [Embodied ICL 调研综述（Mbot，2026-09）](../../sources/blogs/wechat_mbot_embodied_icl_survey_2026-09.md)

## 推荐继续阅读

- 四篇原文：[WAM-TTT (2607.06988)](https://arxiv.org/abs/2607.06988) · [RoboTTT (2607.15275)](https://arxiv.org/abs/2607.15275) · [StellaVLA (2608.11671)](https://arxiv.org/abs/2608.11671) · [Zero-WAM (2608.26103)](https://arxiv.org/abs/2608.26103)
- Zero-WAM 开源代码：<https://github.com/robbyant-research/Zero-WAM>
- ICI-VLA（补充坐标）：[2609.07581](https://arxiv.org/abs/2609.07581)
