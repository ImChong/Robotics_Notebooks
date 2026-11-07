---
type: comparison
tags: [vlm, vln, vla, vlx, world-model, embodied-ai, multimodal, taxonomy]
status: complete
updated: 2026-11-07
related:
  - ../entities/paper-vgi-white-paper.md
  - ./robot-learning-five-paradigms-taxonomy.md
  - ../methods/vla.md
  - ../entities/paper-unified-robot-learning-survey.md
  - ../tasks/vision-language-navigation.md
  - ../concepts/world-action-models.md
  - ../concepts/functional-taxonomy-world-models.md
  - ../entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md
  - ../methods/generative-world-models.md
  - ../overview/hub-vla.md
  - ../overview/world-models-15-open-source-technology-map.md
  - ../concepts/humanoid-policy-network-architecture.md
  - ../methods/unified-multimodal-tokens.md
original:
  - ../../sources/papers/paligemma_arxiv_2407_07726.md
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
  - ../../sources/papers/cosmos_wfm_arxiv_2501_03575.md
  - ../../sources/papers/pi07.md
  - ../../sources/blogs/worldlabs_functional_taxonomy_world_models.md
sources:
  - ../../sources/blogs/wechat_shenlan_five_embodied_model_taxonomy.md
  - ../../sources/blogs/wechat_shenlan_robot_learning_five_paradigms.md
  - ../../sources/blogs/worldlabs_functional_taxonomy_world_models.md
  - ../../sources/papers/world_model_definition_roadmap_arxiv_2607_06401.md
summary: "VLM、VLN、VLA、VLX 与世界模型共享 Transformer 与多模态编码底座，按感知→导航→执行→融合→推演递进分工；VL 系列直接对接硬件，WM 专注时序虚拟预演并与 VLA 形成决策–预演闭环。"
---

# 五大具身模型分类：VLM、VLN、VLA、VLX 与世界模型

## 一句话定义

**五大具身模型** 是 2025–2026 年产业报告与论文中高频出现的 **功能分层缩写**：共享 **Transformer + 多模态编码** 底座，按 **跨模态理解 → 空间导航 → 动作执行 → 一体化多任务 → 时序推演** 递进分工；理解其 **输入/输出边界** 比死记缩写更能指导系统拆分与选型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉–语言跨模态理解与语义解析 |
| VLN | Vision-Language Navigation | 视觉–语言条件下的空间导航决策 |
| VLA | Vision-Language-Action | 视觉–语言–动作端到端执行策略 |
| VLX | Vision-Language X | 融合感知/导航/执行的多分支通用架构（X=可扩展任务） |
| WM | World Model | 环境时序动态建模与虚拟推演，不直接驱动物理执行 |

## 为什么重要

- **术语爆炸**：VLM/VLN/VLA 常被混用或「端到端一体化」叙事掩盖模块边界，导致架构设计与数据接口错位。
- **系统拆分**：真实机器人栈常需 **感知塔（VLM）→ 导航/操作头（VLN/VLA）→ 仿真预演（WM）** 分层部署；厘清 I/O 有助于 Sim2Real 与延迟预算。
- **趋势判断**：分立专精模型、VLX 式统一大模型、WM 赋能三条路线并行——本分类是阅读 [VLA 开源谱系](../overview/vla-open-source-repro-landscape-2025.md) 与 [世界模型地图](../overview/world-models-15-open-source-technology-map.md) 的前置框架。

## 一手资料核验（2026-11-07）

「VLM / VLN / VLA / VLX / WM」这套五分法**出自产业二手文章**（深蓝具身智能公众号），不是某篇论文提出的标准定义。本页用已通读的一手论文给每个家族找"代表样本"，逐条核对二手叙述：

| 家族 | 用来核对的一手样本 | 核对结论 |
|------|--------------------|----------|
| **VLM** | [PaliGemma](../entities/paper-rcl-2407-07726-paligemma-a-versatile-3b-vlm-for-transfer.md)（arXiv:2407.07726） | 与"无动作、纯认知层"一致：输入图像 + 文本前缀，输出文本后缀（含检测框 / 分割 token 的文本化表示），< 3B。论文本身不含机器人内容。 |
| **VLA** | [GR00T N1](../entities/paper-hrl-stack-34-gr00t_n1.md)（arXiv:2503.14734）、[π₀.₇](../methods/pi07-policy.md)（arXiv:2604.15483） | 与"整合 VLM、观测到动作"一致：N1 以 Eagle-2 VLM（System 2）+ DiT flow-matching 动作头（System 1）联合训练，输入图像 + 语言 + 本体状态，输出 16 步动作块。但二手文章说的"单网闭环"需放宽：N1 是**同一模型内两个模块**；π₀.₇ 在部署时是 **VLA + 同架构高层语言策略 + 独立世界模型**组成的系统。 |
| **WM** | [Cosmos 1.0](../entities/paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi.md)（arXiv:2501.03575） | 与"观测 + 扰动（动作/文本）→ 未来观测、不直接驱动硬件"一致。但"VLA 出候选动作 → WM 逐帧推演 → 择优执行"这条级联**在 Cosmos 1.0 里只是列出的五类用途之一，论文明确写"本文不含这些用途的实证结果"**；真实出现的是别的接法：π₀.₇ 用轻量世界模型（BAGEL 14B 初始化）生成**子目标图像**喂给 VLA；GR00T N1 用视频生成模型生成 827 小时**神经轨迹**当训练数据。 |
| **VLN** | 本次未通读任何 VLN 一手论文 | "VLN ⊂ VLA""仅底盘移动、无操作分支"等说法**未经一手核实**，只是二手文章的叙述（推测：对多数 VLN 基准成立，但不应当成定义）。 |
| **VLX** | 本次未通读任何以 VLX 自称的一手论文 | VLX 是二手文章归纳的"愿景型"名词，**未经一手核实**；一手资料里更常见的是 VLA / WAM 等名字。 |
| 功能轴对照 | [Fei-Fei / World Labs 博客](../concepts/functional-taxonomy-world-models.md) | 博客把 VLA、基于模型的系统与 World Action Models 都归为 **Planner**（输出动作），世界模型按输出分 Renderer（观测）/ Simulator（状态）；这说明"VLA 与 WM 同属世界模型家族还是互为上下游"取决于所用轴，与本页的 I/O 家族轴正交。 |

## 核心原理：统一技术底座

五类模型 **不存在本质架构割裂**：

1. **多模态混合编码**：图像、文本、深度/空间、本体状态、时序信息 → 专属编码器 → 共享隐向量空间对齐。
2. **Transformer 主干**：自注意力完成全局关联；差异来自 **模态配比、任务头、时序跨度、分支配置**。
3. **WM 特殊定位**：技术栈与其余四类一致，但 **运行方式** 为虚拟推演，输出不直接下发硬件。

## 递进关系与能力边界

```mermaid
flowchart TB
  vlm["VLM<br/>感知·语义理解"]
  vln["VLN<br/>空间导航"]
  vla["VLA<br/>端到端动作"]
  vlx["VLX<br/>多分支一体化"]
  wm["WM<br/>时序虚拟推演"]

  vlm --> vln
  vln --> vla
  vlm --> vla
  vla --> vlx
  vln --> vlx
  vlm --> vlx
  vla -.->|候选动作| wm
  wm -.->|择优反馈| vla
```

| 模型 | 核心 I/O | 能力边界 | 典型下游 |
|------|----------|----------|----------|
| **VLM** | 图像/视频 + 语言 → 语义、物体关系、指令解析 | **无动作、无轨迹**；纯认知层 | VLN、VLA 视觉塔、[3D VQA](../concepts/3d-spatial-vqa.md) |
| **VLN** | VLM 输入 + 深度/拓扑 → 路径、避障、目标坐标 | **仅底盘移动**；无力控/操作分支 | 室内/户外 [VLN 任务](../tasks/vision-language-navigation.md) |
| **VLA** | 全模态 + 本体状态 → 关节/底盘/末端控制量 | 感知–决策–执行 **单网闭环** | [manipulation](../tasks/manipulation.md)、[loco-manip](../tasks/loco-manipulation.md) |
| **VLX** | 同 VLA 输入 → **并行** 感知/导航/动作三路输出 | 单模型多任务；可按场景启停分支 | 通用人形「一脑多能」叙事 |
| **WM** | 观测 + 候选动作 → 未来多帧状态/物理反馈 | **长时序推演**；不即时执行 | [生成式世界模型](../methods/generative-world-models.md)、Sim 数据增强 |

### VLN ⊂ VLA（文内强调）

- **共享**：视觉编码器、语言编码器、跨模态融合主干。
- **VLN**：仅空间移动分支，训练目标为路径与避障。
- **VLA**：完整搭载 VLN 导航能力，并扩展机械臂、力控、全身协同等 **物理交互** 分支。

## VLA × 世界模型：决策–预演闭环

行业常见组合（与 [World Action Models](../concepts/world-action-models.md) 的联合建模路线不同，属 **级联** 范式）：

1. **VLA** 基于当前观测生成多组 **瞬时** 可执行动作候选。
2. **WM** 在虚拟空间对每组候选 **逐帧推演** 环境变化与风险。
3. 筛选最优方案后在真机执行，降低长流程任务试错成本。

二者 **多模态编码与物理规律学习可互通**，便于特征与数据协同；WM 也可独立服务数据生成、策略评估与 [Sim2Real](../concepts/sim2real.md) 闭环。

## 工程实践

1. **用 I/O 契约而不是缩写做接口评审。** 每个模块写清：输入模态、输出是文本 / 动作块 / 未来观测；例如 GR00T N1 的动作头输出 $H=16$ 的动作块，PaliGemma 输出文本。
2. **VLM 骨干既可冻结也可联合训练，别照搬二手文章的"通常冻结"。** PaliGemma 论文的消融显示预训练时**不冻结**图像编码器与语言模型更好；GR00T N1 后训练冻结 VLM 的语言部分、其余可训；GR00T N1.5 则全程冻结 VLM（项目页）。
3. **世界模型接入点至少有三种：** 数据生成（N1 的神经轨迹）、子目标生成（π₀.₇ 的世界模型生图）、候选动作评估（Cosmos 1.0 列为愿景，未验证）。选型时先问接入点。
4. **延迟预算分层核算。** π₀.₇ 论文数字：最小变体 VLA 推理约 38 ms，开启历史与子目标图最坏约 127 ms，子目标生图 25 步约 1.25 s（4 卡）——世界模型环节常在慢速异步线程里，不进控制主回路。

## 常见误区或局限

1. **「VLA 已包含一切，VLN/VLM 过时。」** 工程上仍常 **冻结 VLM 骨干**、单独训导航头或分层部署以满足延迟与可解释性。
2. **「WM = 仿真器。」** WM 是学习得到的 **可微/可生成** 动态模型，可与传统物理引擎互补，而非简单替代。
3. **「VLX 已量产。」** VLX 多为 **架构愿景**；当前开源主力仍是分立 VLA + 专用 WBC/导航模块。
4. **术语融合加速**：WAM（[World Action Models](../concepts/world-action-models.md)）等新概念把 WM 与动作生成 **耦合**，与本页「分立 WM + VLA」级联叙事并存——选型时需看清 **联合训练 vs 事后串联**。

## 关联页面

- [世界模型功能分类](../concepts/functional-taxonomy-world-models.md)：按 POMDP **输出**（观测 / 状态 / 动作）划分，与本页 VL* I/O 家族正交；本页的 WM 推演层在那边多半是 Renderer 或 Simulator。
- [世界模型定义与路线图](../entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md)：再加表征轴，并声明 WAM 不是第四实现列。
- [机器人学习五大范式](./robot-learning-five-paradigms-taxonomy.md)：按 **学习信号**（示范/奖励/视频/多模态/持续学习）划分的正交 taxonomy；本页按 **模型族 I/O** 划分。
- [Query：具身大模型分类学选型闭环知识链](../queries/embodied-fm-taxonomy-loop.md)：把本页五大家族沉淀为「感知 → 导航 → 执行 → 扩展 → 推演」的端到端选型决策链，逐层给出 I/O 边界、数据需求与实时性/泛化取舍。
- [VLA 方法页](../methods/vla.md)：执行层代表方法与训练数据。
- [统一机器人学习综述](../entities/paper-unified-robot-learning-survey.md)：I/O 分层之外，用六种耦合诊断系统病（TMLR 2026）。
- [VLN 任务页](../tasks/vision-language-navigation.md)：导航基准与开源复现范式。
- [统一多模态 token](../methods/unified-multimodal-tokens.md)：VLX/端到端大模型的表征接口。
- [人形策略网络架构](../concepts/humanoid-policy-network-architecture.md)：全身控制与高层 VLA 分层。
- [Visual General Intelligence 白皮书](../entities/paper-vgi-white-paper.md)：视觉优先对照——VGI 主张视觉可以是组织原则，而不是只当 VL 系列的输入通道。

## 推荐继续阅读

- 深蓝具身智能原文（微信公众号）：<https://mp.weixin.qq.com/s/xj-rc6v64Ge6onoUPvkHLg>
- [VLA Open-Source Landscape 2025](../overview/vla-open-source-repro-landscape-2025.md)
- [世界模型 15 项开源技术地图](../overview/world-models-15-open-source-technology-map.md)
- [具身大模型分类学选型闭环（知识链枢纽）](../overview/hub-embodied-foundation-model.md) — 把本页五大家族沉淀为一条贯通的选型链

## 参考来源

- **一手论文（已通读，2026-11-07）：** [PaliGemma](../../sources/papers/paligemma_arxiv_2407_07726.md)、[GR00T N1](../../sources/papers/gr00t_n1_arxiv_2503_14734.md)、[Cosmos 1.0](../../sources/papers/cosmos_wfm_arxiv_2501_03575.md)、[π₀.₇](../../sources/papers/pi07.md)、[World Labs 功能分类博客](../../sources/blogs/worldlabs_functional_taxonomy_world_models.md)
- [wechat_shenlan_five_embodied_model_taxonomy.md](../../sources/blogs/wechat_shenlan_five_embodied_model_taxonomy.md) — 深蓝具身智能《五大具身模型详解：VLM、VLA、VLN、VLX、世界模型》（<https://mp.weixin.qq.com/s/xj-rc6v64Ge6onoUPvkHLg>）
- [worldlabs_functional_taxonomy_world_models.md](../../sources/blogs/worldlabs_functional_taxonomy_world_models.md) — Fei-Fei / World Labs 功能分类
- [world_model_definition_roadmap_arxiv_2607_06401.md](../../sources/papers/world_model_definition_roadmap_arxiv_2607_06401.md) — 功能×架构二维表
