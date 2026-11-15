---
type: method
tags: [vla, foundation-policy, embodied-ai, qwen, benchmark]
status: complete
original:
  - ../../sources/papers/star_vla_alpha_arxiv_2604_11757_full_reading_2026_11_15.md
  - ../../sources/papers/starvla_framework_arxiv_2604_05014_full_reading.md
updated: 2026-11-15
related:
  - ../overview/vla-open-source-repro-landscape-2025.md
  - ./vla.md
  - ../entities/paper-thaw-vla.md
  - ../entities/rldx-1.md
  - ../concepts/foundation-policy.md
  - ../entities/nvidia-omniverse.md
  - ./diffusion-policy.md
  - ../entities/paper-objdream.md
summary: "StarVLA 是模块化 VLA 基准框架，以强 VLM 配合轻量动作头作为对照；具体控制表现须按模型版本、训练数据和评测协议比较。"
sources:
  - ../../sources/papers/star_vla.md
  - ../../sources/papers/star_vla_alpha_arxiv_2604_11757_full_reading_2026_11_15.md
  - ../../sources/papers/starvla_framework_arxiv_2604_05014_full_reading.md
---

# StarVLA 方法入口：受控基线与模块化实验

## 一句话定义

StarVLA 方法路线是先用统一数据/动作接口建立VLM到动作的简单基线，再控制骨干、动作头和训练配方进行比较；[α论文](../entities/paper-starvla-alpha.md)给受控实验，[框架论文](../entities/paper-starvla-framework.md)给实现基础设施。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLM | Vision-Language Model | 表征图像与指令的预训练骨干 |
| VLA | Vision-Language-Action | 输出机器人动作的策略 |
| MLP | Multi-Layer Perceptron | 连续动作回归的简单基线 |
| SFT | Supervised Fine-Tuning | 以示范动作监督策略训练 |
| OOD | Out-of-Distribution | 相对于训练分布的测试条件 |

## 为什么重要

不同VLA系统的成绩常同时改变骨干、数据、动作格式和评测环境。统一接口使比较更可解释；简单基线能帮助判断额外结构的收益是否超出训练预算或数据差异。

## 核心原理

原始RGB和语言经VLM编码，动作token隐藏状态交给动作头，输出动作chunk后由本体适配器执行。α的默认是Qwen3-VL-4B和轻量MLP；框架还能替换FAST、PI流匹配和GR00T双系统。动作补零只统一张量形状，不统一物理单位、关节含义或控制频率。

## 主要方法路线

| 动作头路线 | 接口与用途 | 比较时固定的条件 |
|---|---|---|
| OFT 连续回归 | MLP 预测动作 chunk，作为 α 的简单基线 | VLM、数据、有效动作维与归一化 |
| FAST 自回归 | 将动作离散化后生成 token | tokenizer、解码长度与推理预算 |
| PI 流匹配 | 对连续动作块建模 | 采样步数、动作时域与训练预算 |
| GR00T 双系统 | 组合视觉语言与动作生成模块 | 骨干规模、动作头及跨模块接口 |

这些路线由框架提供可替换接口；α 的受控结果集中在对应数据与训练条件下，不能把支持列表当作任意模型头已获得同样成绩。

## 工程实践

1. 固定骨干、数据划分、动作归一化、chunk长度与评测种子，先跑单基准。
2. 比较动作头时保持其他因素相同，分别记录成功率、推理延迟和资源成本。
3. 联合训练时保存数据配比与本体适配规则，明确哪些测试基准已出现在训练中。
4. 当前官方LIBERO脚本默认QwenPI/Qwen3.5-0.8B；要复现α，必须显式匹配Qwen3-VL-4B+MLP的配置。模型服务和仿真客户端分别检查，不能从一次smoke推断全基准复现。

## 实验读法

α的specialist LIBERO98.8%和Google VM76.0%来自各自训练；generalist对应97.8%和74.3%。ARX5的SR为33.6%对π0.5的12.7%，差20.9个百分点；进度分是另一指标。逐表协议、训练设置和代码边界集中在[α详情页](../entities/paper-starvla-alpha.md)，避免在方法入口混用多套成绩。

## 局限与风险

强VLM加MLP是已测设置下的基线，不是所有动态/接触任务的最优结构。轻量头不消除VLM推理开销；框架支持列表不代表每个训练分支已完整发布，RL adaptation在已读指南中仍为待办。OOD成绩限于原文测试条件。

## 关联页面

- [模型架构纵深路线](../../roadmap/depth-model-architecture.md)

- [StarVLA-α 独立论文详情](../entities/paper-starvla-alpha.md)


- [StarVLA 框架论文](../entities/paper-starvla-framework.md) — 模块化训练和评测基础设施。
-   [VLA 开源复现景观（2025）](../overview/vla-open-source-repro-landscape-2025.md) — 含 starVLA/starVLA 仓库与单卡训练语境
-   [VLA (Vision-Language-Action)](./vla.md) — 具身大模型全景
-   [RLDX-1](../entities/rldx-1.md) — 同 Qwen3-VL 底座下的多流扩散动作头与物理传感路线对照
-   [Qwen-VLA](../entities/qwen-vla.md) — 同生态下的 **大规模通才**（操作–导航统一 + embodiment prompt）对照
-   [VLAct](../entities/paper-vlact.md) — StarVLA 生态下的 **表征中心 VLA 持续预训练** 骨干（arXiv:2608.27550）
-   [THAW-VLA](../entities/paper-thaw-vla.md) — 在 StarVLA QwenGR00T 0.8B 上离线蒸馏 Cosmos3-Nano 特征；LIBERO 97.9%（arXiv:2609.24682；已开源）
-   [Foundation Policy](../concepts/foundation-policy.md) — 基础策略模型概念
-   [Diffusion Policy](./diffusion-policy.md) — 另一种主流的生成式动作头路线
- [ObjDream](../entities/paper-objdream.md) — 物体中心 VLA：时空物体槽分离身份与状态，身份引导路由 VLM 注意力

## 参考来源

- [StarVLA-α 主论文全文与代码核对](../../sources/papers/star_vla_alpha_arxiv_2604_11757_full_reading_2026_11_15.md)
- [StarVLA 框架论文全文与官方指南](../../sources/papers/starvla_framework_arxiv_2604_05014_full_reading.md)
- [官方仓库索引](../../sources/repos/star_vla.md)

## 推荐继续阅读

- [α论文与附录](https://arxiv.org/html/2604.11757)
- [框架论文与附录](https://arxiv.org/html/2604.05014v1)
- [官方代码与指南](https://github.com/starVLA/starVLA)
