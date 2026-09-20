---
type: overview
tags: [wechat-curator, lumina, embodied-ai-guide, llm-planner, vla, diffusion-policy, manipulation]
status: complete
updated: 2026-09-20
related:
  - ../entities/lumina-embodied.md
  - ../methods/vla.md
  - ../methods/diffusion-policy.md
  - ../methods/saycan.md
  - ../concepts/llm-robotics-control-interfaces.md
  - ../overview/vla-evolution-lineage.md
  - ../entities/paper-diffusion-policy.md
  - ../entities/paper-act.md
  - ../entities/paper-pi0.md
sources:
  - ../../sources/raw/wechat_lumina_embodied_practice_album_4608355279393816579.md
  - ../../sources/blogs/wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md
  - ../../sources/blogs/wechat_lumina_embodied_practice_part2_code_as_policy_2026-09-20.md
  - ../../sources/blogs/wechat_lumina_vla_survey_part1_2026-09-20.md
  - ../../sources/blogs/wechat_lumina_vla_survey_part2_2026-09-20.md
  - ../../sources/blogs/wechat_lumina_diffusion_policy_primer_2026-09-20.md
  - ../../sources/repos/embodied-ai-guide.md
summary: "Lumina「机器人技术指南」微信专辑 5 篇的策展索引：LLM 规划、Code-as-Policy、VLA 路线（上下）、Diffusion Policy；文内项目一律映射到独立 wiki 实体，已有页只回链不重复造页。"
---

# Embodied-AI-Guide 微信专辑 — 五篇策展索引

## 一句话定义

本页把 Lumina **[Embodied-AI-Guide](../../sources/repos/embodied-ai-guide.md)** 配套的微信专辑（5 篇入门/综述）拆成**可点击的知识图谱入口**：每篇对应一个 blog 归档 + 一张项目映射表；算法细节仍以各 `paper-*` 实体与方法页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| CaP | Code as Policy | LLM 生成可执行程序作中间策略 |
| DP | Diffusion Policy | 扩散去噪生成动作 chunk |
| ACT | Action Chunking Transformer | Transformer 预测动作块 |
| LLM | Large Language Model | 高层规划与工具调用接口 |

## 专辑入口

| # | 标题 | 原文 | blog 归档 |
|---|------|------|-----------|
| 1 | LLM 做任务规划器 | [微信](https://mp.weixin.qq.com/s?__biz=MzA5NTM1OTgxNA==&mid=2247485105&idx=1&sn=6631b3b366acc2cde8e0d84e62070afc) | [part1](../../sources/blogs/wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md) |
| 2 | 代码即策略 | [微信](https://mp.weixin.qq.com/s?__biz=MzA5NTM1OTgxNA==&mid=2247485113&idx=1&sn=ac3e3e2031a9522e6cb71910d6d1bc5e) | [part2](../../sources/blogs/wechat_lumina_embodied_practice_part2_code_as_policy_2026-09-20.md) |
| 3 | VLA 综述 (一) | [微信](https://mp.weixin.qq.com/s?__biz=MzA5NTM1OTgxNA==&mid=2247485266&idx=1&sn=c3c42934e34f0d9dabd1a7655e6ec2b2) | [vla1](../../sources/blogs/wechat_lumina_vla_survey_part1_2026-09-20.md) |
| 4 | VLA 综述 (二) | [微信](https://mp.weixin.qq.com/s?__biz=MzA5NTM1OTgxNA==&mid=2247485308&idx=1&sn=bc5dbaad64792c7aabc5352276ca1abd) | [vla2](../../sources/blogs/wechat_lumina_vla_survey_part2_2026-09-20.md) |
| 5 | Diffusion Policy | [微信](https://mp.weixin.qq.com/s?__biz=MzA5NTM1OTgxNA==&mid=2247485404&idx=1&sn=f0b24d4d70afc9b4800e4c4adc5ac771) | [dp](../../sources/blogs/wechat_lumina_diffusion_policy_primer_2026-09-20.md) |

## 篇 1 · LLM 规划器 — 项目节点

| 项目 | 实体页 |
|------|--------|
| SayCan | [paper-saycan](../entities/paper-saycan.md) |
| PaLM-E | [paper-palm-e-embodied-language-model](../entities/paper-palm-e-embodied-language-model.md) |
| EmbodiedGPT | [paper-embodiedgpt](../entities/paper-embodiedgpt.md) |
| LBYL | [paper-look-before-you-leap](../entities/paper-look-before-you-leap.md) |
| RT-2 | [paper-rt-2](../entities/paper-rt-2.md) |
| LLM+P | [paper-llm-p](../entities/paper-llm-p.md) |
| AutoTAMP | [paper-autotamp](../entities/paper-autotamp.md) |
| Text2Motion | [paper-text2motion](../entities/paper-text2motion.md) |
| OpenVLA | [paper-openvla](../entities/paper-openvla.md) |
| Octo | [paper-octo](../entities/paper-octo.md) |

## 篇 2 · 代码即策略 — 项目节点

| 项目 | 实体页 |
|------|--------|
| Code as Policy | [paper-pai-2209-07753-codeaspolicies](../entities/paper-pai-2209-07753-codeaspolicies.md) |
| Instruction2Act | [paper-instruction2act](../entities/paper-instruction2act.md) |
| VoxPoser | [paper-voxposer](../entities/paper-voxposer.md) |
| OmniManip | [paper-omnimanip](../entities/paper-omnimanip.md) |

## 篇 3 · VLA 经典 — 项目节点

| 项目 | 实体页 |
|------|--------|
| RT-1 | [paper-rt-1](../entities/paper-rt-1.md) |
| RT-2 | [paper-rt-2](../entities/paper-rt-2.md) |
| OpenVLA | [paper-openvla](../entities/paper-openvla.md) |
| RoboFlamingo | [paper-pai-2311-01378-roboflamingo](../entities/paper-pai-2311-01378-roboflamingo.md) |
| Octo | [paper-octo](../entities/paper-octo.md) |
| π0 | [paper-pi0](../entities/paper-pi0.md) |
| CogACT | [paper-cogact](../entities/paper-cogact.md) |
| Diffusion-VLA | [paper-diffusion-vla](../entities/paper-diffusion-vla.md) |
| 3D-VLA | [paper-sa-2403-09631-3d-vla-a-3d-vision-language-action-generative-wo](../entities/paper-sa-2403-09631-3d-vla-a-3d-vision-language-action-generative-wo.md) |
| SpatialVLA | [paper-spatialvla](../entities/paper-spatialvla.md) |
| RDT-1B | [paper-rdt-1b](../entities/paper-rdt-1b.md) |

## 篇 4 · 分层 VLA 与 2025 工作

- **系统级：** [Helix](../entities/helix-25.md)、[Isaac GR00T](../entities/isaac-gr00t.md)、[Gemini Robotics](../entities/gemini-robotics.md)、[π0 / openpi](../entities/paper-pi0.md)
- **纵览对照：** [VLA 演进](../overview/vla-evolution-lineage.md)
- **滚动列表中尚无独立页的 2025 工作**（WorldVLA、UniVLA 等）标「待升格」，见 [vla2 blog](../../sources/blogs/wechat_lumina_vla_survey_part2_2026-09-20.md)

## 篇 5 · Diffusion Policy 三基线

| 基线 | 实体页 |
|------|--------|
| ACT | [paper-act](../entities/paper-act.md) |
| Diffusion Policy | [paper-diffusion-policy](../entities/paper-diffusion-policy.md) |
| DP3 | [painode-209-3ddiffusionpolicydp3](../entities/painode-209-3ddiffusionpolicydp3.md) |

## 常见误区

- **误区：LLM 直接输出关节力矩。** 专辑与 [Embodied-AI-Guide](../../sources/repos/embodied-ai-guide.md) 均强调分层：规划/代码/3D map 在上，VLA 或技能库在下。
- **误区：VLA = 更大的 VLM。** 动作表示与数据配方往往比换 backbone 更决定成败（见 [VLA 方法页](../methods/vla.md)）。

## 关联页面

- [Lumina 社区](../entities/lumina-embodied.md)
- [Embodied-AI-Guide 仓库归档](../../sources/repos/embodied-ai-guide.md)
- [LLM 机器人控制接口](../concepts/llm-robotics-control-interfaces.md)
- [VLA](../methods/vla.md)、[Diffusion Policy](../methods/diffusion-policy.md)

## 参考来源

- [wechat_lumina_embodied_practice_album_4608355279393816579.md](../../sources/raw/wechat_lumina_embodied_practice_album_4608355279393816579.md)
- [Embodied-AI-Guide algorithm.md](https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/topics/algorithm.md)

## 推荐继续阅读

- [Embodied-AI-Guide GitHub](https://github.com/TianxingChen/Embodied-AI-Guide)
- [Lumina 官网](https://lumina-embodied.ai/)
