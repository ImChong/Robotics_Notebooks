---
type: entity
tags: ['paper', 'humanoid', 'tsinghua', 'galbot', 'social-hri', 'co-speech-gesture', 'diffusion', 'unitree-g1']
status: complete
updated: 2026-09-07
arxiv: "2608.28693"
summary: "RoboGesture（arXiv:2608.28693，清华/银河通用等）：300+ 类手势 + 半合成 1000 h 数据；分层语义–声学对齐 + DiT-CFM + Anti-Inertia Masking；G1 真机 ≈120 FPS；项目页未见代码。"
related:
  - ../tasks/loco-manipulation.md
  - ./paper-pamor.md
  - ./unitree-g1.md
  - ../methods/diffusion-motion-generation.md
sources:
  - ../../sources/papers/robogesture_arxiv_2608_28693.md
  - ../../sources/sites/robogesture-arxiv.md
---

# RoboGesture：人形实时语义对齐伴随语音手势

**RoboGesture**（[arXiv:2608.28693](https://arxiv.org/abs/2608.28693)）由 **清华大学、银河通用机器人（Galbot）、北理工、哈工大、北大、上海期智研究院** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)）。

## 一句话定义

让人形从 **听** 到 **做手势** 走 **原始音频 token → 机器人运动** 的端到端流式管线，并用 MPC 保证真机无碰撞。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DiT | Diffusion Transformer | 连续运动空间的扩散 Transformer |
| CFM | Conditional Flow Matching | 条件流匹配生成运动块 |
| CFG | Classifier-Free Guidance | 本文 Anti-Inertia 变体防历史惯性 |
| HRI | Human-Robot Interaction | 社交人机交互 |

## 为什么重要

文本中间表示丢韵律；纯运动惯性会让模型 **忽视音频**；avatar→retarget 在线不安全——RoboGesture 在 **机器人空间** 联合设计数据、模型与控制。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 清华大学、银河通用机器人（Galbot）、北理工、哈工大、北大、上海期智研究院 |
| **开源** | 见 [工程实践](#工程实践) |

## 核心原理

三模块：**Hierarchical Semantic-Acoustic Aligner**（Mimi codec 多粒度 token + beat/300 类语义头）、**Streaming DiT+CFM**（Cross-Attn 微对齐 + FiLM 宏调制；15% 历史 mask）、**MPC 安全滤波**（5.6 ms/帧）。平台：G1 + BrainCo 灵巧手，41 DoF 上身。

### 流程总览

```mermaid
flowchart LR
  audio[流式语音] --> align[语义-声学对齐器]
  align --> dit[DiT-CFM 运动生成]
  hist[历史运动] --> dit
  dit --> mpc[MPC 安全滤波]
  mpc --> g1[Unitree G1]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-07** 无可运行官方代码（或本文为硬件/协议类工作）。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 见论文摘录与项目页核查结论 |
| 复现入口 | 以 arXiv 为准 |

## 实验与评测

| 基准 | FGD↓ | Col.↓ | 读法 |
|------|------|-------|------|
| BEAT | **0.845** | 0.88% | 优于 LivelySpeaker/DiffSHEG 等 |
| SemanticBEAT | BC **0.295** | **0.13%** | 语义手势对齐 SOTA |

## 结论

RoboGesture 把 **语义–韵律–安全** 绑成可部署的 listen–respond–gesture 闭环；Anti-Inertia Masking 是打破「运动抄历史」的关键训练技巧。

1. **300+** 手势类 + **1000 h** 半合成训练对。
2. 真机四场景：社交协助/情感支持/知识分享/办公沟通。
3. 与 Speech-LLM 管线级联；瓶颈在上游语音而非运动。
4. ECCV 2026 Poster。
5. 截至入库日 **未开源**。

## 局限与风险

上身为主；未覆盖全身行走协同；数据合成依赖 LLM 场景与 TTS。

## 关联页面

- [loco-manipulation](../tasks/loco-manipulation.md)
- [paper-pamor.md](./paper-pamor.md)
- [unitree-g1.md](./unitree-g1.md)
- [diffusion-motion-generation](../methods/diffusion-motion-generation.md)

## 参考来源

- [robogesture_arxiv_2608_28693.md](../../sources/papers/robogesture_arxiv_2608_28693.md)
- [robogesture-arxiv.md](../../sources/sites/robogesture-arxiv.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)

## 推荐继续阅读

- [https://RoboGesture.github.io](https://RoboGesture.github.io)
