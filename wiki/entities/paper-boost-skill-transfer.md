---
type: entity
tags: [paper, skill-transfer, vq-vae, imitation-learning, libero, snu, georgia-tech]
status: complete
updated: 2026-08-18
arxiv: "2608.10600"
related:
  - ../methods/imitation-learning.md
  - ./libero-benchmark.md
  - ./droid-policy-learning.md
  - ./paper-seeker.md
  - ../methods/diffusion-policy.md
sources:
  - ../../sources/papers/boost_skill_transfer_arxiv_2608_10600.md
  - ../../sources/sites/boost-robots.md
  - ../../sources/blogs/wechat_embodied_station_contact_predict_adapt_10_papers_2026-08-18.md
summary: "BooST（RA-L 2026，SNU/Georgia Tech）：跨模态 VQ-VAE 统一语义意图与运动动态，再蒸馏约 60 Hz 策略。LIBERO-90 10 demo 0.70。项目页已发，训练仓未开。"
---

# BooST：技能要同时记住「做什么」和「怎么动」

**BooST**（*Bridging Semantics and Motions for Efficient Skill Transfer*；[arXiv:2608.10600](https://arxiv.org/abs/2608.10600)，[项目页](https://boost-robots.github.io/)）由 **首尔大学 / 佐治亚理工** 提出（RA-L 2026）：可复用技能若只编码语义或只编码关节轨迹，下游少样本适应就会崩。

## 一句话定义

**先用跨模态 VQ-VAE 把指令语义和动作动态写入同一码本（只重建动作），再蒸馏成能在真机跑的轻量策略。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BooST | Bridging Semantics and Motions for Efficient Skill Transfer | 本文两阶段框架 |
| VQ-VAE | Vector Quantized VAE | 共享 codebook 的离散技能 |
| DROID | Distributed Robot Interaction Dataset | Stage I 预训练域（关节速度） |
| LIBERO | Lifelong Robot Learning benchmark | 下游笛卡尔空间评测 |
| BC | Behavior Cloning | Stage II 低层策略 |
| Hz | Hertz | 部署约 60 Hz |

## 为什么重要

- 跨域技能迁移的失败模式经常是：语义对了但动作空间变了，或动作像了但任务意图丢了。
- 像素重建会把背景行人吃进表征；BooST 只重建动作，所以对动态干扰更稳。
- LIBERO-90 在 10 条示范上相对次优 **+140%**，说明先验质量随数据变少更明显。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 首尔大学（SNU）；佐治亚理工学院（Georgia Tech） |
| **期刊** | IEEE RA-L 2026（项目页） |
| **开源** | **项目页已发，实现未开源** |

## 核心原理

### 方法栈

Stage I：CLIP ViT patch × 指令做 visuo-linguistic 通路，动作轨迹做 motion 通路，交替写 codebook。Stage II：冻编码器当教师，因果 skill prior + 小 Transformer 低层 BC；执行只看过去观测。

### 流程总览

```mermaid
flowchart LR
  droid["DROID 关节速度"]
  vq["跨模态 VQ-VAE"]
  dist["蒸馏轻量策略"]
  lib["LIBERO / UR3"]
  droid --> vq --> dist --> lib
```

## 工程实践

| 项 | 建议 |
|----|------|
| 源码运行时序图 | **不适用**（组织下仅 github.io） |
| 读表 | 必须看 10/20/50 demo 三档，只报 50 会掩盖先验价值 |
| 跨本体 | 低层基线在动作空间切换时会直接归零，这是卖点不是彩蛋 |

## 实验与评测

项目页 LIBERO-90：**0.91 / 0.82 / 0.70**（50/20/10 demo）。干扰物预训练后四套均分 **0.90**。真机：Franka 技能 → UR3，每任务 5 条示范，低层 QueST / VQ-BeT 失败。

## 与其他工作对比

相对 Diffusion Policy / VQ-BeT：本页多了语义通路与动作重建目标。相对 [Seeker](./paper-seeker.md)：Seeker 解决「看哪」，BooST 解决「技能码跨域」。相对 [DROID Policy Learning](./droid-policy-learning.md)：DROID 是数据入口，BooST 是其上的技能抽象。

## 结论

**少样本迁移要的是「语义+运动」共享离散技能，而不是更大的像素生成器。**

1. **只重建动作** — 这是抗干扰的主因。
2. **数据越少增益越大** — 10 demo 才是读表点。
3. **动作空间会变** — 预训练关节速度、评测笛卡尔，接口必须解耦。
4. **代码未开** — 数字以项目页为准，不能本地复现。

## 局限与风险

- 官方训练仓不存在，无法核对种子与预处理。
- 真机表是项目页视频/定性为主，完整百分比以论文为准。
- 共享 codebook 容量过小会合并不同技能。

## 关联页面

- [模仿学习](../methods/imitation-learning.md)
- [LIBERO](./libero-benchmark.md)
- [DROID Policy Learning](./droid-policy-learning.md)
- [Seeker](./paper-seeker.md)
- [Diffusion Policy](../methods/diffusion-policy.md)

## 参考来源

- [BooST 论文摘录](../../sources/papers/boost_skill_transfer_arxiv_2608_10600.md)
- [项目页归档](../../sources/sites/boost-robots.md)
- [具身智能小站 10 篇盘点（2026-08-18）](../../sources/blogs/wechat_embodied_station_contact_predict_adapt_10_papers_2026-08-18.md)

## 推荐继续阅读

- [BooST 项目页](https://boost-robots.github.io/)
- [arXiv:2608.10600](https://arxiv.org/abs/2608.10600)
