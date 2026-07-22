---
type: entity
tags: [paper, robot-learning, vision-transformer, dense-representation, vla, imitation-learning]
status: complete
updated: 2026-07-22
arxiv: "2607.18236"
related: [../methods/vla.md, ../concepts/vision-transformer.md, ../tasks/manipulation.md]
sources: [../../sources/papers/patch_policy_arxiv_2607_18236.md, ../../sources/sites/patch-policy-github-io.md]
summary: "Patch Policy 直接使用预训练 ViT 的密集 patch token，以块因果注意力实现轻量高频控制；相对全局池化提升 40%，参数约为 OpenVLA-OFT 的 0.7%。"
---

# Patch Policy：密集视觉表征的轻量高频控制

**Patch Policy** 是一种不依赖大型 VLM、直接把预训练 ViT 密集 patch 特征接入机器人策略 transformer 的轻量架构。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| ViT | Vision Transformer | 输出图像 patch token 的视觉骨干 |
| VLM | Vision-Language Model | 大型视觉语言模型 |
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| OFT | OpenVLA Optimized Fine-Tuning | 论文中的大型策略对照 |

## 核心机制

全局池化把每帧压成单 token，会抹去操作所需的局部空间结构；完整 VLA 又带来较大参数量和推理延迟。Patch Policy 以 block-causal attention 允许同一时刻的 patch 相互注意，同时阻止未来观测泄漏，保留标准策略的时间因果性。

## 实验与工程价值

- 四个仿真、三个真实环境套件上，相对先进全局池化表征取得 **40% 相对提升**。
- 相对微调 OpenVLA-OFT 高 **18%**，参数量约为其 **0.7%**。
- 适合将冻结视觉骨干与小策略头部署到需要高频反应的控制端，并作为大型 VLA 的低延迟替代基线。

## 与其他工作对比

| 维度 | Patch Policy | 全局池化表征策略 | 大型 VLA（OpenVLA-OFT 等） |
|------|--------------|------------------|----------------------------|
| 视觉接口 | 直接消费预训练 ViT 的密集 patch token | 每帧压成单 token，抹去局部空间结构 | 经大型 VLM 编码，附带语言对齐 |
| 参数规模 | 冻结骨干 + 小策略头，参数远小于大型 VLA | 轻量 | 大，推理延迟高 |
| 时间因果 | block-causal mask：帧内密集注意 + 阻止未来泄漏 | 视实现而定 | 视实现而定 |
| 控制频率 | 面向高频反应式控制 | 中 | 受参数量与延迟预算约束 |
| 相对表现 | 较先进全局池化与微调 OpenVLA-OFT 均有提升（见「实验与工程价值」） | 作为被超越基线 | 精度可观但难满足高频预算 |

## 局限与开源状态

- 尚需核对不同 ViT、相机数量与真实控制频率下的延迟–精度曲线。
- **源码运行时序图：不适用。** 截至 2026-07-22，[项目页](https://patch-policy.github.io/) 的 GitHub 入口仍为 “coming soon”。

## 关联页面

- [VLA](../methods/vla.md)
- [Vision Transformer](../concepts/vision-transformer.md)
- [Manipulation](../tasks/manipulation.md)

## 推荐继续阅读

- [Patch Policy 项目页](https://patch-policy.github.io/)
- [论文 PDF](https://arxiv.org/pdf/2607.18236)

## 参考来源

- [论文归档](../../sources/papers/patch_policy_arxiv_2607_18236.md)
- [项目页归档](../../sources/sites/patch-policy-github-io.md)

