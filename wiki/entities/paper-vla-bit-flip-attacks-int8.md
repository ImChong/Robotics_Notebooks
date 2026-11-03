---
type: entity
tags:
  - paper
  - vla
  - security
  - quantization
  - deployment
status: complete
updated: 2026-09-27
arxiv: "2608.15475"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md
sources:
  - ../../sources/papers/vla-bit-flip-attacks-int8_arxiv_2608_15475.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md
summary: "VLA Bit-Flip Attacks（arXiv:2608.15475）：量化 VLA 的 INT8 权重易受 Rowhammer 类比特翻转；关键位集中在动作解码层，少量翻转即可让闭环任务失效。"
---

# VLA Bit-Flip Attacks（arXiv:2608.15475）

**VLA Bit-Flip Attacks**（*Bit-Flip Attacks on Vision-Language-Action Models: Action-Decoding Architecture Shapes the Vulnerability*，[arXiv:2608.15475](https://arxiv.org/abs/2608.15475)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第四篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md) **安全防御** 段。

## 一句话定义

**量化 VLA 的 INT8 权重易受 Rowhammer 类比特翻转；关键位集中在动作解码层，少量翻转即可让闭环任务失效。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| PRM | Process Reward Model | 过程/进度奖励模型 |
| OOD | Out-of-Distribution | 分布外场景或轨迹 |

## 为什么重要

- 部署侧需把权重完整性纳入威胁模型；动作头架构决定脆弱性分布。
- 策展机构：香港科技大学；阿德莱德大学；佐治亚理工；北卡罗莱纳大学教堂山；中国石油大学（华东）
- 开源结论：**待核实**（步骤 2.5，2026-09-27）。

## 流程总览

```mermaid
flowchart TD
    A["INT8 VLA<br/>逐输出通道对称量化权重"] --> B["可微执行动作 ã_θ<br/>回归头直接用 / 离散 token 用期望解码 / flow 反传穿过 solver 步"]
    B --> C["Step 1：固定方向 manifold-escape 损失 L_dir<br/>u=1，M=2-3 个固定噪声采样"]
    D["校验集 D_cal（≤6 帧，与评测不相交）"] --> C
    C --> E["一次反向传播得权重梯度 g"]
    E --> F["Step 2：量化感知比特排序<br/>对每个权重 8 个比特算增益 G = -g·Δ"]
    F --> G["选 top-K 正增益比特翻转"]
    G --> H["被翻转的 INT8 权重模型"]
    H --> I["开环动作偏差 / 闭环成功率（LIBERO、SimplerEnv）"]
    G --> J["防御：局部完整性保护<br/>把受保护权重移出候选集后重建攻击"]
    J --> F
```

图依据：论文 §5 Method（Step 1 / Step 2，Algorithm 1）、Fig. 2、§6.3 防御设定。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.15475](https://arxiv.org/abs/2608.15475) |
| **开源** | **待核实** |
| **文内评测** | LIBERO、SimplerEnv；双臂实机 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** LIBERO、SimplerEnv；双臂实机
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第四篇技术地图](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md) | 同批 14 篇横向索引；本文属 **安全防御** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**VLA Bit-Flip Attacks 适合作为本期「安全防御」路线的快速索引页。**

1. 核心贡献：量化 VLA 的 INT8 权重易受 Rowhammer 类比特翻转；关键位集中在动作解码层，少量翻转即可让闭环任务失效。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第四篇技术地图](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第四篇）](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part4.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md)
- [arXiv:2608.15475](https://arxiv.org/abs/2608.15475)
- [arXiv PDF](https://arxiv.org/pdf/2608.15475)：已通读方法部分（实验与附录以略读为准）：§3 INT8 量化与比特翻转定义、§4 Threat Model、§5 Method、§6.1/6.3/6.4 实验设定与防御

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.15475)
