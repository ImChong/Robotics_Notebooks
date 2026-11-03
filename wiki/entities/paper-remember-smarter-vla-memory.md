---
type: entity
tags:
  - paper
  - vla
  - memory
  - long-horizon
  - manipulation
status: complete
updated: 2026-09-27
arxiv: "2608.15269"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md
sources:
  - ../../sources/papers/remember-smarter-vla-memory_arxiv_2608_15269.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md
summary: "Remember Smarter（RS）（arXiv:2608.15269）：近期多视角压缩为短记录 + 双曲空间经验库后台检索转提示，扩大可用历史而不拖慢 chunk 推理。"
---

# Remember Smarter（RS）（arXiv:2608.15269）

**Remember Smarter（RS）**（*Remember Smarter: Visual History Compressor and Hyperbolic Experience Space for Robotic Memory*，[arXiv:2608.15269](https://arxiv.org/abs/2608.15269)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第四篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md) **长程记忆** 段。

## 一句话定义

**近期多视角压缩为短记录 + 双曲空间经验库后台检索转提示，扩大可用历史而不拖慢 chunk 推理。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| PRM | Process Reward Model | 过程/进度奖励模型 |
| OOD | Out-of-Distribution | 分布外场景或轨迹 |

## 为什么重要

- 长程 VLA 需近期连续性与成功经验复用；直接堆历史会膨胀上下文。
- 策展机构：西安电子科技大学
- 开源结论：**待核实**（步骤 2.5，2026-09-27）。

## 流程总览

```mermaid
flowchart TD
    A["多视角历史图像流"] --> B["复用 VLA 视觉编码器 E_vis + 适配器 A_ϕ<br/>得到 patch 特征"]
    B --> C["逐帧双向 serpentine 空间 Mamba<br/>+ learned query 池化"]
    C --> D["因果时序 Mamba（两层，每个视角一个递归状态）"]
    D --> E["多视角记忆 M_t"]
    E --> F["残差 cross-attention 适配器<br/>H~_t = H_t + γΔH_t"]
    F --> G["原动作头输出动作"]
    H["成功示范的动作侧隐表示"] --> I["P-VAE：映射为 Poincaré 球叶节点"]
    I --> J["HypHC 式二叉层级 + Karcher 均值原型 + 蕴含锥校准"]
    K["当前隐表示编码为查询 q_t"] --> L["后台非阻塞 worker：沿层级束搜索<br/>最多返回 8 个叶节点"]
    J --> L
    L --> M["取最低公共祖先原型<br/>沿 q_t 到原型的测地线采样 4 点"]
    M --> N["解码为 4 个经验 token<br/>插入 VLM 输入首个图像嵌入之前"]
    N --> O["VLM 自注意力融合 → 隐表示"]
    O --> F
    O --> K
```

图依据：论文 §4.1 / §4.2 / §4.3、Fig. 1、Fig. 2。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.15269](https://arxiv.org/abs/2608.15269) |
| **开源** | **待核实** |
| **文内评测** | LIBERO、LIBERO-Plus；单臂实机 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** LIBERO、LIBERO-Plus；单臂实机
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第四篇技术地图](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md) | 同批 14 篇横向索引；本文属 **长程记忆** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**Remember Smarter（RS） 适合作为本期「长程记忆」路线的快速索引页。**

1. 核心贡献：近期多视角压缩为短记录 + 双曲空间经验库后台检索转提示，扩大可用历史而不拖慢 chunk 推理。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第四篇技术地图](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第四篇）](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part4.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md)
- [arXiv:2608.15269](https://arxiv.org/abs/2608.15269)
- [arXiv PDF](https://arxiv.org/pdf/2608.15269)：已通读方法部分（实验与附录以略读为准）：§4.1–4.3（含 Fig. 1/2 与异步检索、测地线经验 token 段）

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.15269)
