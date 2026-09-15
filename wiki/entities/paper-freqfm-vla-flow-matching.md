---
type: entity
tags: [paper, vla, flow-matching, manipulation]
status: complete
updated: 2026-09-15
arxiv: "2609.10405"
related:
  - ../tasks/manipulation.md
  - ./paper-tfgca-chunked-vla.md
  - ./paper-wm-craftnet.md
sources:
  - ../../sources/papers/freqfm_vla_arxiv_2609_10405.md
summary: "FreqFM（arXiv:2609.10405）：FreqFM; DCT frequency-conditioned source; LIBERO-Plus +9.3; 6 real tasks；截至入库日未见官方代码。"
---

# FreqFM（arXiv:2609.10405）

**FreqFM**（*Frequency-Conditioned Flow Matching for Vision-Language-Action Models*，[arXiv:2609.10405](https://arxiv.org/abs/2609.10405)）由 **（论文未在 digest 标注机构）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_manipulation_2026-09-14.md)）。

## 一句话定义

面向视觉-语言-动作模型的频率条件 Flow Matching — FreqFM。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FM | Flow Matching | 流匹配生成 |
| DCT | Discrete Cosine Transform | 离散余弦变换 |
| VLA | Vision-Language-Action | 视觉-语言-动作 |

## 为什么重要

VLA 动作序列高频噪声影响执行；频域条件化 flow matching 可塑形频谱。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | （论文未在 digest 标注机构） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

用 DCT 将动作轨迹映射到频域；flow matching 在频率条件源上学习向量场；解码回时域执行。

### 流程总览

```mermaid
flowchart LR
  obs[视觉+语言] --> vla[VLA 骨干]
  vla --> dct[DCT 频域]
  dct --> fm[FreqFM]
  fm --> act[动作轨迹]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-14** arXiv 与常见项目页 **未见** 官方可运行代码仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 截断高频分量可抑抖；真机需对齐控制频率。 |

## 实验与评测

LIBERO-Plus +9.3；6 real tasks。

## 结论

FreqFM 用频率条件 flow matching 提升 VLA 动作质量与真机成功率。

1. DCT 提供可解释频谱控制。
2. Flow matching 替代扩散采样更高效。
3. LIBERO-Plus +9.3。
4. 6 项真机任务验证。
5. 与 chunking 方法互补。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 时域 diffusion VLA | 高频噪声多 |
| 无频域条件 FM | 平滑性不足 |

## 局限与风险

机构未知；极端长 horizon 未测。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [./paper-tfgca-chunked-vla.md](./paper-tfgca-chunked-vla.md)
- [./paper-wm-craftnet.md](./paper-wm-craftnet.md)

## 参考来源

- [freqfm_vla_arxiv_2609_10405.md](../../sources/papers/freqfm_vla_arxiv_2609_10405.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_manipulation_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.10405](https://arxiv.org/abs/2609.10405)
