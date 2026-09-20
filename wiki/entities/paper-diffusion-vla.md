---
type: entity
tags: [paper, vla, diffusion, manipulation]
status: complete
updated: 2026-09-20
arxiv: "2412.03293"
related:
  - ./paper-cogact.md
  - ./paper-diffusion-policy.md
  - ./paper-pi0.md
sources:
  - ../../sources/blogs/wechat_lumina_vla_survey_part1_2026-09-20.md
summary: "Diffusion-VLA（arXiv:2412.03293）：扩散解码的 VLA。"
---

# Diffusion-VLA

**Diffusion-VLA**（[arXiv:2412.03293](https://arxiv.org/abs/2412.03293)）收录于 Lumina [Embodied-AI-Guide 微信专辑](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md)。本页为独立详情节点；实验数字以原文为准。

## 一句话定义

**在 VLA 内用扩散解码动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| LLM | Large Language Model | 大语言模型 |
| IL | Imitation Learning | 模仿学习 |
| BC | Behavior Cloning | 行为克隆 |

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | — |
| **arXiv** | [2412.03293](https://arxiv.org/abs/2412.03293) |
| **开源** | 待核实 |

## 实验与评测

- **本页为索引级节点**（Lumina Embodied-AI-Guide 微信专辑）：正文固化定位与开源边界，**未转存原文实验表**。
- **回原文须核对的证据**：本页结论已点明两条读法——回原文须核对：(a) 相对纯 Diffusion Policy 的 **VLM 增益消融**；(b) 去噪步数与控制频率的取舍曲线。两者缺一，都无法判断收益来自语义条件还是动作头。
- **读法：** 先对齐本体、任务集与成功判定，再读任何数字；勿把专辑摘要当实验结论。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **对照工作** | 与纯 Diffusion Policy 对照，差异在是否引入 VLM 语义条件；与 [CogACT](./paper-cogact.md) 同属「大 VLM + 扩散动作头」一支 |
| **横比口径** | 去噪步数不同则实际控制频率不同；跨论文比成功率前必须先对齐推理频率与动作块长度。 |
| **开源状态** | **待核实** — 复现前以项目页 / 官方仓实际链接为准 |

## 结论

Diffusion-VLA 把 DP 多峰优势带入 VLA。

- 去噪步数影响频率
- 与纯 DP 对照 VLM 增益

## 源码运行时序图

**不适用**

## 关联页面

- [paper-cogact](./paper-cogact.md)
- [paper-diffusion-policy](./paper-diffusion-policy.md)
- [paper-pi0](./paper-pi0.md)

## 参考来源

- [wechat_lumina_vla_survey_part1_2026-09-20.md](../../sources/blogs/wechat_lumina_vla_survey_part1_2026-09-20.md)

## 推荐继续阅读

- [arXiv:2412.03293](https://arxiv.org/abs/2412.03293)
