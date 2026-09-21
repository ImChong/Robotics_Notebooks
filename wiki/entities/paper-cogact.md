---
type: entity
tags: [paper, vla, diffusion, manipulation, microsoft, thu]
status: complete
updated: 2026-09-20
arxiv: "2411.19650"
code: https://github.com/microsoft/CogACT
related:
  - ./paper-pi0.md
  - ./paper-openvla.md
  - ./paper-diffusion-vla.md
  - ../methods/vla.md
sources:
  - ../../sources/blogs/wechat_lumina_vla_survey_part1_2026-09-20.md
summary: "CogACT（arXiv:2411.19650）：认知式 VLA，扩散动作头；microsoft/CogACT 已开源。"
---

# CogACT

**CogACT**（[arXiv:2411.19650](https://arxiv.org/abs/2411.19650)，[代码](https://github.com/microsoft/CogACT)）收录于 Lumina [Embodied-AI-Guide 微信专辑](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md)。本页为独立详情节点；实验数字以原文为准。

## 一句话定义

**VLM 语义主干 + 专用扩散动作专家。**

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
| **机构** | 清华大学 / 微软亚洲研究院（THU / MSRA） |
| **arXiv** | [2411.19650](https://arxiv.org/abs/2411.19650) |
| **开源** | **已开源** |

## 实验与评测

- **本页为索引级节点**（Lumina Embodied-AI-Guide 微信专辑）：正文固化定位与开源边界，**未转存原文实验表**。
- **回原文须核对的证据**：本页结论已点明「扩散动作头影响接触任务」——回原文须核对扩散头 vs 自回归头的消融，以及接触密集任务上的成功率差；官方仓已开源，指标可自行复跑而不必依赖二手摘要。
- **读法：** 先对齐本体、任务集与成功判定，再读任何数字；勿把专辑摘要当实验结论。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **对照工作** | 与 OpenVLA 的自回归动作头对照：本文用扩散头换取动作分布的多峰表达，代价是去噪步数带来的推理延迟；与 [Diffusion-VLA](./paper-diffusion-vla.md) 同属扩散动作头一支 |
| **横比口径** | VLA 成功率与底座 VLM 规模、微调数据配方强绑定；换底座或换数据配方即须重测，跨论文的 headline 成功率不可直接比。 |
| **开源状态** | **已开源** — 复现前以项目页 / 官方仓实际链接为准 |

## 结论

CogACT 是 2024 末大 VLM + 扩散动作头代表。

- 扩散动作头影响接触任务
- 官方 GitHub 可复现
- 与 OpenVLA 自回归对照

## 源码运行时序图

官方仓库提供训练/推理入口；节点对齐 README。

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Repo as 官方仓库
    participant Policy as 策略
    participant Env as 仿真/真机
    Dev->>Repo: clone + 依赖
    Dev->>Policy: 加载权重
    loop 控制环
        Env->>Policy: 观测
        Policy->>Env: 动作
    end
```

## 关联页面

- [paper-pi0](./paper-pi0.md)
- [paper-openvla](./paper-openvla.md)
- [paper-diffusion-vla](./paper-diffusion-vla.md)
- [vla](../methods/vla.md)

## 参考来源

- [wechat_lumina_vla_survey_part1_2026-09-20.md](../../sources/blogs/wechat_lumina_vla_survey_part1_2026-09-20.md)

## 推荐继续阅读

- [arXiv:2411.19650](https://arxiv.org/abs/2411.19650)
