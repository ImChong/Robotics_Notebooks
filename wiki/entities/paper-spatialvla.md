---
type: entity
tags: [paper, vla, 3d, manipulation, shanghai-ai-lab]
status: complete
updated: 2026-09-20
arxiv: "2501.15830"
code: https://github.com/SpatialVLA/SpatialVLA
related:
  - ./paper-sa-2403-09631-3d-vla-a-3d-vision-language-action-generative-wo.md
  - ./paper-openvla.md
sources:
  - ../../sources/blogs/wechat_lumina_vla_survey_part1_2026-09-20.md
summary: "SpatialVLA（arXiv:2501.15830）：3D 对齐 VLA；SpatialVLA/SpatialVLA 已开源。"
---

# SpatialVLA

**SpatialVLA**（[arXiv:2501.15830](https://arxiv.org/abs/2501.15830)，[代码](https://github.com/SpatialVLA/SpatialVLA)）收录于 Lumina [Embodied-AI-Guide 微信专辑](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md)。本页为独立详情节点；实验数字以原文为准。

## 一句话定义

**把 3D 空间结构写进 VLA 对齐。**

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
| **机构** | 上海人工智能实验室（Shanghai AI Lab） |
| **arXiv** | [2501.15830](https://arxiv.org/abs/2501.15830) |
| **开源** | **已开源** |

## 实验与评测

- **本页为索引级节点**（Lumina Embodied-AI-Guide 微信专辑）：正文固化定位与开源边界，**未转存原文实验表**。
- **回原文须核对的证据**：本页结论已点明「Adaptive Action Grid 是关键」——回原文须核对该动作离散化方式的消融（换成固定网格或连续回归后的掉点）；官方仓已开源，可自行复跑。
- **读法：** 先对齐本体、任务集与成功判定，再读任何数字；勿把专辑摘要当实验结论。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **对照工作** | 与非 3D 对齐的 VLA 对照：本文用空间对齐的自适应动作网格换取跨本体、跨相机外参的迁移性 |
| **横比口径** | 动作网格粒度决定精度上限；跨论文比成功率前须先对齐动作空间定义，否则比的是离散化而非策略。 |
| **开源状态** | **已开源** — 复现前以项目页 / 官方仓实际链接为准 |

## 结论

SpatialVLA 代表 3D-aware VLA。

- Adaptive Action Grid 是关键
- 官方仓库可复现

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

- [paper-sa-2403-09631-3d-vla-a-3d-vision-language-action-generative-wo](./paper-sa-2403-09631-3d-vla-a-3d-vision-language-action-generative-wo.md)
- [paper-openvla](./paper-openvla.md)

## 参考来源

- [wechat_lumina_vla_survey_part1_2026-09-20.md](../../sources/blogs/wechat_lumina_vla_survey_part1_2026-09-20.md)

## 推荐继续阅读

- [arXiv:2501.15830](https://arxiv.org/abs/2501.15830)
