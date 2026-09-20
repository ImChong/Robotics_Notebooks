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
