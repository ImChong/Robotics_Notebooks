---
type: entity
tags: [paper, vla, diffusion, bimanual, thu]
status: complete
updated: 2026-09-20
arxiv: "2410.07864"
code: https://github.com/thu-ml/RoboticsDiffusionTransformer
related:
  - ./paper-pi0.md
  - ./paper-cogact.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/blogs/wechat_lumina_vla_survey_part1_2026-09-20.md
summary: "RDT-1B（arXiv:2410.07864）：双臂扩散 Transformer；thu-ml/RoboticsDiffusionTransformer 已开源。"
---

# RDT-1B（Robotics Diffusion Transformer）

**RDT-1B（Robotics Diffusion Transformer）**（[arXiv:2410.07864](https://arxiv.org/abs/2410.07864)，[代码](https://github.com/thu-ml/RoboticsDiffusionTransformer)）收录于 Lumina [Embodied-AI-Guide 微信专辑](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md)。本页为独立详情节点；实验数字以原文为准。

## 一句话定义

**十亿级扩散 Transformer 面向双臂操作。**

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
| **机构** | 清华大学（THU） |
| **arXiv** | [2410.07864](https://arxiv.org/abs/2410.07864) |
| **开源** | **已开源** |

## 实验与评测

- **本页为索引级节点**（Lumina Embodied-AI-Guide 微信专辑）：正文固化定位与开源边界，**未转存原文实验表**。
- **回原文须核对的证据**：本页结论已点明「双臂数据配方是上限」——回原文须核对数据规模 / 配方与成功率的 **scaling 关系**，而非单点成功率；官方仓含训练脚本，可自行复跑验证。
- **读法：** 先对齐本体、任务集与成功判定，再读任何数字；勿把专辑摘要当实验结论。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **对照工作** | 与 ACT / Diffusion Policy 等小模型对照 scaling：参数与数据同时放大后，收益是否仍在延续 |
| **横比口径** | scaling 结论绑定该双臂平台与数据配方；换本体后 scaling 曲线须重测，斜率不可外推。 |
| **开源状态** | **已开源** — 复现前以项目页 / 官方仓实际链接为准 |

## 结论

RDT-1B 把扩散+Transformer+双臂推到基础模型尺度。

- 双臂数据配方是上限
- 官方仓含训练脚本
- 与 ACT/DP 小模型对照 scaling

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
- [paper-cogact](./paper-cogact.md)
- [bimanual-manipulation](../tasks/bimanual-manipulation.md)

## 参考来源

- [wechat_lumina_vla_survey_part1_2026-09-20.md](../../sources/blogs/wechat_lumina_vla_survey_part1_2026-09-20.md)

## 推荐继续阅读

- [arXiv:2410.07864](https://arxiv.org/abs/2410.07864)
