---
type: entity
project_id: acts
tags:
  - paper
  - world-model
  - adaptation
status: complete
updated: 2026-11-01
arxiv: "2610.24285"
project: https://acts-anon.github.io/
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../concepts/tactile-sensing.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/acts_arxiv_2610_24285.md
  - ../../sources/sites/acts.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "ACTS（arXiv:2610.24285）：ACTS把传感器运动与法向力变化作为共享动作，联合预测视觉和触觉；人类数据训练后出现机器人触觉迁移，但视觉迁移并不稳定，尚非闭环控制结论。"
---

# ACTS（arXiv:2610.24285）

**ACTS**（*ACTS: Learning Visuo-Tactile World Models from Human Physical Interaction*，[arXiv:2610.24285](https://arxiv.org/abs/2610.24285)，[项目页](https://acts-anon.github.io/)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「世界模型与在线适配」组。

## 一句话定义

**ACTS把传感器运动与法向力变化作为共享动作，联合预测视觉和触觉；人类数据训练后出现机器人触觉迁移，但视觉迁移并不稳定，尚非闭环控制结论。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ACTS | ACTS: Learning Visuo-Tactile World Models from Human Physical Interaction | 本文方法 / 基准的简称 |

## 为什么重要

- 从人手交互学习接触如何变化。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-11-01）。项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
    A["人类双手持 GelSight 传感器<br/>采集 React 数据集 4.9 h"] --> B["同步 30 Hz：俯视场景 / 腕部相机 / 触觉图像 / 6-DoF 位姿"]
    B --> C["触觉图像估计法向力 f"]
    B --> D["冻结 VAE 编码（Wan2.2 因果 VAE；触觉用适配变体）"]
    C --> E["力感知传感器动作<br/>位姿增量 + kf·Δf·ns（Rot6D）"]
    B --> F["视角对齐动作<br/>传感器三点投影到场景视图"]
    D --> G["空间 patchify + 线性投影为 token<br/>历史干净 token + 未来噪声 token"]
    E --> H["打包为单一序列<br/>MRoPE 时间对齐 + 动作窗口全可见的注意力掩码"]
    F --> H
    G --> H
    H --> I["Cosmos3 初始化生成器<br/>rectified flow matching 预测速度"]
    I --> J["推理：逐块自回归联合去噪未来潜变量"]
    J --> K["冻结解码器重建<br/>场景 / 腕部 / 触觉预测"]
    K --> L["零样本评测：机器人 YAM + GELLO 遥操作回放"]
```

图依据：论文 §3 数据集、§4.1–4.4 / Fig.2、§5.2 机器人评测。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.24285](https://arxiv.org/abs/2610.24285) |
| **开源** | **项目页未列 GitHub** |
| **要点** | ACTS把传感器运动与法向力变化作为共享动作，联合预测视觉和触觉；人类数据训练后出现机器人触觉迁移，但视觉迁移并不稳定，尚非闭环控制结论。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**从人手交互学习接触如何变化：ACTS把传感器运动与法向力变化作为共享动作，联合预测视觉和触觉；人类数据训练后出现机器人触觉迁移，但视觉迁移并不稳定，尚非闭环控制结论。**

1. 开源边界：**项目页未列 GitHub** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
2. 核心机制：ACTS把传感器运动与法向力变化作为共享动作，联合预测视觉和触觉；人类数据训练后出现机器人触觉迁移，但视觉迁移并不稳定，尚非闭环控制结论。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [acts_arxiv_2610_24285.md](../../sources/papers/acts_arxiv_2610_24285.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.24285](https://arxiv.org/abs/2610.24285)
- [arXiv PDF](https://arxiv.org/pdf/2610.24285)：已通读方法部分（实验与附录以略读为准）：数据与评测设定（§3、§5.2）、方法（§4.1–4.4、Fig.2）

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.24285)
- [项目页](https://acts-anon.github.io/)
