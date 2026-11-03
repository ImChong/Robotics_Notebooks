---
type: entity
tags: ['paper', 'dexterous-manipulation', 'tactile', 'proximity', 'manipulation']
status: complete
updated: 2026-09-20
arxiv: "2609.16586"
related:
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
  - ../concepts/tactile-sensing.md
  - ./paper-stereopatch.md
  - ../overview/vla-deploy-12-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md
  - ../../sources/papers/proxidex_arxiv_2609_16586.md
  - ../../sources/sites/proxidex.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md
summary: "ProxiDex（arXiv:2609.16586）：把手—物体接近关系转为硬件无关交互表征，学习动作条件下的接近动态，补视觉不可靠时的策略信号。"
---

# ProxiDex（arXiv:2609.16586）

**ProxiDex**（*ProxiDex: Learning Dynamics-Guided Proximity Policy for Dexterous Manipulation*，[arXiv:2609.16586](https://arxiv.org/abs/2609.16586)，[项目页](https://proxidex.github.io/)）来自 [具身智能小站 12 篇盘点](../../sources/blogs/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md)。

## 一句话定义

**把手—物体接近关系转为硬件无关交互表征，学习动作条件下的接近动态，补视觉不可靠时的策略信号。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ProxiDex | Proximity-guided Dexterous policy | 本文接近动态策略 |
| Dex | Dexterous | 灵巧手操作 |
| IL | Imitation Learning | 模仿学习管线 |
| Sim2Real | Simulation to Real | 仿真到真机 |

## 为什么重要

- 灵巧手操作中手部遮挡与触觉硬件差异使接触状态难稳定观测。
- 开源结论：**待发布**（步骤 2.5，2026-09-16）。
- 与 [12 篇技术地图](../overview/vla-deploy-12-papers-technology-map.md) 中同类工作可横向对照。

## 流程总览

```mermaid
flowchart TD
    subgraph S1["阶段 1：交互感知数据采集 (§4.1)"]
        A["VR 臂-手协同遥操作<br/>记录机器人状态、物体位姿、专家动作"] --> D
        B["Grounded-SAM + SAM3D 由单张初始图重建物体 → 尺度标定得到 M_o<br/>FoundationPose++ 跟踪 6D 位姿"] --> C["物体与手部连杆表面采样点，转到机器人基座系<br/>交互点云 o^p_t = P^o ∪ P^h"]
        C --> D["手-物体距离 → 分区域接近度向量 o^prox_t<br/>掌心响应阈值 → 阶段标签 o^traj_t（0 接近 / 1 接触交互）"]
    end
    D --> E
    subgraph S2["阶段 2：潜在交互动力学 (§4.2，自监督)"]
        E["VAE 预训练：o^prox → 接近度潜变量 z^prox"]
        F["观测编码器 E_θ：交互点云 + 机器人状态 s_t → z^obs"]
        F --> G["前向动力学 FDM (DiT)：z^obs + 动作序列 → 未来 z^obs<br/>L_for"]
        G --> H["逆向差分模块 IDM：潜变量差 → 接近度潜变量，对齐 VAE 的 z^prox<br/>L_inv"]
        F --> I["轨迹头：交互阶段概率 p^int<br/>L_traj"]
    end
    S2 -->|冻结编码器| J
    subgraph S3["阶段 3：轨迹自适应接近度策略 (§4.3)"]
        J["Transformer 去噪（扩散）策略<br/>z^obs 对所有动作步可见；接近度 token 经因果记忆掩码"] --> K["门控 g(p^int)：运动阶段抑制、交互阶段增强接近度分支"]
        K --> L["训练用 VAE 的 z^prox，推理时换成 IDM 估计"]
    end
    L --> M
    subgraph S4["在线推理 (§4.4，Fig. 3)"]
        M["实时 z^obs，IDM 估计 z^prox，轨迹头给 p^int"] --> N["策略生成动作序列，滚动时域执行"]
        N --> O{"动力学偏差 e_dyn = ‖z^obs_t+1 − FDM(z^obs_t, a_t)‖ ≤ τ_dyn？"}
        O -->|是| P["视觉反馈可靠：用感知到的潜变量更新状态"]
        O -->|否| Q["dynamics-dream 模式：用 FDM 自回归预测的潜变量继续生成动作"]
        Q -->|偏差回落到阈值以下| P
    end
```

图依据：论文 §4.1–4.4 与 Fig. 2 / Fig. 3、式 2–5

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.16586](https://arxiv.org/abs/2609.16586) |
| **开源** | **待发布** |
| **要点** | 接近关系表征 + 动作条件接近动态；项目页入库日为占位模板，无有效 GitHub。 |
| **文内指标** | 以项目页与原文为准。 |


## 源码运行时序图

**不适用（待发布）** — 截至 2026-09-16 项目页未列可运行官方仓库。


## 实验与评测

- 以项目页与原文为准。
- **读法：** 清单摘要；逐项对照与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [12 篇技术地图](../overview/vla-deploy-12-papers-technology-map.md)。

## 结论

**ProxiDex 把「接近」当可迁移中间表征，适合跟踪视觉退化下的灵巧操作，但代码待发布。**

1. 开源边界：**待发布** — 以项目页实际链接为准（入库日 2026-09-16）。
2. 核心机制：接近关系表征 + 动作条件接近动态；项目页入库日为占位模板，无有效 GitHub。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [imitation-learning](../methods/imitation-learning.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [paper-stereopatch](./paper-stereopatch.md)

## 参考来源

- [proxidex_arxiv_2609_16586.md](../../sources/papers/proxidex_arxiv_2609_16586.md)
- [wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md](../../sources/blogs/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md)
- [arXiv:2609.16586](https://arxiv.org/abs/2609.16586)
- [arXiv PDF](https://arxiv.org/pdf/2609.16586)：已通读方法部分（实验与附录以略读为准）：§1–3，§4 方法（数据采集、潜在交互动力学、轨迹自适应策略、动力学一致性推理），§5 实验设定与部分结果

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.16586)
- [项目页](https://proxidex.github.io/)

