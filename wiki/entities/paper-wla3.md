---
type: entity
tags: ['paper', 'latent-action', 'world-model', 'pretraining', 'manipulation']
status: complete
updated: 2026-09-16
arxiv: "2609.15870"
related:
  - ../concepts/world-action-models.md
  - ./paper-act-lam.md
  - ../methods/vla.md
  - ./paper-dido-wam.md
  - ../overview/vla-deploy-12-papers-technology-map.md
sources:
  - ../../sources/papers/wla3_arxiv_2609_15870.md
  - ../../sources/sites/wla3.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md
summary: "WLA³（arXiv:2609.15870）：从相邻世界状态变化学 latent action，同一表征复用于语义、动力学与运动学；人类视频可参与预训练。"
---

# WLA³（arXiv:2609.15870）

**WLA³**（*WLA^3: World Latent Action Modeling for Semantics, Dynamics, and Kinematics*，[arXiv:2609.15870](https://arxiv.org/abs/2609.15870)，[项目页](https://wla-3.github.io/)）来自 [具身智能小站 12 篇盘点](../../sources/blogs/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md)。

## 一句话定义

**从相邻世界状态变化学 latent action，同一表征复用于语义、动力学与运动学；人类视频可参与预训练。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WLA³ | World Latent Action Modeling | 本文三域潜动作框架 |
| LAM | Latent Action Model | 潜动作模型 |
| WM | World Model | 世界状态预测 |
| IL | Imitation Learning | 异构示范预训练 |

## 为什么重要

- 异构数据缺少统一低噪声动作监督，限制通用策略扩展。
- 开源结论：**待发布**（步骤 2.5，2026-09-16）。
- 与 [12 篇技术地图](../overview/vla-deploy-12-papers-technology-map.md) 中同类工作可横向对照。

## 流程总览

```mermaid
flowchart TD
    subgraph W["WLAM 世界潜在动作模型（§3.1，Fig. 2）"]
        A["世界状态 s_t = (多视角相机 I_t, 本体状态 r_t, 本体配置 e)，缺失模态省略"] --> B["跨模态 transition dropout：随机丢弃部分相机/状态流，t 与 t+h 用同一掩码"]
        B --> C["Transition 编码器（[CLS]）→ 1024D 转移特征 g_t,h"]
        C --> D["高斯后验 → 32D 连续潜在动作 z_t,h"]
        D --> E["解码器：由初始状态重建终点各模态"]
        D --> F["重叠窗口余弦一致性 L_overlap"]
        C --> G["30 步段内均值池化 → 段级特征 c_seg（SLA 目标）"]
    end
    D -->|冻结 WLAM 给每个控制间隔标注潜在动作| H
    subgraph S1["阶段 1：LAC-WM 预训练 (§3.2)"]
        H["LAC-WM：当前视觉特征 + 指令 + 潜在动作块<br/>条件流匹配预测未来视觉特征<br/>语言与潜在动作独立随机丢弃"]
    end
    subgraph S2["阶段 2：策略 mid-training"]
        I["VLM：观测 + 指令 + 语义查询 q_sem → 预测 SLA"] --> J["L_sem 对齐 c_seg"]
        G --> J
        H -->|冻结，提供动力学特征| K
        I -->|VLM 上下文| K["拼接得到策略上下文 H_pol"]
        L["零填充到 32D 的本体状态 + 噪声化联合动作块"] --> M
        K --> M["LARA action expert（Perceiver，流匹配）<br/>联合预测 32D 潜在动作 + 32D 机器人原生动作（带掩码）"]
        M --> N["L_policy = λ_lat·L_lat + λ_rob·L_rob + λ_sem·L_sem"]
        J --> N
    end
    N --> O["阶段 3：post-training（去掉语义查询、SLA 头与语义损失）"]
    O --> P["部署：积分得到两路输出，仅执行机器人动作分支的有效维度"]
```

图依据：论文 §3.1–3.2 与 Fig. 2 / Fig. 3、式 1–15

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.15870](https://arxiv.org/abs/2609.15870) |
| **开源** | **待发布** |
| **要点** | 世界状态差分 → latent action；跨语义/动力学/运动学复用；项目页 arXiv 链入库日标注 coming soon。 |
| **文内指标** | 以原文与项目页为准。 |


## 源码运行时序图

**不适用（待发布）** — 截至 2026-09-16 项目页未列可运行官方仓库。


## 实验与评测

- 以原文与项目页为准。
- **读法：** 清单摘要；逐项对照与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [12 篇技术地图](../overview/vla-deploy-12-papers-technology-map.md)。

## 结论

**WLA³ 把「世界变化」当动作共同语言，适合与世界模型/潜动作文献对照阅读。**

1. 开源边界：**待发布** — 以项目页实际链接为准（入库日 2026-09-16）。
2. 核心机制：世界状态差分 → latent action；跨语义/动力学/运动学复用；项目页 arXiv 链入库日标注 coming soon。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [paper-act-lam](./paper-act-lam.md)
- [vla](../methods/vla.md)
- [paper-dido-wam](./paper-dido-wam.md)

## 参考来源

- [wla3_arxiv_2609_15870.md](../../sources/papers/wla3_arxiv_2609_15870.md)
- [wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md](../../sources/blogs/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md)
- [arXiv:2609.15870](https://arxiv.org/abs/2609.15870)
- [arXiv PDF](https://arxiv.org/pdf/2609.15870)：已通读方法部分（实验与附录以略读为准）：摘要，§2 相关工作略读，§3 方法（WLAM、LAC-WM、SLA、LARA 与训练目标）

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.15870)
- [项目页](https://wla-3.github.io/)

