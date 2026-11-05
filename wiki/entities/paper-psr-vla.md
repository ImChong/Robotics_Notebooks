---
type: entity
tags:
  - paper
  - vla
  - contact-rich
  - force-torque
  - manipulation
status: complete
updated: 2026-09-21
arxiv: "2609.21753"
related:
  - ../overview/contact-rich-sim-10-papers-technology-map.md
  - ../concepts/sim2real.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/psr_vla_arxiv_2609_21753.md
  - ../../sources/sites/psr-vla.md
  - ../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md
summary: "PSR-VLA（arXiv:2609.21753）：从视觉/力矩/本体感知历史预测未来接触动力学，六层 sensorimotor hierarchy 注入 VLA；6 真机接触任务总体 91.7%，相对力觉基线 +19.2–30.0 pp。"
---

# PSR-VLA（arXiv:2609.21753）

**PSR-VLA**（*PSR: Predictive Sensorimotor Representation Learning for Contact-Rich Manipulation*，[arXiv:2609.21753](https://arxiv.org/abs/2609.21753)）来自 [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md)（策展档位：**扫读**）。

## 一句话定义

**从视觉/力矩/本体感知历史预测未来接触动力学，六层 sensorimotor hierarchy 注入 VLA；6 真机接触任务总体 91.7%，相对力觉基线 +19.2–30.0 pp。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习 |
| SR | Success Rate | 任务成功率 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |
| WAM | World-Action Model | 联合未来观测与动作的策略 |

## 为什么重要

- 公众号将本文归入「接触丰富操作为何总在仿真里失真」专题；扫读档位。
- **（匿名投稿，待正式披露）**；开源结论：**待发布**（步骤 2.5，2026-09-21）。
- 接触丰富任务需提前感知接触如何演化；PSR 用预测式 sensorimotor 表征桥接物理历史与动作生成。

## 流程总览

```mermaid
flowchart TD
    A["输入：视觉 token + 状态 / 关节力矩（effort）历史 + 腕部 F/T 历史"] --> B["Stage 1：PSR 编码器（六层预测 Transformer）"]
    B --> C["浅层：视觉 / F/T / 本体各流独立处理"]
    C --> D["中层：视觉与 F/T 流用 MoE（4 专家，Top-2 路由）+ 瓶颈 token 跨模态交换"]
    D --> E["深层：精炼融合 token"]
    E --> F["保留各层输出 Z = z1…z6（sensorimotor hierarchy）"]
    F --> G["共享预测解码器（仅训练期使用）"]
    G --> H["预测未来 F/T 序列 / 未来关节 effort 序列 / 终端关节位移"]
    H --> I["L_pred = αL_ft + βL_eff + γL_disp + δL_route（VLA 冻结）"]
    F --> J["Stage 2：门控 cross-attention 注入 Action Expert 的多个深度层"]
    J --> K["联合微调 VLM + Action Expert + PSR 编码器，仅用 π0.5 flow-matching 动作损失"]
    K --> L["动作 chunk（推理时预测解码器不参与）"]
```

图依据：论文 §III-B / §III-C / §III-D、Fig. 2 / Fig. 3。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.21753](https://arxiv.org/abs/2609.21753) |
| **开源** | **待发布** |
| **策展摘要** | 从视觉/力矩/本体感知历史预测未来接触动力学，六层 sensorimotor hierarchy 注入 VLA；6 真机接触任务总体 91.7%，相对力觉基线 +19.2–30.0 pp。 |

## 源码运行时序图

**不适用**（截至 2026-09-21 项目页/arXiv 未发布可运行官方代码，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- 定量指标与 baseline 协议以 arXiv PDF 与项目页为准；本页为清单级摘要。
- 读法：先确认任务设定（仿真/真机、传感器、成功定义）再对比 headline 数字。

## 与其他工作对比

> 本页为清单级摘要，下表只做**定位对照**；数字未与下列各页核对同一评测协议，不可横比。

| 对照 | 差异读法 |
|------|----------|
| [RAPID](./paper-rapid-vlm-rl.md) | 同专辑 **深读** 档：并行仿真 + VLM 奖励管线，关注训练吞吐与 API 成本 |
| [CRISP](./paper-crisp.md) | 同专辑 **跟进** 档：接触仿真几何与求解器，关注 peg-in-hole/装配物理准确性 |
| [10 篇技术地图](../overview/contact-rich-sim-10-papers-technology-map.md) | 同批次横向对照入口：本文列 **扫读** 档位 |

## 结论

**PSR-VLA 代表「扫读」档位的 vla 方向样本——部署前以开源状态与评测协议为准绳。**

1. 开源状态：**待发布**；勿凭 PDF 臆断可复现性。
2. 与同专辑 [RAPID](./paper-rapid-vlm-rl.md) / [GALA](./paper-gala.md) / [CRISP](./paper-crisp.md) 形成「并行奖励 → 跨形态表征 → 接触仿真」阅读链。
3. 若做工程选型，先对齐传感器栈、仿真器与任务是否匹配文内设定。
4. 关注项目页/arXiv 版本更新与代码发布。

## 关联页面

- [contact-rich-sim-10-papers-technology-map](../overview/contact-rich-sim-10-papers-technology-map.md)
- [sim2real](../concepts/sim2real.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [psr_vla_arxiv_2609_21753.md](../../sources/papers/psr_vla_arxiv_2609_21753.md)
- [wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md](../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md)
- [arXiv:2609.21753](https://arxiv.org/abs/2609.21753)
- [arXiv PDF](https://arxiv.org/pdf/2609.21753)：已通读方法部分（实验与附录以略读为准）：§III-A~D（两阶段预测预训练、多模态预测 Transformer、多深度动作增强）、§IV 实验设定

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.21753)
- [10 篇技术地图](../overview/contact-rich-sim-10-papers-technology-map.md)
