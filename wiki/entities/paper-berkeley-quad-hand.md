---
type: entity
project_id: berkeley-quad-hand
tags:
  - paper
  - dexterous-hand
  - hardware
  - underactuated
  - quasi-direct-drive
status: complete
updated: 2026-10-18
arxiv: "2609.27240"
project: https://benudavis.github.io/berkeley-quadhand/
related:
  - ../concepts/dexterous-kinematics.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-10-18-26-papers-technology-map.md
sources:
  - ../../sources/papers/berkeley_quad_hand_arxiv_2609_27240.md
  - ../../sources/sites/berkeley-quad-hand.md
  - ../../sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md
summary: "Berkeley QUAD Hand（arXiv:2609.27240）：用非对称驱动和柔顺耦合让不同手指分工，兼顾灵巧与持续握力，并控制体积与发热。"
---

# Berkeley QUAD Hand（arXiv:2609.27240）

**Berkeley QUAD Hand**（*A Quasi-Direct-Drive Underactuated Asymmetric Hand for Dexterous and Efficient Grasping and Manipulation*，[arXiv:2609.27240](https://arxiv.org/abs/2609.27240)，[项目页](https://benudavis.github.io/berkeley-quadhand/)）来自 [具身智能小站 2026-10-18 盘点](../../sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)，属「灵巧与触觉」组。

## 一句话定义

**用非对称驱动和柔顺耦合让不同手指分工，兼顾灵巧与持续握力，并控制体积与发热。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QDD | Quasi-Direct-Drive | 准直驱 |

## 为什么重要

- 灵巧与持续握力难以由同一指实现，指分工是硬件层面的折中。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-10-18）。项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
- 同期同组工作见 [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../overview/embodied-station-2026-10-18-26-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
    A["尺侧 QDD 电机 CubeMars AKE60-8"] --> B["齿条-小齿轮"]
    B --> C["预紧柔顺尺侧传输（六杆机构）：link 1 接齿条"]
    C --> D{"中指指尖力是否超过约 3.2 N 阈值？"}
    D -->|否：弹簧不变形| E["link 1/2/3 同向竖直运动"]
    E --> F["中指 MCP 直接被驱动；无名指 MCP/IP 随之同步（镜像中指）"]
    D -->|是：弹簧变形| G["link 2 相对 link 1 运动，中指顺应"]
    G --> H["经 link 6 带动输出 link 3 向桡侧内扫"]
    H --> I["无名指绕 CMC 内收，向抓握中心合拢，施加更好的闭合力"]
    J["拇指、食指：指根 QDD GIM3505-08"] --> K["各自独立驱动，负责灵巧操作"]
```

图依据：论文 §III（Fig. 2，A. Hybrid Actuation Strategy、B. Finger Transmissions）及摘要。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.27240](https://arxiv.org/abs/2609.27240) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 准直驱 + 欠驱动 + 非对称手指分工 + 柔顺耦合。 |
| **文内指标** | 公众号称硬件实现了 Feix 分类中 29 种抓型，分母为 33 种。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号称硬件实现了 Feix 分类中 29 种抓型，分母为 33 种。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../overview/embodied-station-2026-10-18-26-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**29/33 的抓型覆盖说明该硬件范围广但并非全覆盖。**

1. 开源边界：**项目页未列 GitHub** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
2. 核心机制：准直驱 + 欠驱动 + 非对称手指分工 + 柔顺耦合。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [dexterous-kinematics](../concepts/dexterous-kinematics.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-10-18-26-papers-technology-map](../overview/embodied-station-2026-10-18-26-papers-technology-map.md)

## 参考来源

- [berkeley_quad_hand_arxiv_2609_27240.md](../../sources/papers/berkeley_quad_hand_arxiv_2609_27240.md)
- [wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md](../../sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)
- [arXiv:2609.27240](https://arxiv.org/abs/2609.27240)
- [arXiv PDF](https://arxiv.org/pdf/2609.27240)：已通读方法部分（实验与附录以略读为准）：§I 引言、§III 设计、§IV 运动学分析（略读）、§V 实验评估（略读）

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.27240)
- [项目页](https://benudavis.github.io/berkeley-quadhand/)
