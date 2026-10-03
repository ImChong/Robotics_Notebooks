---
type: entity
tags:
  - paper
  - humanoid
  - loco-manipulation
  - rl
  - sim2real
status: complete
updated: 2026-10-03
arxiv: "2609.19340"

related:
  - ../tasks/loco-manipulation.md
  - ../tasks/locomotion.md
  - ./paper-wholebodywam.md
  - ../concepts/sim2real.md
  - ../overview/contact-wm-10-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md
  - ../../sources/papers/viloman_arxiv_2609_19340.md
  - ../../sources/sites/viloman.md
  - ../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md
summary: "ViLoMan（arXiv:2609.19340）：人类–物体交互重定向 + 特权 teacher + 在线 DAgger 蒸馏；Unitree G1 机载深度 + 本体 whole-body 关门。"
---

# ViLoMan（arXiv:2609.19340）

**ViLoMan**（*Learning Visual-Proprioceptive Whole-Body Loco-Manipulation Skills for Humanoid Robots*，[arXiv:2609.19340](https://arxiv.org/abs/2609.19340)，[项目页](https://viloman-anonymous.pages.dev/)）来自 [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)（策展档位：**扫读**）。arXiv 列出的作者为 Zejie Tian、Ruibing Hou、Bingpeng Ma、Börje F. Karlsson、Shiguang Shan；单位为中国科学院计算技术研究所人工智能安全重点实验室、中国科学院大学、北京智源人工智能研究院。

## 一句话定义

**人类–物体交互重定向 + 特权 teacher + 在线 DAgger 蒸馏；Unitree G1 机载深度 + 本体 whole-body 关门。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 将行走、平衡与接触操作作为全身协调任务处理 |
| GMT | General Motion Tracking | ViLoMan 冻结并扩展为交互教师的通用动作跟踪先验 |
| PPO | Proximal Policy Optimization | 用于训练特权交互残差策略的强化学习算法 |
| DAgger | Dataset Aggregation | 学生在线采样并请求教师标签的模仿学习过程 |
| DoF | Degrees of Freedom | 项目页列出的 G1 策略控制 29 个关节自由度 |

## 为什么重要

- 公众号将本文归入「接触时视觉之外还需预测什么」专题；扫读档位。
- arXiv 已列出作者和机构；截至 2026-10-03，项目补充页仍显示 “Anonymous Authors”，并未列出代码仓库或数据下载入口。
- 与 tactile/WAM、主动视角、多智能体场景理解、Sim2Real、人形导航等主线交叉。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.19340](https://arxiv.org/abs/2609.19340) |
| **开源** | 项目页截至 2026-10-03 未列官方代码仓库或数据下载链接；当前无法据此复现训练和部署 |
| **数据到策略** | 保留人–物交互几何进行重定向、扩增接近运动并做物理可执行性修正；冻结通用动作跟踪先验，训练特权交互残差教师，再用在线 DAgger 蒸馏出参考无关的视觉学生 |
| **策略接口** | 项目页给出的学生输入为 4 帧 36×64 深度历史和 808 维本体历史，输出 29 维关节位置动作，控制频率 50 Hz |
| **部署设定** | Unitree G1 仅凭头部深度相机和本体感觉完成关门；部署时不需要参考动作、特权物体状态或中间控制指令 |

## 源码运行时序图

**不适用**（截至 2026-10-03，项目页与 arXiv 未提供可运行官方代码仓库；项目页提供方法、训练参数和演示视频）。

## 实验与评测

- 论文摘要报告了多种门配置与机器人初始条件下的仿真和真机门关闭实验，并声称单一策略可泛化到任务变化且能从仿真迁移到真实 Unitree G1。
- 项目页列出 7 段真机演示、11 段仿真迁移演示和 1 段失败案例；这些演示支持定性检查，不能替代完整成功率和跨方法评测表。
- 项目页公开了 PPO 与学生网络的训练超参数、输入输出规格及逐项奖励定义；在代码和数据未公开的情况下，这些细节有助于理解方法，但不足以复现全流程。
- 读定量结果时应回到论文 PDF 核对任务划分、成功标准、样本量与 baseline 协议，避免将关门演示与不同任务的结果横向比较。

## 与其他工作对比

> 本页为清单级摘要，下表只做**定位对照**：G1 关门演示与下列各页不共享任务与评测协议，不可横比。

| 对照 | 差异读法 |
|------|----------|
| [WholeBodyWAM](./paper-wholebodywam.md) | 同为人形全身操作，中间件不同：WholeBodyWAM 走世界模型预测，ViLoMan 走特权 teacher 蒸馏到机载深度 + 本体。预测未来 vs 压缩专家 |
| [RebarSim](./paper-rebarsim.md) | 同批次里同一套「特权 teacher → 视觉 student + DAgger」配方的另一端：RebarSim 用在毫米级插入，ViLoMan 用在全身关门。配方能跨这个尺度本身是看点 |
| [OmniMimic](./paper-omnimimic.md) | 同批次另一条「先造监督」：OmniMimic 在同 embodiment 内增广步态，ViLoMan 跨 embodiment 重定向人–物交互。监督从哪来，决定覆盖边界 |
| [Loco-manipulation](../tasks/loco-manipulation.md) | 该页给任务族评测口径；ViLoMan 属「机载感知 + 全身接触」一支，与分离式先导航后操作一支的取舍是**接触时机是否需要与步态协同** |
| [10 篇技术地图](../overview/contact-wm-10-papers-technology-map.md) | 同批次横向对照入口：本文列 **扫读** 档位 |

## 结论

**ViLoMan 的可复用思路是先用交互保持型重定向和物理修正构造可执行监督，再把特权接触控制蒸馏到机载深度与本体策略；复现判断仍受限于未公开的代码和数据。**

1. **分解训练难题** — 冻结通用动作跟踪器，只学习与门接触相关的残差教师，再用在线 DAgger 训练学生，减少直接从视觉探索完整全身控制的负担。
2. **留意输入开销** — 部署策略融合 4 帧低分辨率深度与 808 维本体历史，以 50 Hz 输出 29 维动作；传感器和时序堆叠是复现接口的一部分。
3. **区分演示与基准** — 项目页有真机和仿真视频及详细奖励表，但页面未列完整统计评测协议；比较前需核对论文中的成功定义和样本量。
4. **当前开源边界明确** — 截至 2026-10-03，项目页仍为匿名补充材料页，没有代码仓库或数据下载入口，因此不应把超参数披露等同于可复现实现。

## 关联页面

- [loco-manipulation](../tasks/loco-manipulation.md)
- [locomotion](../tasks/locomotion.md)
- [WholeBodyWAM](./paper-wholebodywam.md)
- [sim2real](../concepts/sim2real.md)
- [10 篇技术地图](../overview/contact-wm-10-papers-technology-map.md)

## 参考来源

- [viloman_arxiv_2609_19340.md](../../sources/papers/viloman_arxiv_2609_19340.md)
- [viloman.md](../../sources/sites/viloman.md)
- [wechat_embodied_station_10_papers_contact_wm_2026-09-18.md](../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)
- [arXiv:2609.19340](https://arxiv.org/abs/2609.19340)

## 推荐继续阅读

- [项目页：ViLoMan 补充材料](https://viloman-anonymous.pages.dev/)
- [arXiv PDF](https://arxiv.org/pdf/2609.19340)
