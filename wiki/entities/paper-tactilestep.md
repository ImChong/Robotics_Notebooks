---
type: entity
tags: [paper, humanoid, locomotion, tactile, unitree-g1]
status: complete
updated: 2026-09-27
arxiv: "2609.28959"
related:
  - ../overview/embodied-research-12-papers-recover-wam-technology-map.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
  - ../../sources/papers/tactilestep_arxiv_2609_28959.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md
summary: "TactileStep（2609.28959）：足底压力→法向力/接触面积/CoP，按步态阶段设计奖励调节 G1 足地交互。"
---

# TactileStep

**TactileStep: Sole Tactile Learning for Regulating Foot-Terrain Interaction in Humanoid Locomotion**（[arXiv:2609.28959](https://arxiv.org/abs/2609.28959)，[项目页](https://tactilestep.github.io/)）收录自 [具身智能小站 12 篇盘点（恢复/WAM 专题）](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)。

## 一句话定义

**仅用地形触觉学习调节人形落脚冲击与支撑质量，而非额外力传感器堆栈。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SR | Success Rate | 任务成功率 |
| WAM | World Action Model | 联合预测未来观测与动作 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| RL | Reinforcement Learning | 强化学习 |

## 为什么重要

- 纳入 [12 篇技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md) 与同期失败恢复、异步 WAM、接触感知、持续学习、安全 RL 条目横向对照。
- 步骤 2.5 开源结论：**待发布**。

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.28959](https://arxiv.org/abs/2609.28959) |
| **项目页** | https://tactilestep.github.io/ |
| **代码** | — |
| **开源** | **待发布** |

## 实验与评测（公众号口径）

- 指标与数字以 [公众号盘点](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md) 与 **原文 PDF** 为准；读复现前核对仿真/真机与 attempt 定义。


## 源码运行时序图

**不适用**（无统一官方入口或未开源）。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [Contact Estimation](../concepts/contact-estimation.md) | 常见做法从关节力矩/电流推断足端接触或装足端 F/T；TactileStep 直接用 **足底压力** 得到法向力、接触面积与 CoP |
| [Tactile Sensing](../concepts/tactile-sensing.md) | 触觉主场是灵巧抓取与接触丰富操作；TactileStep 把触觉放到 **人形足底**，调节足地交互 |
| [人形运控奖励函数](../concepts/humanoid-policy-reward-functions.md) | 「步态与接触」类奖励（相位、冲击、滑移）通常用仿真量；TactileStep 用触觉量 **按步态阶段** 设计奖励 |
| [CoPRE](./paper-copre-proprioceptive-contact.md) | 同期接触感知条目；CoPRE 只用 **本体力矩残差** 做机械臂接触检测，TactileStep 依赖 **足底触觉** |
| [DWMP](./paper-dwmp.md) | 同为 G1 人形 locomotion；DWMP 靠本体 + 深度双世界模型越障，TactileStep 聚焦 **落脚冲击与支撑质量** |

## 结论

**总判：TactileStep 适合作为「仅用地形触觉学习调节人形落脚冲击与支撑质量，而非额外力传感器堆栈。…」方向的入口页；机制细节以 arXiv 与项目页为准。**

1. 与 [12 篇技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md) 对照，避免与其它 arXiv 混淆。
2. 开源为 **待发布** 时从项目页/GitHub 再核实一次再写复现计划。
3. 涉及异步 WAM 或恢复评测时，同时记录 **正常起点 SR** 与 **偏差后恢复率**（若适用）。

## 关联页面

- [具身研究 12 篇（恢复/WAM）技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [论文归档](../../sources/papers/tactilestep_arxiv_2609_28959.md)
- [公众号 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)

## 推荐继续阅读

- [arXiv:2609.28959](https://arxiv.org/abs/2609.28959)
- [项目页](https://tactilestep.github.io/)

