---
type: entity
tags: [paper, project, humanoid, whole-body-manipulation, contact-rich-manipulation, loco-manipulation, reinforcement-learning, sim2real, proprioception, unitree-g1, amazon, ucsd, umich]
status: complete
updated: 2026-10-17
project_id: whole-body-dexterity
arxiv: "2610.22184"
project: https://www.wholebodydex.com/
related:
  - ../tasks/loco-manipulation.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/whole-body-control.md
  - ./paper-pantact-whole-body-contact-interface.md
  - ./paper-notebook-visualmimic.md
sources:
  - ../../sources/papers/whole_body_dexterity_arxiv_2610_22184.md
  - ../../sources/sites/wholebodydex-project.md
summary: "Whole-Body Dexterity（arXiv:2610.22184，Amazon/UCSD/密歇根大学）把人形机器人的手臂与躯干作为可切换接触面，用仿真可行抓取数据训练 RL 专家并蒸馏为单策略；Unitree G1 真机全身操作试验完成率为 45/50，项目页尚未提供代码或数据下载。"
---

# Whole-Body Dexterity：让人形机器人用身体重定位物体

**Whole-Body Dexterity（WBD）**（[arXiv:2610.22184](https://arxiv.org/abs/2610.22184)，[项目页](https://www.wholebodydex.com/)）把手臂、躯干等身体部位视为可用接触面，让人形机器人通过不断改变支撑位置来搬运大件物体。作者来自 Amazon、加州大学圣地亚哥分校与密歇根大学。

## 一句话定义

用身体多处接触支撑物体，并学习在这些接触区域之间重定位的全身灵巧操作框架。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBD | Whole-Body Dexterity | 将机器人身体表面作为物体操作接触面的能力与框架 |
| RL | Reinforcement Learning | 用于训练全身抓取重定位专家策略 |
| Sim2Real | Simulation to Real | 从仿真训练迁移到 Unitree G1 真机 |
| G1 | Unitree G1 Humanoid Robot | 论文真机验证所用的人形机器人平台 |

## 为什么重要

- **扩展操作接触面：** 操作不再只依赖双手抓握，机器人可把物体靠在手臂或躯干上支撑、滑动和转移。
- **目标是接触间重定位：** 框架覆盖从一种可行全身抓取到另一种抓取的任意到任意重定位，而不是维持单次稳定抓取。
- **与示教跟踪分工：** 操作员提供粗粒度运动参考，学习到的策略负责身体接触变化与物体重定位。

## 流程总览

```mermaid
flowchart LR
  grasps["仿真生成可行全身抓取数据"] --> sample["采样起始与目标抓取"]
  sample --> experts["训练强化学习重定位专家"]
  experts --> policy["蒸馏为单一策略"]
  operator["操作员粗粒度动作参考"] --> policy
  policy --> robot["Unitree G1 全身接触与物体重定位"]
```

数据集提供可行的起始和目标身体抓取。强化学习专家学习把物体从起始接触状态移到目标接触状态，随后蒸馏得到单一策略；部署时由操作员给出较粗的动作参考。

## 方法与系统边界

- **身体接触：** 手臂与躯干都可参与物体支撑，机器人通过建立、转移和释放接触完成重定位。
- **策略训练：** 先训练若干覆盖不同起始与目标抓取组合的 RL 专家，再将其压缩为单一策略。论文摘要描述的策略输入包括本体感觉与物体几何信息。
- **真机执行：** 项目页描述通过 Xsens 动作捕捉服采集操作员的 SMPL 身体动作；操作员控制上身，另一个控制器处理下身。
- **长程任务：** 从货架拿起纸箱、夹在一侧手臂下、搬到桌边，再解夹并放下。

## 实验与评测

项目页汇总真实全身操作试验的总体完成率如下：

| 方法 | 完成次数 | 完成率 |
|------|----------|--------|
| **WBD 蒸馏策略** | 45/50 | **90%** |
| 直接运动重定向（GMR） | 9/50 | 18% |
| SONIC 通用全身控制器 | 2/50 | 4% |

评测还包括篮球和纸箱从双臂前抱姿态到肩部或腋下区域的往返重定位。货架到桌面的长程任务分解为取物、腋下夹持、搬运、解夹和放置，项目页按阶段报告了成功次数。上述比例来自该项目报告的真机任务集，不能直接外推到未评测物体、机器人或操作场景。

## 结论

**WBD 的关键贡献是把“身体接触状态变化”作为通用操作目标，再用仿真抓取数据和专家蒸馏训练单策略；真机 90% 完成率展示了这一设定的潜力。**

- 复现或评估时应把接触区域转移作为核心任务变量，单看末端抓取成功率不足以衡量此类能力。
- 45/50 的总体结果比单个演示更有说服力，但基线对比仍受试验设置、物体范围和熟练遥操作员影响。
- 系统仍由人提供粗运动参考，论文展示的是交互式全身操作控制器，不是自主任务规划系统。
- 项目页截至 2026-10-17 未列代码或数据下载入口，目前无法据公开页面独立复现完整训练和部署管线。

## 源码运行时序图

**不适用：**截至 2026-10-17，官方项目页和 arXiv 页面未提供可运行代码仓库或部署入口。

## 局限与风险

- **遥操作参与任务执行：** 人类通过 Xsens/SMPL 提供动作参考，因此高完成率不等于机器人从语言或目标指令自主完成整项任务。
- **公开复现材料有限：** 项目页目前提供论文、指标与演示说明，没有公开训练代码、部署代码或数据集下载链接。
- **实验覆盖范围有限：** 真机展示集中于 Unitree G1、篮球与纸箱等任务；不能据此推断接触机制可直接迁移到任意物体或本体。
- **信息输入需看正文：** 论文摘要写明本体感觉与物体几何信息；项目页将策略简称为 proprioception-only。精确观测接口应以论文方法章节为准。

## 与其他页面的关系

- [Loco-Manipulation](../tasks/loco-manipulation.md) — 全身移动与接触操作的任务脉络。
- [PanTact](./paper-pantact-whole-body-contact-interface.md) — 从人类穿戴接触界面示范全身接触任务，与 WBD 的仿真抓取重定位训练形成对照。
- [VisualMimic](./paper-notebook-visualmimic.md) — 以视觉策略与关键点跟踪实现全身 loco-manipulation。
- [接触丰富操作](../concepts/contact-rich-manipulation.md) — 接触状态与多约束操作的相关概念。

## 参考来源

- [Whole-Body Dexterity 论文来源](../../sources/papers/whole_body_dexterity_arxiv_2610_22184.md)
- [Whole-Body Dexterity 项目页归档](../../sources/sites/wholebodydex-project.md)
- 论文：<https://arxiv.org/abs/2610.22184>
- 项目页：<https://www.wholebodydex.com/>

## 推荐继续阅读

- [Whole-Body Dexterity 项目页](https://www.wholebodydex.com/)
- [arXiv 论文](https://arxiv.org/abs/2610.22184)
- [PanTact](./paper-pantact-whole-body-contact-interface.md) — 以可穿戴接触示范教会机器人用身体操作。
