---
type: entity
tags: [paper, humanoid-paper-notebooks, manipulation, bimanual, active-perception, egocentric, vr-teleoperation, memory, berkeley]
status: complete
updated: 2026-10-03
arxiv: "2511.00153"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../tasks/bimanual-manipulation.md
  - ./paper-notebook-vision-in-action-learning-active-perception-from.md
  - ./paper-notebook-learning-to-look-seeking-information-for-decisio.md
  - ../methods/imitation-learning.md
  - ./paper-ego-03-egomimic.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/papers/humanoid_pnb_egomi.md
summary: "机器人从人类视频学操作，要跨越具身差距。人在做任务时会主动协调头与手，用动态视角变化与视觉搜索策略。EgoMI 捕捉同步的末端执行器与头部轨迹，可迁移到半人形机器人；并引入一个记忆增强策略（memory-augmented policy），选择性纳入历史观测以应对视角切换。在带可动相机头的双臂机器人上测试：显式建模头部运动的策略持续优于基线，说明协调的手眼学习能有效弥合人-机具身差距（针对半人形）。"
---

# EgoMI

**EgoMI: Learning Active Vision and Whole-Body Manipulation from Egocentric Human Demonstrations** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

机器人从人类视频学操作，要跨越具身差距。人在做任务时会主动协调头与手，用动态视角变化与视觉搜索策略。EgoMI 捕捉同步的末端执行器与头部轨迹，可迁移到半人形机器人；并引入一个记忆增强策略（memory-augmented policy），选择性纳入历史观测以应对视角切换。在带可动相机头的双臂机器人上测试：显式建模头部运动的策略持续优于基线，说明协调的手眼学习能有效弥合人-机具身差距（针对半人形）。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Active Vision | 主动视觉，主动调整视角/搜索 |
| Head-Hand Coordination | 头手协调 |
| Memory-Augmented | 记忆增强，选择性用历史观测 |
| Embodiment Gap | 具身差距，人与机器人形态差异 |
| Actuated Camera Head | 可动相机头 |
| Semi-humanoid | 半人形（双臂 + 可动头） |

## 为什么重要

- **主动视觉（头手协调）是人类操作的隐藏要素**，忽略它会限制从人类视频学习的上限；
- **记忆增强**对视角动态变化的任务很关键；
- **半人形（双臂 + 可动头）**是连接人类数据与人形的实用载体；
- 与 Vision in Action、Learning to Look 等主动感知工作呼应。

## 解决什么问题

从人类视频学操作有**具身差距**，且： - 人会**主动协调头与手**（动态视角、视觉搜索），机器人若忽略则学不好； - **视角快速切换**让策略难以利用历史观测。

EgoMI 要：把**头部主动运动**显式建模，并用**记忆**应对视角切换，迁移到半人形。

## 核心机制

1. **同步末端 + 头部轨迹捕捉**：把主动视觉纳入学习；
2. **记忆增强策略**：选择性用历史观测应对视角切换；
3. **显式头部运动建模**：弥合人-机具身差距；
4. **半人形验证**：可动相机头双臂机器人上优于基线。

方法拆解（深读笔记小节）：同步捕捉末端 + 头部轨迹；记忆增强策略（应对视角切换）；显式头部运动建模；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos.html> |
| arXiv | <https://arxiv.org/abs/2511.00153> |
| 源码 | **未开源**：项目页 <https://egocentric-manipulation-interface.github.io> 标注 “Code (coming soon)”，截至 2026-09-28 未列 GitHub 链接 |
| 作者 | Justin Yu、Yide Shentu、Di Wu、Pieter Abbeel、Ken Goldberg、Philipp Wu（UC Berkeley） |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**设置**：采集端为 VR 头显 + 手持夹爪，同步记录头部位姿、手部轨迹、夹爪动作与第一视角 / 腕部视频（约 200 小时 EgoMI 演示）；部署端为改装的 Rainbow RBY1 轮式半人形（6-DoF 躯干、2×7-DoF 手臂、6-DoF YAM 机械臂当「脖子」+ ZED2i 相机）。**全部策略只用人类演示训练，零机器人遥操作数据。**

| 实验 | 对照 | 结果 |
|------|------|------|
| 桌面跨工作区双手交接 | 29D（含头部 SE(3) 动作 + 头部相机） vs 20D（仅腕部相机） | **36/40** vs 29/40 |
| 同上，头部相机保留但头部固定 | 去掉主动头部运动 | 仅 2/20 |
| 货架搜索 + 空中交接 + 放篮（每罐 1 分） | 29D vs 20D | **35/40** vs 0/40 |
| 需要记住侧桌信息的选择任务 | SPARKS 关键帧记忆 vs 单帧策略 | **31/40** vs 21/40（接近随机） |

- **失败模式**：20D 策略主要败在跨工作区交接——腕部视野缺少放置位置的上下文；29D 策略受益于人「先看后伸手」的预注意头动。
- **记忆的必要性**：单帧策略不会去「看左边」，直接在前方桌子上凭模糊视野抓取；SPARKS 把关键帧加入记忆后能正确消歧。
- 论文观察到不需要视觉增广也能迁移，作者假设主动头部带来的视角变化迫使策略学到更聚焦任务的特征。

## 与其他工作对比

| 工作 | 主动视觉的来源 | 与 EgoMI 的差异 |
|------|------|------|
| [Vision in Action](./paper-notebook-vision-in-action-learning-active-perception-from.md) | 遥操作时记录操作者头动，机器人侧用 6-DoF 颈部相机 | 同样模仿人的主动视角；EgoMI 采集时完全不需要机器人 |
| [Learning to Look](./paper-notebook-learning-to-look-seeking-information-for-decisio.md) | 策略学习主动寻找信息 | 以强化 / 模仿学习信息寻求行为；EgoMI 直接模仿人头动 |
| [EgoMimic](./paper-ego-03-egomimic.md) | 第一视角人类视频 + 机器人数据协同 | 需机器人数据；EgoMI 零机器人数据且显式建模头部 |
| 20D 腕部相机策略（论文基线） | 无 | 货架任务 0/40，说明没有头部视角时无法定位视野外目标 |

## 结论

**EgoMI 的主张是：从人类视频学操作时被丢掉的不只是手的轨迹，还有「人是怎么看」——把头部主动运动显式建模，才是弥合具身差距的那一步。**

- 两个机制互相依赖：**同步捕捉末端执行器 + 头部轨迹** 把主动视觉纳入学习，**记忆增强策略**（选择性纳入历史观测）则用来扛住视角快速切换造成的观测不连续；只做前者会被后者卡住。
- 证据形态是对照而非绝对指标：在带可动相机头的双臂机器人上，**显式建模头部运动的策略持续优于基线**。
- 适用边界写得很直白——迁移目标是 **半人形（双臂 + 可动头）**，本页并未把结论外推到全身人形。
- 与 Vision in Action、Learning to Look 等主动感知工作同向，EgoMI 的独特处在于主动视觉是 **作为演示数据的一部分被采集和模仿**，而不只是机器人侧的一个控制模块。
- 头部运动是决定性变量而非锦上添花：货架任务去掉头部后从 35/40 掉到 0/40，保留头部相机但固定视角也只剩 2/20。

## 局限与风险

- **采集设备笨重**：部分用户难以长时间佩戴（论文自述）。
- **物理具身不匹配**：机器人可动头的活动范围超出人类自然范围，重定向会约束性能。
- **SPARKS 是固定打分启发式**：不自适应更新记忆。
- **只验证轮式半人形**：Rainbow RBY1 双臂 + 可动头，未扩展到全身人形或腿足平台。
- **开源边界**：代码标注 coming soon；源码运行时序图 **不适用**（无公开实现）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 任务语境：双臂操作：[bimanual-manipulation](../tasks/bimanual-manipulation.md)
- 主动感知同向工作 ViA：[paper-notebook-vision-in-action-learning-active-perception-from](./paper-notebook-vision-in-action-learning-active-perception-from.md)
- 信息寻求式主动视觉：[paper-notebook-learning-to-look-seeking-information-for-decisio](./paper-notebook-learning-to-look-seeking-information-for-decisio.md)
- 模仿学习：[imitation-learning](../methods/imitation-learning.md)
- 第一视角人类演示路线对照：[paper-ego-03-egomimic](./paper-ego-03-egomimic.md)
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 主动视觉（头部运动）在感知栈选型中的位置
- 第一视角全身人类数据预训练路线：[λ₀ / HumanVerse-500](./paper-lambda0-egocentric-human-pretraining.md) — 500 h 第一视角全身移动操作数据三阶段预训练，迁到全身人形 VLA（arXiv:2610.00438）

## 参考来源

- [humanoid_pnb_egomi.md](../../sources/papers/humanoid_pnb_egomi.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos.html>
- 论文：<https://arxiv.org/abs/2511.00153>
- 论文正文（实验与结论 / 局限）：<https://arxiv.org/html/2511.00153>

## 推荐继续阅读

- [机器人论文阅读笔记：EgoMI](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos/EgoMI__Learning_Active_Vision_and_Whole-Body_Manipulation_from_Egocentric_Human_Demos.html)
- 项目页：<https://egocentric-manipulation-interface.github.io>
