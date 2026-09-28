---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, tactile, dataset, deformable-object, contact-rich, gist, unitree]
status: complete
updated: 2026-09-28
arxiv: "2510.25725"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../methods/action-chunking.md
  - ./paper-notebook-learning-visuotactile-skills-with-two-multifinge.md
sources:
  - ../../sources/papers/humanoid_pnb_a-humanoid-visual-tactile-action-dataset-for-con.md
summary: "接触丰富操作在机器人学习中越来越重要，但以往机器人学习数据集多聚焦刚体，低估了真实操作中压力条件的多样性。为填补此空白，本文提出一个面向可变形软物体操作的人形视觉-触觉-动作数据集。数据用带灵巧手的人形通过遥操作采集，包含视觉与触觉多模态信号，并覆盖不同压力条件。该工作旨在激励未来研究——开发具备先进优化策略、能有效利用复杂多样触觉信号的模型，而非在摘要中报告具体数值。"
---

# A Humanoid Visual-Tactile-Action Dataset for Contact-Rich Manipulation

**A Humanoid Visual-Tactile-Action Dataset for Contact-Rich Manipulation** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

接触丰富操作在机器人学习中越来越重要，但以往机器人学习数据集多聚焦刚体，低估了真实操作中压力条件的多样性。为填补此空白，本文提出一个面向可变形软物体操作的人形视觉-触觉-动作数据集。数据用带灵巧手的人形通过遥操作采集，包含视觉与触觉多模态信号，并覆盖不同压力条件。该工作旨在激励未来研究——开发具备先进优化策略、能有效利用复杂多样触觉信号的模型，而非在摘要中报告具体数值。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Visual-Tactile-Action | 视觉-触觉-动作多模态 |
| Contact-Rich | 接触丰富（频繁/复杂接触） |
| Deformable Object | 可变形软物体 |
| Pressure Condition | 压力条件（按压力度等） |
| Teleoperation | 遥操作采集 |
| Dexterous Hand | 灵巧手 |

## 为什么重要

- **触觉是接触丰富/软物体操作的关键模态**，视觉常不足；
- **可变形物体 + 多压力**更贴近真实家务/护理场景；
- **数据集 + 灵巧手人形**为触觉学习提供稀缺资源；
- 与 CHIP、HMC 等"柔顺/力"工作在"接触"主题上互补。

## 解决什么问题

接触丰富操作的数据缺口： - 现有数据集多为**刚体**，少**可变形软物体**； - **压力条件多样性**被低估； - 缺**人形 + 灵巧手 + 视触觉**的多模态数据。

论文要：构建一个**人形视触觉-动作数据集**，专门覆盖**软物体 + 多压力**的接触丰富操作。

## 核心机制

1. **人形视觉-触觉-动作数据集**：面向接触丰富操作；
2. **可变形软物体 + 多压力条件**：填补刚体/单一压力的空白；
3. **多模态 + 灵巧手遥操作采集**：真实可执行；
4. **激励触觉模型研究**：推动有效利用复杂触觉信号。

方法拆解（深读笔记小节）：人形 + 灵巧手 + 遥操作采集；视觉 + 触觉多模态；覆盖可变形软物体与多压力条件；目标；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Humanoid_Visual-Tactile-Action_Dataset_for_Contact-Rich_Manipulation/A_Humanoid_Visual-Tactile-Action_Dataset_for_Contact-Rich_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2510.25725> |
| 源码 | **未公开**：论文与 arXiv 页面未给出数据集或代码链接，截至 2026-09-28 未找到项目页 |
| 作者 | Eunju Kwon、Seungwon Oh、In-Chang Baek、Yunho Choi、Kyung-Joong Kim 等（GIST 等） |
| 发表 | 2025 年 10 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**采集设置**：在 Unitree 人形遥操作流程上扩展多模态记录——头部第一视角相机（848×480）、左侧 1 m 处 RealSense D435 第三视角、手臂与手指关节本体、Inspire RH56-DFX 灵巧手触觉（每手 1062 个传感点，覆盖手指与手掌）；压阻式触觉地毯实时给出压力热图，帮助操作者保持设定的压力档位。

**数据规模**：毛巾 / 海绵 × 强 / 弱压力 = 4 个任务，每任务约 77–80 段（每段 20–30 秒），3 名采集者，共约 **10.19 万**个样本；另采集刚体对照数据。触觉读数由 0–4095 归一化到 0–1。

- **数据分析**：软物体操作时接触区域多变、分布随时间演化；刚体接触分布基本恒定。稠密触觉（2124 点）的 t-SNE 能清晰分开 4 种压力条件，降到 42 点的稀疏表示（模仿 FSR 触觉手套，约减少 98%）则区分不开。
- **策略实验**：触觉按手部分块转成 2D 图像，经 CNN 与视觉一起送入 ACT；比较 ACT-Dense 与 ACT-Sparse，指标为动作 MAE（80/20 划分，3 个 seed，训练 10 万步）。**两者差距很小**；训练损失都稳定下降，但测试损失下降有限；Towel Weak 测试曲线波动大，Sponge 强 / 弱之间差距明显。
- **真机部署**：只给出定性观察与补充视频——需要精细接触控制的海绵抓取更难。

## 与其他工作对比

| 工作 | 触觉数据 | 与本文的差异 |
|------|------|------|
| [HATO](./paper-notebook-learning-visuotactile-skills-with-two-multifinge.md) | 双 UR5e + 改装义肢手触觉 | 给出任务成功率与模态消融；本文只报告 MAE 与定性真机结果 |
| [Deform360](./paper-deform360-deformable-visuotactile-dataset.md) | 可变形物体多视角视触觉数据 | 同为可变形物体视触觉数据集，可对照数据规模与采集方式 |
| FSR 触觉手套式稀疏表示 | 约 42 个传感点 | 本文用它作为「稀疏」对照，显示其难以区分压力条件 |
| [Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md) | 人形触觉预测 | 方法侧工作；本文提供的是数据与分析 |

## 结论

**这是一篇「补数据缺口」而非「提新方法」的工作：它押注接触丰富操作的瓶颈在于缺可变形物体 + 多压力条件的视触觉数据，而不是缺模型。**

- 真正的交付物是数据集本身——人形 + 灵巧手遥操作采集，视觉与触觉多模态同步，并刻意覆盖可变形软物体与不同压力条件，补的是以往数据集偏刚体、低估压力多样性的空白。
- 论文自陈目标是**激励后续研究**：正文里稠密与稀疏触觉的 ACT 动作 MAE 差距很小，只有离线指标、没有任务成功率，因此不要把它当作性能对比的引用来源。
- 适用边界：面向家务/护理这类软物体、接触丰富的场景；纯刚体抓取或低接触任务从中获益有限。
- 遥操作采集决定了规模与多样性受人力成本约束：4 个任务各约 80 段，共约 10.19 万样本，数据集目前也未公开。
- 与 CHIP、HMC 等「柔顺/力」方向在「接触」主题上互补：那一侧解决怎么控，本页解决拿什么数据学。

## 局限与风险

- **策略收益未证实**：稠密与稀疏触觉的 ACT 在动作 MAE 上差距很小，论文把原因归于优化难度（维度高、噪声大），尚未给出能利用稠密触觉的方法。
- **只有离线指标**：没有任务成功率；真机部署仅定性描述与视频。
- **任务覆盖窄**：毛巾、海绵两种软物体 × 两档压力，共约 320 段演示。
- **压力档位依赖人工控制**：操作者看触觉地毯热图维持「强 / 弱」，档位是定性的。
- **数据未公开**：截至核查日未见下载链接，无法直接复用。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 接触丰富型操作：[contact-rich-manipulation](../concepts/contact-rich-manipulation.md)
- 触觉感知：[tactile-sensing](../concepts/tactile-sensing.md)
- 视触觉融合：[visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- 基线策略 ACT：[action-chunking](../methods/action-chunking.md)
- 双手视触觉策略学习 HATO：[paper-notebook-learning-visuotactile-skills-with-two-multifinge](./paper-notebook-learning-visuotactile-skills-with-two-multifinge.md)

## 参考来源

- [humanoid_pnb_a-humanoid-visual-tactile-action-dataset-for-con.md](../../sources/papers/humanoid_pnb_a-humanoid-visual-tactile-action-dataset-for-con.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Humanoid_Visual-Tactile-Action_Dataset_for_Contact-Rich_Manipulation/A_Humanoid_Visual-Tactile-Action_Dataset_for_Contact-Rich_Manipulation.html>
- 论文：<https://arxiv.org/abs/2510.25725>
- 论文正文（数据集统计、ACT 实验）：<https://arxiv.org/html/2510.25725>

## 推荐继续阅读

- [机器人论文阅读笔记：A Humanoid Visual-Tactile-Action Dataset for Contact-Rich Manipulation](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Humanoid_Visual-Tactile-Action_Dataset_for_Contact-Rich_Manipulation/A_Humanoid_Visual-Tactile-Action_Dataset_for_Contact-Rich_Manipulation.html)
