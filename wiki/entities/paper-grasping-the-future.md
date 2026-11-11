---
type: entity
project_id: grasping-the-future
tags:
  - paper
  - manipulation
  - flow-matching
  - hand-object-interaction
  - human-video
  - retargeting
  - uw
  - jhu
  - skild-ai
  - unc-chapel-hill
status: complete
updated: 2026-11-11
arxiv: "2611.12879"
related:
  - ../concepts/flow-matching-embodied-policy.md
  - ../concepts/diffusion-model.md
  - ../concepts/motion-retargeting.md
  - ../concepts/contact-rich-manipulation.md
  - ../methods/spider-physics-informed-dexterous-retargeting.md
  - ../methods/imitation-learning.md
  - ../entities/paper-acrossdex-bimanual-dexterous-human-motion-pretraining.md
  - ../entities/molmo-motion.md
sources:
  - ../../sources/papers/grasping-the-future_arxiv_2611_12879.md
original:
  - ../../sources/papers/grasping-the-future_arxiv_2611_12879.md
summary: "Grasping-the-Future（arXiv:2611.12879，UW 等）：掩码时空补全 + 流匹配的统一模型，联合预测 3D 手（腕/指）与物体运动；用 6.2 万条自互联网视频重建的轨迹作监督；预测经 SPIDER 在仿真灵巧手上复现率 73.8%，不等于闭环操作成功。"
---

# Grasping the Future（arXiv:2611.12879）

**Grasping-the-Future**（*Grasping the Future: Joint 3D Hand–Object Forecasting From Internet Videos*，[arXiv:2611.12879](https://arxiv.org/abs/2611.12879)，[项目页](https://grasping-the-future.github.io/)）来自华盛顿大学、北卡教堂山、约翰斯·霍普金斯与 Skild AI。本页依据论文 v1 全文与附录写成；源码截至 2026-11-11 未发布。

## 一句话定义

**一个流匹配 Transformer，用「给定一部分、补全其余」的掩码方式，统一做 3D 手—物体运动的未来预测、手→物 / 物→手预测、目标条件生成与时间补全；训练监督一半以上来自自动从互联网视频重建的 3D 轨迹（论文 Abstract，§3.1，§4.1）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MANO | hand Model with Articulated and Non-rigid defOrmations | 参数化手模型；本文每只手状态含腕位姿 + 15 个指关节旋转（§3.1） |
| FM | Flow Matching | 流匹配生成训练目标（§3.2） |
| DiT | Diffusion Transformer | 骨干采用 AdaLN-Zero 调制（Fig. 2，Appendix A） |
| ADE | Average Displacement Error | 生成帧上位置误差均值，单位 mm（§4.1） |
| FDE | Final Displacement Error | 最后一帧位置误差（§4.1） |
| ES | Energy Score | 多样本评分：平均误差减去样本间平均距离的一半（§4.1，式 4/5） |
| SPIDER | Scalable Physics-Informed DExterous Retargeting | 物理仿真中的灵巧手重定向（§4.3） |
| XHand | — | 论文仿真所用的机器人灵巧手（§4.3） |

## 为什么重要

- 把「手怎么动」与「物体怎么动」放进同一个 3D 生成模型，而不是只预测手或只预测物体（§1）。
- 展示了一条从无 3D 标注的互联网视频获取 3D 手—物体监督的管线，并给出对照实验：加入这些轨迹后单样本预测误差下降（§4.2）。
- 把预测接到仿真机器人手上做物理可行性检验，给出三档重定向成功率（§4.3）。

## 核心原理与流程总览

```mermaid
flowchart LR
  A[互联网视频 Action100M] --> B[感知模型 WiLoR SAM3 SAM3D MoGe MegaSaM]
  B --> C[联合几何优化与质检 约12%保留]
  C --> D[6.2万条3D轨迹]
  E[8个标注数据集] --> F[掩码时空补全 流匹配训练]
  D --> F
  F --> G[观察0.5秒 生成4.5秒 20条样本]
  G --> H[SPIDER重定向到仿真XHand]
```

要点（均来自论文）：

- **表示（§3.1）：** 5 秒窗口，最多两只手 + 一个刚体；手状态 99 维（腕位置旋转 + 15 个 MANO 关节），物体 9 维；全部重表达到「当前帧」相机坐标系以去除相机运动。
- **任务即掩码（§3.1，Table 3）：** 预测 / 手→物 / 物→手 / 目标条件 / 时间补全，训练时各占约 0.19 概率，另有 0.05 随机块掩码。
- **关系 token（§3.2，式 1）：** 每个手—物体对一个 14 维 token（腕在物体系下的位姿 + 五指尖到物体表面距离），让模型直接输出接触关系。
- **Δ-codec 与采样（§3.2）：** 对时间差分编码；推理 10 步 Euler；置信度输入区分标注数据与视频提取数据。
- **附加损失（§3.3）：** 附着损失（指尖距表面 ≤5 mm 的接触帧）与 M=3 样本的能量分数损失。
- **视频提取（§3.4，Appendix B）：** Molmo 2 / SAM 3 / WiLoR / SAM 3D / MoGe / MegaSaM 估计，CoTracker3 + 束调整，联合优化，多重质检；约 12% 候选保留。

## 评测与实验结果

**基准（§4.1，Appendix C）：** 331 个五秒窗口，256 个来自标注数据集、75 个来自互联网视频（后者以论文自己提取的轨迹为参照，不是独立真值）；观察 0.5 s、生成 4.5 s，每个输入采样 20 条。指标为手 / 物体位置的 ADE、FDE（mm），以及多样本能量分数（ES）；接触质量另用接触 F1 与指尖误差（Appendix D）。

| 项目 | 结果 | 出处 |
|------|------|------|
| 对已发布预测器（含 MolmoMotion、MotionForesight 等） | 能量分数口径下 ADE/FDE 低 16–37% | §4.2，Table 2 |
| 单样本未来预测 | 不总优于 stay-put（静止基线反而低 2–8% 误差）；MotionForesight 单样本物体误差更低但几乎不动物体 | §4.2 |
| 加入互联网视频轨迹 | 单样本预测误差下降 | §4.2 |
| 接触消融 | 去掉物理损失或关系 token，接触 F1 与指尖误差变差；主模型抓取帧中 held 仅 15.5% | Appendix D，Table 10 |
| 仿真重定向（XHand，103 例 × 3 种子） | 运动学 14.6% → 加抓取预备 43.7% → SPIDER 动态优化 73.8% | §4.3，Fig. 5 |

## 与其他工作对比

| 工作 | 关注点 | 与本文的区别 |
|------|--------|--------------|
| [MolmoMotion](./molmo-motion.md) | 运动预测模型（本文 Table 2 的对比方法之一） | 各方法输入不同（Table 2），本文在能量分数下误差更低；本页未核对该页细节 |
| [SPIDER 灵巧重定向](../methods/spider-physics-informed-dexterous-retargeting.md) | 物理仿真中的人手到灵巧手重定向 | 本文把它当下游可行性检验工具，73.8% 衡量的是 SPIDER 跟随预测轨迹，而非本文模型直接控制机器人 |
| [AcrossDex](./paper-acrossdex-bimanual-dexterous-human-motion-pretraining.md) | 用人类双手灵巧运动做预训练 | 同样利用人手运动数据；本文重点是手—物体联合 3D 预测与互联网视频 3D 提取管线 |

## 结论

**Grasping-the-Future 用一个掩码时空补全的流匹配模型统一了多种手—物体 3D 运动预测任务，并证明从互联网视频自动提取的 3D 轨迹能作为有效监督。**

1. 任务即掩码，同一模型覆盖预测、手→物、物→手、目标条件与时间补全（§3.1）。
2. 多样本能量分数下比已发布预测器低 16–37%，但单样本未来预测不稳定优于静止基线（§4.2）。
3. 关系 token 与附着损失改善接触，但抓取帧仍以 detached / slipping 为主（Table 10）。
4. 局限：训练与部分评测参照来自含误差的自动提取；重定向只在仿真、103 条已筛选样例上验证，没有闭环策略或真机实验。

## 工程实践

- 规模：103.3M 可训练参数 + 12.7M 冻结（Uni3D、LitePT 等编码器），16×H100 训练约两天，40,000 步（Appendix A，Table 4）。
- 训练采样含课程：视频提取数据占比前 20,000 步约 23.5%，之后线性降到 2%，让末期聚焦更准的标注数据（Appendix A）。
- 提取算力：约 2.8 GPU 分钟 / 候选 clip，61 万候选 clip 约 28,700 H100 小时（Appendix B.3）。
- 重定向：运动学重定向 → 加 SPIDER 抓取预备 → SPIDER 动态优化，对应成功率 14.6% / 43.7% / 73.8%（§4.3，Fig. 5）。成功判据：物体全程被移动，位置误差 <100 mm 且旋转误差 <0.5 rad（均值）。
- 开源状态：§4.1 承诺发布代码与数据；项目页 Code / Model / Data 均为 soon，截至 2026-11-11 无仓库地址，故本页不写 `code:`。

## 与文章说法的核对

对照公众号文章《机器人动作块太长，什么时候该重新规划？》对该论文的摘述（"用流匹配联合预测手—物体三维运动，显式建模接触关系，并探索机器人动作重定向。训练依赖自动恢复的三维监督；预测与重定向结果尚不等于闭环操作成功。"）：

| 文章说法 | 核对结果 |
|----------|----------|
| 流匹配联合预测手—物体三维运动 | 属实（§3.2）。 |
| 显式建模接触关系 | 属实：关系 token（式 1）+ 附着损失（式 3）；消融（Appendix D，Table 10）显示去掉物理损失或关系 token 后接触 F1 与指尖误差变差。 |
| 探索机器人动作重定向 | 属实但限定：仅在**仿真**（XHand + SPIDER）、103 条例子，且这 103 条是「参照轨迹可被 SPIDER 成功重定向」的样例（§4.3），非随机样本；无真机。 |
| 训练依赖自动恢复的三维监督 | 属实：互联网视频轨迹占 62,562/74,312 条（§4.1），但被论文自述含误差（Appendix F），且互联网视频部分的评测参照也是这些提取轨迹（§4.1，Appendix C）。 |
| 预测与重定向结果尚不等于闭环操作成功 | 与论文一致：论文没有闭环策略或真机实验；73.8% 衡量的是仿真中跟随预测轨迹，由 SPIDER 优化机器人控制，不是以预测为输入的策略。该句是文章的判断，论文未用此措辞。 |

文章未提的数字：能量分数下比已发布方法低 16–37%，单样本不总是优于 stay-put（§4.2）。文章与项目页（Code soon）均未给代码；一致。

## 局限

论文自述（Appendix F）：感知模型错误进入训练数据（MoGe 单图深度尺度歧义、WiLoR 手指丢失、SAM 3D 朝向翻转、细长物体失败、MegaSaM 在三脚架镜头下的伪相机运动）；大多数候选 clip 被拒，扩展提取要大量算力。Appendix C 还指出提取轨迹运动偏短（真实 50–150 mm 的运动约短 10–15%）。

我的判断（未经论文验证）：

- 单样本预测对「未来预测」并不优于 stay-put（§4.2），主要优势体现在多样本能量分数；能量分数与 stay-put+匹配噪声对比（Appendix E.1）是论文自己设计的检验。
- 接触质量仍有限：Table 10 中主模型 24.9% 的抓取帧为 detached、47.5% 为 slipping，held 仅 15.5%（作者解释为单参照逐帧评分对合理但不同的抓取不公平，Appendix D）。
- 互联网视频的 75 个评测窗口以提取轨迹为参照，不是独立真值。
- 重定向仅 103 条、仿真、已筛选样例；与闭环策略成功率不可直接比较。
- 评测基线的输入各不相同（Table 2），对比是否完全公平未独立核实。
- 图（Fig. 1–6）只读了图注；视频结果未观看。

## 关联页面

- [Flow Matching 具身策略](../concepts/flow-matching-embodied-policy.md)：流匹配训练目标的背景。
- [扩散模型](../concepts/diffusion-model.md)：生成模型背景。
- [动作重定向](../concepts/motion-retargeting.md)：人手到机器人手的映射问题。
- [SPIDER 灵巧重定向](../methods/spider-physics-informed-dexterous-retargeting.md)：本文物理重定向所用方法。
- [接触丰富操作](../concepts/contact-rich-manipulation.md)：接触约束的任务背景。
- [模仿学习](../methods/imitation-learning.md)：人类视频作监督的上位方法。
- [AcrossDex 双臂灵巧人类运动预训练](paper-acrossdex-bimanual-dexterous-human-motion-pretraining.md)：同类利用人类手部运动的工作。
- [MolmoMotion](molmo-motion.md)：本文 Table 2 的对比方法之一（本页只据论文表格，未核对该页内容）。

## 参考来源

- [论文归档 grasping-the-future_arxiv_2611_12879](../../sources/papers/grasping-the-future_arxiv_2611_12879.md)
- 论文：<https://arxiv.org/pdf/2611.12879v1>
- 项目页：<https://grasping-the-future.github.io/>（Code / Model / Data 标 soon）

## 继续阅读

- 本文前作 ObjectForesight（项目页 <https://objectforesight.github.io>）：物体 3D 轨迹预测。
- 论文 Appendix B 与 C：提取管线与基准协议细节，复现前必读。
