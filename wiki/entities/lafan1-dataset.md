---
type: entity
title: LaFAN1（Ubisoft La Forge 动捕数据集）
tags: [dataset, mocap, bvh, locomotion, animation, ubisoft, siggraph-2020]
summary: "LaFAN1 是 Ubisoft La Forge 在 GitHub 发布的 BVH 动捕数据集（SIGGRAPH 2020 论文配套），覆盖行走、舞蹈、跌倒恢复等多主题；许可为 CC BY-NC-ND 4.0，工程上常被用作全身跟踪与重定向研究的基准动作源。"
updated: 2026-05-15
status: complete
related:
  - ../concepts/motion-retargeting.md
  - ./wbc-fsm.md
  - ../tasks/locomotion.md
  - ../methods/motion-retargeting-gmr.md
  - ./amass.md
sources:
  - ../../sources/repos/ubisoft-laforge-animation-dataset.md
---

# LaFAN1（Ubisoft La Forge Animation Dataset）

**LaFAN1** 指 Ubisoft 在仓库 [`ubisoft/ubisoft-laforge-animation-dataset`](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 中发布的 **La Forge 动捕动画数据集**：序列以 **BVH** 存储，README 给出 **5 名被试、77 条序列、约 49.6 万帧 @30Hz（约 4.6 小时）** 等统计，并配套 **Python 评估脚本** 与若干插值基线。

## 为什么重要？

- **动作语义覆盖**：除平地行走外，包含 **障碍地形、跌倒起身、爬行、瞄准移动** 等，对 **recovery / 多接触** 研究比单一行走库更有信息量。
- **社区可复现基准**：README 定义 **L2Q / L2P / NPSS** 等指标与 `evaluate_test.py` 期望区间，便于论文对比。
- **与部署案例衔接**：例如 **[wbc_fsm](./wbc-fsm.md)** 公开链路中采用 LAFAN1 子集经重定向训练 ONNX 跟踪策略，读者可把「数据形态」与「上机策略」对照阅读。

## 许可与合规（工程必读）

- 仓库声明采用 **CC BY-NC-ND 4.0**（见 `license.txt`）：**非商业**且对**衍生作品**限制严格。
- 若目标包含 **商业产品**、**再分发清洗后的子集**或与训练权重一并发布，必须在法务流程中单独评估；本页不构成法律意见。

## 工程要点

- **Git LFS**：大 zip 通过 LFS 存储；未安装 LFS 时克隆可能导致 zip 损坏与 `BadZipfile`（README 明确提示）。
- **文件命名**：`[theme][take]_[subjectID].bvh`；同主题同 take 号表示同期棚内录制，可做多被试对齐分析。

## 常见误区或局限

- **「BVH 可直接驱动任意人形」**：骨骼比例、根坐标与世界缩放仍须与 **[Motion Retargeting](../concepts/motion-retargeting.md)** 管线一致处理。
- **许可≠开源权重**：在 GitHub 上训练得到的模型权重是否可随项目分发，取决于你的训练数据使用场景与上层许可，不等同于「代码 MIT 即可商用」。

## 与其他页面的关系

- **[wbc_fsm](./wbc-fsm.md)**：以 LAFAN1 为 MoCap 源的重定向 + RL + ONNX 部署范例。
- **[GMR](../methods/motion-retargeting-gmr.md)**：讨论几何重定向时常与 BVH / 骨架比例问题一并出现。
- **[AMASS](./amass.md)**：另一条「SMPL 统一库」路线；LaFAN1 则是「单一棚拍、原始 BVH」路线，二者在表示与许可上均不同。

## 参考来源

- [Ubisoft La Forge Animation Dataset 仓库归档](../../sources/repos/ubisoft-laforge-animation-dataset.md)
- Harvey et al., *Robust Motion In-betweening*, ACM TOG (SIGGRAPH) 2020 — README 内引用信息
- GitHub 仓库：<https://github.com/ubisoft/ubisoft-laforge-animation-dataset>

## 关联页面

- [Motion Retargeting](../concepts/motion-retargeting.md)
- [wbc_fsm（G1 部署框架）](./wbc-fsm.md)
- [Locomotion](../tasks/locomotion.md)
- [GMR（通用动作重定向）](../methods/motion-retargeting-gmr.md)
- [AMASS](./amass.md)

## 推荐继续阅读

- Ubisoft Montreal 论文介绍页：[Robust Motion In-betweening](https://montreal.ubisoft.com/en/automatic-in-betweening-for-faster-animation-authoring/)（README 外链）
- README 中列出的后续使用该数据集的代表性论文（Learned Motion Matching、DReCon 等）
