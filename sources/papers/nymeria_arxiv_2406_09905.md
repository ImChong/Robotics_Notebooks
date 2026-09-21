# Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild

> 来源归档（深读 · arXiv:2406.09905 · ECCV 2024）

- **标题：** Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild
- **作者：** Lingni Ma, Yuting Ye, Fangzhou Hong, Vladimir Guzov, Yifeng Jiang, Rowan Postyeni 等（Meta Reality Labs / Project Aria）
- **类型：** paper / dataset / egocentric / human-motion
- **arXiv：** <https://arxiv.org/abs/2406.09905>
- **项目页 / 数据：** <https://www.projectaria.com/datasets/nymeria/>
- **代码：** <https://github.com/facebookresearch/nymeria_dataset>
- **入库日期：** 2026-09-21
- **一句话说明：** 野外最大规模多模态 egocentric 人类运动数据集论文：300 h 日常活动、Aria + miniAria + XSens GT + observer 第三人称，全设备同步定位到统一度量 3D 世界，并附层级 motion-language 标注（310.5K 句 / 8.64M 词）。

## 核心摘录（面向 wiki 编译）

### 1) 采集栈与同步

- **要点：** 参与者佩戴 **Project Aria** 头显（RGB/灰度/眼动/IMU/磁力计/气压/音频）、**miniAria 腕带**、**XSens MVN Link** 全身惯导动捕；另有一名 **observer** 佩戴 Aria 提供第三人称视角；硬件同步 + 优化注册到 **同一 metric 3D 世界**。
- **对 wiki 的映射：** [`wiki/entities/paper-nymeria.md`](../../wiki/entities/paper-nymeria.md)、[`wiki/entities/nymeria-dataset.md`](../../wiki/entities/nymeria-dataset.md)

### 2) 规模与 motion-language

- **要点：** **300 h** 活动 · **1200** 序列 · **264** 参与者 · **50** 地点 · **20** 场景脚本；层级语言：**motion narration / atomic action / activity summarization**；合计 **310.5K** 句、**8.64M** 词、词表 **6545**。
- **对 wiki 的映射：** 同上

### 3) 基准实验（论文演示潜力）

- **要点：** 在 egocentric **body tracking**、**motion synthesis**、**action recognition** 上评测多种 SOTA；证明多设备 egocentric + GT 运动 + 语言对野外理解任务的增益。
- **对 wiki 的映射：** 同上；交叉 [`wiki/entities/paper-egoexomocap.md`](../../wiki/entities/paper-egoexomocap.md)

### 4) 与 Light-O1 Transfer Scaling Law 的关系

- **要点：** [Light-O1 Tech Blog](https://www.lightorigins.com/en/blog/light-o1) 将 **Nymeria** 作为 **egocentric 人类动作** 适配目标之一，报告预训练 token 预算 D 增大后 held-out next-action loss / MPJPE **幂律下降**。
- **对 wiki 的映射：** [`wiki/entities/light-o1.md`](../../wiki/entities/light-o1.md)

## 开源边界（步骤 2.5）

| 状态 | 说明 |
|------|------|
| **已开源** | 数据与工具经 Project Aria 申请下载；GitHub `facebookresearch/nymeria_dataset`；HF `projectaria/Nymeria` |
| **许可** | **CC BY-NC 4.0**（非商用需遵守 Meta 条款） |
| **NymeriaPlus** | 升级版（MHR/SMPL 优化、3D/2D bbox、ShapeR 物体重建等）；`main` 分支优先支持 Plus |

## 对 wiki 的映射

- [paper-nymeria.md](../../wiki/entities/paper-nymeria.md)
- [nymeria-dataset.md](../../wiki/entities/nymeria-dataset.md)
- [nymeria_dataset.md](../repos/nymeria_dataset.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2406.09905>
- 数据集页：<https://www.projectaria.com/datasets/nymeria/>
