---
type: entity
tags: [paper, vla, cross-embodiment, action-geometry, camera-frame, qwen3-vl, xiaomi-robotics, umac]
status: complete
updated: 2026-08-28
arxiv: "2608.26058"
code: https://github.com/Public-BOTs/UCAG-P
related:
  - ../methods/vla.md
  - ./libero-benchmark.md
  - ../tasks/manipulation.md
  - ../overview/wam-vla-cross-embodiment-9-papers-technology-map.md
  - ../concepts/world-action-models.md
  - ../queries/cross-embodiment-transfer-strategy.md
  - ./qwen-robot-manip.md
  - ./qwen-vla.md
  - ./paper-zero-wam.md
sources:
  - ../../sources/papers/ucag_p_arxiv_2608_26058.md
  - ../../sources/sites/ucag-p.md
  - ../../sources/repos/ucag-p.md
  - ../../sources/blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md
summary: "UCAG-P（arXiv:2608.26058）：相机坐标锚点运动统一手臂/人形/人手；Qwen3-VL-4B；单检查点 LIBERO 98.3%、RoboTwin 88.7%/89.2%；Piper 真机 60/90/75%；代码待发布。"
---

# UCAG-P

**One Policy, Many Embodiments: Unified Camera-Centric Action Geometry Pre-training for Heterogeneous Embodied Manipulation**（[arXiv:2608.26058](https://arxiv.org/abs/2608.26058)，[项目页](https://public-bots.github.io/UCAG-P)）——小米机器人实验室（Xiaomi Embodied Intelligence）；澳门大学（University of Macau）。核心贡献者包括 Shaoqing Xu、Fang Li、Guozhi Zhan 等。

## 一句话定义

**跨本体学习的共享目标不应是关节指令，而应是相机里人人都能看见的锚点运动。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| UCAG-P | Unified Camera-centric Action Geometry Pre-training | 本文相机中心统一动作预训练 |
| VLA | Vision-Language-Action | 共享策略主干；本文骨干是 Qwen3-VL-4B |
| EEF | End-Effector | 末端执行器，对应腕部/抓取中心锚点 |
| LIBERO | Lifelong Robot Learning benchmark | 桌面操作仿真榜；本文 98.3% |
| GR-1 | Fourier GR-1 / RoboCasa 设定 | 29-DoF 人形厨房评测；本文 62.0% |

## 为什么重要

- 纳入 [具身智能小站 2026-08-28 九篇盘点](../../sources/blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)：动作可以在本体之外用统一几何表示。
- 开源状态（入库日）：**待发布**。
- **单 checkpoint、无榜微调** 是评测协议的核心，不是「我们也报了 LIBERO」。
- 与 [Qwen-RobotManip](./qwen-robot-manip.md) 同属相机系动作：后者用 80 维 canonical + 相机系 EEF delta；UCAG-P 显式加抓取中心锚点，数据量 6374 h（含人手）而非 3 万小时级。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 小米机器人实验室（Xiaomi Embodied Intelligence）；澳门大学（University of Macau） |
| **出处** | arXiv:2608.26058（2026-08），Technical Report |
| **骨干** | Qwen3-VL-4B-Instruct |
| **数据** | 机器人+仿真 4030 h；人类示范 2340 h（合计 6374 h） |
| **真机** | Piper：面包 / 抽屉 / 碗 |
| **开源** | **待发布**：仓为论文图与项目页，README 写 Code Release Soon；无 LICENSE |

### 流程总览

```mermaid
flowchart TB
  data[异构示范 手臂/人形/人手] --> cam[相机系锚点轨迹 p0/p1]
  cam --> share[共享运动头]
  share --> trans[几何条件动作转换器]
  trans --> cmd[本体可执行控制]
```

共享动作：相机坐标系下锚点 \(p_0\)（腕或末端）与 \(p_1\)（抓取中心）的运动。几何条件翻译器按当前本体运动学把该几何增量写成可执行命令。策略在共享几何上学习，部署时只换翻译器，不换视觉–语言骨干。

## 工程实践

| 项 | 内容 |
|----|------|
| **三阶段** | ① 全数据相机中心特化 ② 真值轨迹训转换器 ③ 混合人机联合训练（用策略预测轨迹对齐推理） |
| **锚点** | 腕部 / 末端 + 抓取中心的 2D 图像与 3D 相机系轨迹 |
| **失败源** | 标定、深度、运动学或人手关键点误差会传入共享目标 |
| **跨本体选型** | 先读 [跨本体迁移策略](../queries/cross-embodiment-transfer-strategy.md)：本页是「统一动作空间」路径，不是重定向后重训 |

## 评测

单检查点、**无榜微调**：

| 项 | 内容 |
|----|------|
| **LIBERO** | **98.3%**（四套件） |
| **LIBERO-Plus** | 零样本 **82.0%** |
| **RoboTwin Easy / Hard** | **88.7% / 89.2%** |
| **RoboCasa GR-1** | **62.0%** |
| **Piper 真机** | 面包 / 抽屉 / 碗 **60 / 90 / 75%**（对照 π0.5：20 / 85 / 65%） |

- 数据出处：[ingest 摘录「评测」](../../sources/papers/ucag_p_arxiv_2608_26058.md)。

## 结论

**先共享可观察几何，再翻译执行细节——这比统一关节空间更适合异构数据。LIBERO 98.3% 接近饱和，区分度在厨房 GR-1 62% 与真机面包 60%。**

1. 人手示范能直接监督共享策略，因为目标不在机器人关节里。
2. \(p_1\) 抓取中心才是相对纯 EEF delta 的增量；不要把本文缩成「又一个相机系动作」。
3. 转换器必须看到相机到基座、雅可比与本体状态，否则共享运动落不了地。
4. 无榜微调是协议；拿去和「每榜 SFT」的 VLA 比要先对齐训练预算。
5. Piper 三任务不能外推到双臂/人形；代码未发布，标定/深度误差会放大到控制器。

## 源码运行时序图

**不适用**（截至 **2026-08-28**）：[`Public-BOTs/UCAG-P`](https://github.com/Public-BOTs/UCAG-P) 仅项目页资产，README 写 Code Release Soon。

## 局限与风险

- 项目页 Limitations：标定、深度、运动学、手关键点误差会传播。
- 跨本体接触丰富与关节物体任务仍然难；ALOHA→ARX 零样本会掉到 35%。
- 占位仓、无许可证；clone 只能看到图和项目页。
- 公众号原文「项目页与代码」未给完整 URL，以 arXiv comments 中的项目页为准。

## 与其他工作对比

| 对比轴 | UCAG-P | [Qwen-RobotManip](./qwen-robot-manip.md) | 关节空间 VLA |
|--------|--------|------------------------------------------|--------------|
| 共享量 | 相机系 \(p_0,p_1\) 运动 | 80 维 canonical + 相机系 EEF delta | 各本体关节 |
| 骨干 | Qwen3-VL-4B | Qwen3.5-4B VL + DiT | 不等 |
| 数据 | 6374 h | >38,100 h（含 H2R） | 不等 |
| 开源 | **待发布** | **已开源** 预训练叙事 | 视模型 |

- 相对显式动作重定向 / 人到机器人视频合成 / 数据集分支：UCAG-P 把异构数据对齐到共享几何空间，而不是各训一套头。
- 相对 [Zero-WAM](./paper-zero-wam.md)：一个统一**动作几何**，一个统一**任务规格（视频）**。

## 关联页面

- [VLA](../methods/vla.md)
- [LIBERO](./libero-benchmark.md)
- [Manipulation](../tasks/manipulation.md)
- [跨本体迁移策略](../queries/cross-embodiment-transfer-strategy.md)
- [Qwen-RobotManip](./qwen-robot-manip.md)
- [Qwen-VLA](./qwen-vla.md)
- [WAM / VLA / 跨本体 9 篇技术地图](../overview/wam-vla-cross-embodiment-9-papers-technology-map.md)

## 参考来源

- [ucag_p_arxiv_2608_26058](../../sources/papers/ucag_p_arxiv_2608_26058.md)
- [ucag-p 项目页](../../sources/sites/ucag-p.md)
- [ucag-p 仓库](../../sources/repos/ucag-p.md)
- [具身智能小站 9 篇盘点](../../sources/blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)

## 推荐继续阅读

- [arXiv:2608.26058](https://arxiv.org/abs/2608.26058)
- [UCAG-P 项目页](https://public-bots.github.io/UCAG-P)
- [GitHub 占位仓](https://github.com/Public-BOTs/UCAG-P)
