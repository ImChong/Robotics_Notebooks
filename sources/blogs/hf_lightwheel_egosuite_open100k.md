# EgoSuite-Open100K: 100,000 hours of egocentric human data for Physical AI

> 来源归档（Hugging Face Blog）

- **标题：** EgoSuite-Open100K: 100,000 hours of egocentric human data for Physical AI
- **类型：** blog
- **作者：** Jonathan Stephens（Lightwheel）
- **发布方：** Lightwheel × Hugging Face
- **原始链接：** <https://huggingface.co/blog/LightwheelAI/egosuite-open100k>
- **发布日期：** 2026-08-26
- **入库日期：** 2026-09-14
- **一句话说明：** 光轮智能与 HF 联合发布 **10 万小时** 开放 egocentric 人类活动数据；首批 **1 万小时** 上线，含手/身姿态与事件语义标注，LeRobot v3 + MCAP 双格式，面向 VLA / 世界模型 / 人→机迁移预训练。

## 核心摘录

### 规模一览

| 指标 | 数值 |
|------|------|
| 全量规划 | 100,000 h |
| 已发布 | 10,000 h（分阶段放出余量） |
| 任务 | 15,000+ |
| 采集场景 | 15,000+ |
| 环境大类 | 7（家、酒店、零售、体育、物流、办公、工业） |
| 场景类型 | 128 |
| 任务类别 | 18 |
| 许可用途 | 学术研究 + 商业训练 |

### 采集配置（Sub-SKU）

**EgoStandard**（头戴主视角，约 90k h 规划）：

| Sub-SKU | 规划时长 | 相机 | 姿态 |
|---------|----------|------|------|
| EgoStand | 80,000 h | 头戴 | 手部 |
| EgoStand-Body | 10,000 h | 头戴 | 手部 + 全身 |

**EgoPro**（头戴 + 腕部，约 10k h 规划）：

| Sub-SKU | 规划时长 | 相机 | 姿态 |
|---------|----------|------|------|
| EgoProStandard | 8,000 h | 头戴 + 腕部 | 手部 |
| EgoProStandard-Body | 2,000 h | 头戴 + 腕部 | 手部 + 全身 |

- **EgoDemo**：50 h 小样，覆盖上述四 Sub-SKU + 两种 raw 视频变体。
- LeRobot 与 MCAP 为 **同一 episode 的两种表示**，不可重复计小时。

### 标注类型

1. **手部姿态** — 针对小目标、遮挡、快速运动优化。
2. **身体姿态** — 把手臂动作锚定到任务与环境。
3. **事件级语义** — 部分子集标注「发生了什么」而非仅运动学。

### 典型下游

VLA 预训练、世界模型预训练、人→机行为迁移、egocentric 表征学习、手物交互建模、动作/活动识别、任务与意图理解、长程活动理解、人/手姿态估计、真实操作表征学习。

### 生态与标准

- 与 **EgoVerse** 联盟对齐 egocentric 采集、标注、共享规范。
- 引用 **NVIDIA EgoScale** 与 **Dyna Robotics** 等人视频缩放实验作为开放动机。

## 项目页 / 数据开放核查（步骤 2.5）

| 核查项 | 结论 |
|--------|------|
| **HF 数据** | **已发布** — [Collection](https://huggingface.co/collections/LightwheelAI/egosuite-open100k) |
| **代码** | [LW-Egosuite-DevKit](https://github.com/LightwheelAI/LW-Egosuite-DevKit) **已开源**（MCAP 工具链） |
| **项目页** | [egosuite100k.lightwheel.ai](https://egosuite100k.lightwheel.ai) / 测试域 `egosuite-oepn-100k-test.lightwheel.ai` |

## 对 wiki 的映射

- 主实体：[EgoSuite-Open100K](../../wiki/entities/egosuite-open100k.md)
- 工具：[LW-Egosuite-DevKit](../../wiki/entities/cn-os-lw-egosuite-devkit.md)
- 对照：[Ego4D](../../wiki/entities/paper-ego4d.md)、[EgoScale](../../wiki/methods/egoscale.md)、[EgoVerse](../../wiki/entities/paper-sa-2604-07607-egoverse.md)
