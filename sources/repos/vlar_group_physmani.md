# vLAR-group/PhysMani

> 来源归档

- **标题：** PhysMani（官方仓库 / 项目页）
- **类型：** repo
- **组织：** vLAR Group（香港理工大学 PolyU）
- **代码：** <https://github.com/vLAR-group/PhysMani>
- **论文：** <https://arxiv.org/abs/2607.01938>（PDF：<https://arxiv.org/pdf/2607.01938v1>）
- **venue：** ECCV 2026
- **许可证：** CC BY-NC-SA 4.0
- **入库日期：** 2026-07-14
- **一句话说明：** **PhysMani** 官方 GitHub 入口：physics-principled **3D Gaussian 世界模型** + future-aware **3DFA 策略** 的动态操作框架；含 **PhysMani-Bench（16 任务）** 说明。**当前 release 仅为 landing page**，代码、数据与预训练权重尚未公开。

## Release Status（截至 2026-07-14）

| 组件 | 状态 |
|------|------|
| README / 引用 / License | 已公开 |
| 训练与推理代码 | **未发布** |
| PhysMani-Bench 数据 | **未发布** |
| 预训练模型 | **未发布** |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [PhysMani（动态操作 3D 世界模型）](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md) | 实体归纳页：双模块架构、Benchmark、仿真/真机结果 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | **3DGS 速度场 WM** 相对 **2D 视频扩散 WM** 的机器人控制向样本 |
| [Manipulation](../../wiki/tasks/manipulation.md) | **动态目标**（传送带、旋转架等）操作评测语境 |
| [Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md) | 同为 **世界模型 + 动作策略** 联合；Kairos 走 **视频 DiT WAM**，PhysMani 走 **3D Gaussian 物理速度场 + 3DFA IL** |

## 对 wiki 的映射

- 技术报告摘录：[`sources/papers/physmani_arxiv_2607_01938.md`](../papers/physmani_arxiv_2607_01938.md)
- 沉淀 **[`wiki/entities/paper-physmani-dynamic-manipulation-world-model.md`](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md)**
