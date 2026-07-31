# A Survey on Vision-Language-Action Models for Embodied AI（VLA Survey，HMI P071）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** A Survey on Vision-Language-Action Models for Embodied AI
- **短名：** VLA Survey
- **类型：** paper / hmi-papers / 世界模型、VLA与Agent
- **HMI ID：** P071
- **年份：** 2024
- **原文：** https://arxiv.org/abs/2405.14093
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 系统梳理具身 VLA 的数据、架构、训练与评测维度，便于把「通才策略」主张拆成可比较的技术选择。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P071](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P071.md)

## 开源状态（步骤 2.5）

- **结论：** 综述

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

只要系统同时出现视觉、语言和机器人，就容易被统称为VLA。这篇综述的最大作用是把研究分成三条线：支撑VLA的组件，直接预测低层动作的control policy，以及把长指令拆成子任务的task planner。一个系统可以同时包含后两者，但评估指标、数据和实时性要求完全不同。

**对 wiki 的映射：** [`wiki/entities/paper-vla-survey-embodied.md`](../../wiki/entities/paper-vla-survey-embodied.md)

### 摘录 2

视觉基础表示可以来自CLIP、DINO等预训练，语言编码与多模态对齐给出语义条件；动力学学习、世界模型和RL可以为数据生成、策略预训练或在线规划提供信号；推理和policy steering可以约束动作或生成中间目标。有的方法训练时用世界模型，部署时只保留策略；有的方法每个控制步都用模型规划。因此分类时应标清组件位于训练链还是在线执行链，不能只列模块名字。

**对 wiki 的映射：** [`wiki/entities/paper-vla-survey-embodied.md`](../../wiki/entities/paper-vla-survey-embodied.md)

### 摘录 3

低层VLA可用非Transformer、自回归Transformer或扩散/flow matching生成动作，也可接入3D感知、点位动作或独立action expert。真正需要对齐的是：输出是末端、关节还是技能token；是单步还是action chunk；闭环更新频率和端到端延迟是多少；低层控制器承担了哪些跟踪与安全功能；评测时是否换物体、场景、任务和本体。不指明动作接口的“成功率更高”无法说明身体控制层的优势。

**对 wiki 的映射：** [`wiki/entities/paper-vla-survey-embodied.md`](../../wiki/entities/paper-vla-survey-embodied.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-vla-survey-embodied.md`](../../wiki/entities/paper-vla-survey-embodied.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
