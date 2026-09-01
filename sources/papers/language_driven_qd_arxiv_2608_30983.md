# Language-Driven Quality-Diversity（语言驱动 QD 技能档案）

> 来源归档（ingest）

- **标题：** Autonomously Acquiring Robot Manipulation Skills with Language-Driven Quality-Diversity
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.30983>
- **作者：** Émiland Garrabé, Mahdi Khoramshahi, Stéphane Doncieux
- **代码：** <https://github.com/EGarrabe/Language-driven-robotic-QD>
- **入库日期：** 2026-09-01
- **一句话说明：** 仅需自由形式任务语言，自动探索 fitness 与行为描述符（BD）函数空间，结合 multi-BD MAP-Elites success 生成多样运动原语档案；Genesis 四操作任务优于经典 QD。

## 核心摘录（MVP）

### 1) 问题：QD 需专家写度量，LLM 奖励塑形只出单解

- **摘录要点：** QD 可构建多样运动原语库以零样本适应部署约束，但通常依赖专家手写成功条件、适应度与多样性度量；LLM 奖励塑形虽能自主学习，却多产出单一高性能解。
- **对 wiki 的映射：**
  - [Language-driven QD](../../wiki/entities/paper-language-driven-robotic-qd.md)

### 2) 方法：语言驱动函数空间探索 + multi-BD MES

- **摘录要点：** 将策略探索表述为函数设计问题，用 LLM 在无任务特定 prompt/微调下采样低维 fitness/BD 函数空间；适配 multi-BD MAP-Elites success（MES）利用异构 BD 样本。
- **对 wiki 的映射：**
  - [Language-driven QD](../../wiki/entities/paper-language-driven-robotic-qd.md)

### 3) 实验与开源

- **摘录要点：** Genesis 仿真四操作任务；优于推断或手写参数化的经典 QD。**已开源** `EGarrabe/Language-driven-robotic-QD`。
- **对 wiki 的映射：**
  - [Language-driven-robotic-QD 仓库](../repos/egarrabe-language-driven-robotic-qd.md)

## 当前提炼状态

- [x] arXiv 摘要对齐
- [x] 仓库开源核查
- [x] wiki 映射：`wiki/entities/paper-language-driven-robotic-qd.md` 新建
