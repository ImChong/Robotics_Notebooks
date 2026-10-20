# RoboValue: A Fine-Grained Sim-and-Real Benchmark for Unified Evaluation of Robotic Value Models

> 来源归档（ingest）

- **类型：** 论文（arXiv 预印本）
- **机构：** 清华大学、智元机器人、Xspark AI、中山大学、北京大学、香港大学
- **作者：** Shengbang Liu、Zhengye Du、Zhilong Wan、Honghao Su 等
- **论文：** <https://arxiv.org/abs/2610.26663>
- **提交日期：** 2026-10-19
- **项目页：** <https://robovalue-benchmark.github.io/>
- **代码：** <https://github.com/RoboValue-Benchmark/RoboValue>
- **数据：** <https://huggingface.co/datasets/wzlllong/value-model-benchmark-training-data>
- **一句话说明：** 建立覆盖仿真与真实操作任务的细粒度机器人价值模型评测基准，联合检查指令接地、时序进度、失败恢复和价值一致性。

## 核心论文摘录

### RoboValue: A Fine-Grained Sim-and-Real Benchmark for Unified Evaluation of Robotic Value Models（Liu 等，2026）
- **链接：** <https://arxiv.org/abs/2610.26663>
- **核心贡献：** RoboValue 以共享接口评测四类能力：任务状态理解、时序进度监控、失败与恢复推理、价值一致性。数据覆盖 15 项仿真和 20 项真实双臂操作任务；训练集有 3,500 条专家示范，测试集包含 2,792 条轨迹，并分别测试跨本体与跨环境变化。诊断轨迹包括反事实指令、相似画面下不同历史、失败后的恢复结果、以及不同有效子任务顺序。
- **主要发现：** 强的任务结果区分和前向进度相关性不足以证明价值反馈可靠；模型仍会漏掉细粒度指令差异、历史依赖的进度变化、失败恢复成效以及不同解法中的子任务价值一致性。环境变化造成的泛化退化通常比本体变化更广泛。
- **评测读法：** 论文当前报告 Zero-Shot 与 One-Shot 两条排行榜；Full-Shot 是开放的后续评测轨道，不应与已报告结果混为一谈。RoboMeter-4B 在 Zero-Shot 总榜领先；Robo-Dopamine 2.0-8B Preview 在 One-Shot 总榜领先，但四项能力存在明显分化。
- **对 wiki 的映射：**
  - [RoboValue 统一实体页](../../wiki/entities/paper-robovalue-benchmark.md)
  - [过程奖励建模概念页](../../wiki/concepts/progress-reward-modeling.md)

## 当前提炼状态

- [x] 论文问题、数据范围、评测协议与结论摘录
- [x] 项目页源码及数据开放状态核查
- [x] 统一 wiki 实体页与主题入口映射
