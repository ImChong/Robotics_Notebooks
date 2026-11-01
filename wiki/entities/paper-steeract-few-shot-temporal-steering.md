---
type: entity
project_id: steeract-few-shot-temporal-steering
tags: [paper, action-model, activation-steering, few-shot-adaptation]
status: complete
updated: 2026-11-01
arxiv: "2610.25632"
project: "https://steeract.github.io"
related:
  - ../entities/unifolm-world-model-action.md
  - ../entities/paper-hint-robot-manipulation.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/steeract_arxiv_2610_25632.md
  - ../../sources/sites/steeract.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "SteerAct 从每任务 24 条带结果标签的 rollout 中选择稀疏特征，以时间描述符在推理时调节冻结动作模型激活；论文摘要报告成功率较未引导模型提高约 10%。"
---

# SteerAct：少样本时间维动作引导

## 一句话定义

SteerAct 用少量成功与失败 rollout 找到区分任务结果的稀疏特征，再按该特征的时间演化引导冻结动作模型的推理激活。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SAE | Sparse Autoencoder | 从预训练表征中提取候选稀疏特征 |
| TA | Temporal Descriptor | 描述特征局部激活顺序分布的时间表征 |
| WAM | World Action Model | 结合世界预测或动作生成能力的机器人模型 |
| π0.5 | pi-zero point five | 论文评测中的预训练机器人动作模型 |

## 为什么重要

部署后的动作模型可能在特定任务上表现不足，但重新训练模型权重代价很高。SteerAct 探索通过小批量结果反馈来调节内部激活，在保持原权重冻结的情况下做任务适配。

## 核心原理

离线阶段以成功与失败轨迹比较时间描述符，从稀疏自编码器提供的候选特征中挑选单一任务特征，并构造参考库。在线执行时，系统比较当前特征的时间描述符与成功/失败参考，判断何时、以多大 steering coefficient 增减激活。

## 评测与结果

论文摘要报告在 π0.5、Fast-WAM、RoboTwin 2.0、DexJoCo 和真实操作任务上的评估；每任务仅需 **24 条**结果标注 rollout，相对未引导模型的任务成功率提升约 **10%**。完整实验表应核实指标范围与统计口径。

## 与其他工作对比

与常规微调相比，SteerAct 保持动作模型权重冻结；与固定层、固定时间的激活编辑相比，它先选出任务相关稀疏特征，再按时间演化动态决定介入时机和强度。

## 工程实践与开源状态

| 项目 | 现状 |
|------|------|
| 离线数据 | 成功与失败结果标签的 rollout |
| 在线调节 | 稀疏特征时间描述符 + steering coefficient |
| 权重更新 | 不更新动作模型权重 |
| 代码 | arXiv 摘要称项目页提供代码和演示；本次未核验到独立 GitHub 仓库链接 |

### 源码运行时序图

**不适用：** 暂未核验到官方仓库或可运行入口，项目页访问后补充。

## 局限与风险

- 方法依赖可比较的成功/失败参考轨迹与稀疏特征质量。
- 每任务 24 条 rollout 的结果不代表所有任务、平台和数据条件均可达到同等增益。
- 推理时调节仍需验证实时性、对观测噪声的敏感度和异常恢复。
- 项目页虽被论文摘要列为代码/演示入口，但仓库链接与许可证仍待确认。

## 结论

**SteerAct 把少样本任务适配转化为推理时的时间维激活引导，避免直接更新预训练动作模型参数。**

- 复现时先确认特征选择和成功/失败参考库是否可稳定构建。
- “24 条”是每任务带结果标签的 rollout 数；“约 10%”应按论文中对应成功率定义解释。
- 不更新模型权重可降低微调成本，但不等于无需额外推理计算或不会产生任务偏差。
- 在确认官方代码与许可证前，把该条目视为论文方法记录，不作为可直接部署的软件包。

## 关联页面

- [UnifoLM 世界模型动作](../entities/unifolm-world-model-action.md) — 动作模型家族与接口
- [HINT 长程操作](../entities/paper-hint-robot-manipulation.md) — 基于部署时意图控制的操作方法

## 参考来源

- [SteerAct 论文摘录](../../sources/papers/steeract_arxiv_2610_25632.md)
- [SteerAct 项目页归档](../../sources/sites/steeract.md)

## 推荐继续阅读

- [SteerAct arXiv](https://arxiv.org/abs/2610.25632)
- [SteerAct 项目页](https://steeract.github.io)
