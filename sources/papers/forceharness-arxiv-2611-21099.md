# ForceHarness｜冻结多模态模型的力觉物理接口

- **类型：** paper
- **论文：** https://arxiv.org/abs/2611.21099
- **原文全文：** https://arxiv.org/html/2611.21099v1
- **原文版本：** arXiv v1；核对日期 2026-11-19
- **核对范围：** arXiv v1 全文方法/实验/局限，项目页、README、scripts/run_episode.py 与 forceharness/eval/runner.py；检查数据集入口
- **通读：** 已完成
- **项目：** https://showlab.github.io/ForceHarness/
- **代码：** https://github.com/showlab/ForceHarness
- **wiki：** [唯一论文详情](../../wiki/entities/paper-forceharness-2611-21099.md)

## 核心摘录与知识映射

1. 低层由力/位姿反馈执行 self-terminating force primitives：探测接触、维持力边界、按事件终止，抑制高频负载波动。 → [方法说明](../../wiki/entities/paper-forceharness-2611-21099.md)。
2. 中间层将力和位姿变化估为物体质量、摩擦、刚度、质心等物理属性，并保留不确定性供模型作高层决策。 → [方法说明](../../wiki/entities/paper-forceharness-2611-21099.md)。
3. 上层调用冻结多模态模型选择下一动作与物理试探原语；不做模型梯度微调。 → [方法说明](../../wiki/entities/paper-forceharness-2611-21099.md)。

## 论文实验事实（非本仓复现）

- 作者报告 FR3 真实机器人零样本任务由 2/60 提升到 54/60；每组包含 60 次，需注意任务分布与样本量。
- 论文另给出 HiddenPhys / ForceBench 的模拟环境指标；不能把仿真基准和 FR3 真机 success 混为一谈。

## 论文局限

- 力觉物理量依赖接触模型与传感器标定，不确定性估计并不等于严格安全保证。
- 真机 54/60 的数量级结果受特定操作、模型、原语与 API 预算限制。
- 代码已公开 MIT，提供 HiddenPhys/Fake backend 示例，且多模态模型测试可能产生付费 API 费用。
