# GPT 6 Astra as an Embodied Policy

> 来源归档（ingest）

- **标题：** GPT 6 Astra as an Embodied Policy
- **类型：** technical report / evaluation
- **原始链接：** <https://anonymous-report-421.github.io/public-website/?view=1>
- **代码：** <https://github.com/anonymous-report-421/eval-of-gpt-6-astra-as-policy>
- **站点源码：** <https://github.com/anonymous-report-421/public-website>
- **作者：** Yu-Mool Shu、Lipxin Zheng
- **年份：** 2026
- **入库日期：** 2026-09-14
- **一句话说明：** 独立双语技术报告：在 RoboDojo 与 RoboLab 各选十任务、每任务 5 实例，系统比较 **GPT 6 Astra Direct** 与 **π0.5 + GPT 6 Astra** 混合闭环——混合在 RoboDojo 上以仅 14.4% GPT 修正步达 48% 成功率，Direct 在 RoboLab 语义抓放子集接近 98%。

## 核心摘录（面向 wiki 编译）

### 1) 问题：通用模型能否当具身策略？

- **摘录要点：** 前沿多模态模型（GPT 6 Astra）具备语义理解与推理，但能否在双臂操纵闭环中稳定产出动作、何时需要专用 VLA 先验，尚缺系统性对照。本文在统一观测–动作接口下比较 **Direct**（纯 EEF 输出）与 **Hybrid**（π0.5 候选 + GPT 审核/修正）。
- **对 wiki 的映射：**
  - [GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md) — 问题设定。
  - [foundation-policy](../../wiki/concepts/foundation-policy.md) — 通用模型作策略的语境。

### 2) 任务选择与协议

- **摘录要点：** 按 π0.5 官方成功率将 0–72% 四等分，从低到高取 **6+2+1+1** 共 **10** 个 RoboDojo 双臂任务；每任务 **5** 次，混合与 Direct **逐实例对齐** task / scene / eval / layout / reset / initial / policy seed。偏向语义分类、顺序记忆、装箱、搭建、柔性操作；不以高精度插接为主。
- **对 wiki 的映射：**
  - [RoboDojo](../../wiki/entities/robodojo.md) — 基准与任务语境。
  - [具身评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — 子集选型读法。

### 3) RoboDojo 主结果（50 实例）

- **摘录要点：** 混合 **24/50（48%）**、平均 Score **62.60**；Direct **13/50（26%）**、Score **37.81**；π0.5 同子集公开参照 **15.67%** / **24.43**。混合架构仅 **14.4%** 实际控制步由 GPT 修正，其余沿用 π0.5；token 约 **624.8M** vs Direct **1.13B**（少约 44.8%）。
- **对 wiki 的映射：**
  - [GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md) — 主评测表。
  - [π0.5](../../wiki/entities/paper-pi05-open-world-vla.md) — 学生先验角色。

### 4) RoboLab 补充结果（50 episode）

- **摘录要点：** 同十任务、语义抓放为主、学生零样本迁移子集：Direct **49/50（98%）**、混合 **46/50（92%）**、π0.5 **18/50（36%）**；对照 Cosmos3-Nano-Policy **36%**、DreamZero **34%**。说明当任务与学生先验匹配时 Direct 可接近饱和，混合略低但仍远高于 π0.5 baseline。
- **对 wiki 的映射：**
  - [GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md) — RoboLab 读法与适用边界。

### 5) 行为机制与失败模式

- **摘录要点：** 轨迹显示 GPT 可纠正目标对齐、规划非抓取接触、在仿真未终止时推测遗漏条件并恢复。混合仍可能反复抓取失败、容器边缘碰撞、段内反馈延迟导致滑落。Direct 会尝试扫瓶入桶、单手搭塔等 zero-shot 方案，但抓取/支撑稳定性弱于混合。
- **对 wiki 的映射：**
  - [GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md) — 定性分析与局限。

### 6) 开源状态（截至 2026-09-14，项目页核查）

- **摘录要点：** **已开源** MIT 评测与报告代码 + 静态站；**未分发** 模型权重、仿真资产、原始会话。复现需自备 GPT 6 Astra 授权、Isaac Sim 5.1、RoboDojo π0.5 checkpoint。
- **对 wiki 的映射：**
  - [eval-of-gpt-6-astra-as-policy 仓库](../repos/eval-of-gpt-6-astra-as-policy.md)
  - [项目站](../sites/gpt-6-astra-embodied-policy-eval.md)
