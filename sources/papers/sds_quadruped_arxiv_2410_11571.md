# SDS：See it, Do it, Sorted — 四足单视频技能合成（arXiv:2410.11571）

> 论文来源归档（ingest；作为 E-SDS 方法前身的交叉引用资料）

- **标题：** SDS – See it, Do it, Sorted: Quadruped Skill Synthesis from Single Video Demonstration
- **类型：** paper / quadruped locomotion / VLM reward synthesis / imitation-from-video
- **arXiv：** <https://arxiv.org/abs/2410.11571>（HTML：<https://arxiv.org/html/2410.11571v2>）
- **PDF：** <https://arxiv.org/pdf/2410.11571.pdf>
- **机构：** Robot Perception Lab, UCL（Maria Stamatopoulou, Jeffrey Li, Dimitrios Kanoulas）
- **项目页：** <https://rpl-cs-ucl.github.io/SDSweb/>
- **代码：** <https://github.com/RPL-CS-UCL/SDS>（详见 [`sources/repos/rpl_cs_ucl_sds.md`](../repos/rpl_cs_ucl_sds.md)）
- **入库日期：** 2026-05-17
- **一句话说明：** 用 **GPT-4o** 从 **单条演示视频** 经 **网格帧编码 G_v** 与 **SUS 多智能体分解** 生成 **可执行 Python 奖励字典**，驱动 **IsaacGym PPO** 并在 **姿态网格 + 接触序列 + 训练日志** 闭环下 **迭代进化奖励**；四足多种步态 **sim+实机** 报告高步态匹配，但 **奖励生成不条件于地形传感器统计**（与 E-SDS 的感知缺口对照）。

## 核心摘录（与 E-SDS 的接口）

### 1) G_v 与 SUS

- **要点：** 自适应采样帧拼成 **√n×√n** 图像网格；SUS 链式分解 **任务描述 → 步态/接触分析 → 任务需求 → 汇总 prompt**；可选 **ViTPose++** 关键点减歧义。
- **对 wiki 的映射：** [`wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md`](../../wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md)（E-SDS 显式继承该 prompting 栈并加环境分支）

### 2) 训练—评估—进化闭环

- **要点：** 每轮生成多候选奖励、并行 PPO、用 **rollout 网格图 G_s + 接触图 CP** 交 VLM 打分选优、再带 **子奖励标量日志** 反馈进化下一轮（E-SDS 在人形上改为 **双候选 + 固定 3 轮 + 量化地形统计条件** 等变体，见 E-SDS 源文件）。
- **对 wiki 的映射：** [`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md)

## 当前提炼状态

- [x] 与 E-SDS 对齐的要点摘录
- [x] wiki 映射（见 E-SDS 实体页）
