# Hyperagents｜可修改元改进机制的自指智能体

> 一手论文归档（ingest）｜arXiv:2603.19461v2

- **标题：** Hyperagents（超智能体：将任务执行与改进机制共同编码为可编辑程序）
- **作者：** Jenny Zhang、Bingchen Zhao、Wannan Yang、Jakob Foerster、Jeff Clune、Minqi Jiang、Sam Devlin、Tatiana Shavrina
- **类型：** paper / recursive self-improvement / metacognitive self-modification / agent / robot reward design
- **论文原文：** <https://arxiv.org/pdf/2603.19461v2>（[arXiv v2](https://arxiv.org/abs/2603.19461v2)，2026-11-01 修订）
- **官方实现：** <https://github.com/facebookresearch/Hyperagents>
- **入库日期：** 2026-11-16
- **核对范围：** 阅读 v2 论文正文的问题设定、DGM-H 方法、四领域实验与基线、元层能力及迁移、讨论与局限；核对理解核心机制所需附录中 Genesis Go2 奖励任务设置、实验协议、安全及实现说明。未逐条复现实验或执行完整 GPU 训练。
- **通读：** 已完成
- **对 wiki 的映射：** [Hyperagents 论文独立节点](../../wiki/entities/paper-hyperagents.md)
- **代码档案：** [官方 Hyperagents 仓库入口与调用链](../repos/hyperagents.md)

## 研究问题与贡献

既有 Darwin Gödel Machine（DGM）用编码 Agent 自修改 Agent、再用任务基准检验生成的子代。对编程任务，编程能力与继续改写自己的能力有一定重合；但在论文评估、机器人奖励设计等其他任务上，两者并不自然等价。

**Hyperagents / DGM-H 的关键区别**：把执行任务的 `TaskAgent` 与产生修改的 `MetaAgent` 放入同一可修改程序。每一代不只能修改任务 Agent 的提示、工具和代码，也能修改自己将来**如何提出修改**的元层程序。例如保留长期实验记忆、记录各代性能、改变探索方式。外部基准仍提供评价与筛选信号，且本研究**没有更新基础模型参数**。

## 机制摘录（方法与图 1）

1. **初始 Hyperagent：** 同一代码库包含 task agent 与 meta agent，模型调用由外部 FM API 驱动。
2. **亲本档案：** 以既有性能和搜索分支信息选择亲本，从父节点重放可继承的代码补丁；并不只保留当前分数最高的版本。
3. **元层自修改：** MetaAgent 可以修改自身与 TaskAgent 的可编辑实现，留下代码 diff；研究框架将用于任务评测的 `domains/` 目录复原，避免评估器直接被改。
4. **经验评测：** 在隔离 Docker 环境执行候选，先做阶段评测，符合条件再扩大评测；外层收集分数、版本关系、日志。
5. **持续迭代：** 用档案与历史评测作为后续元 Agent 的上下文，使搜索过程的经验可积累。

**边界：** 主实验外层的任务定义、评测指标与亲本调度主要固定；因此“改进改进者”的案例**不是**所有层都可自由改写或已证明无限自加速的证据。

## 实验设置与论文主结果（作者报告）

| 领域 | 优化对象 / 测试 | 论文所报结果及解释 |
|---|---|---|
| Polyglot 编码 | 改写任务 Agent 并跨全量任务评估 | 初始 `0.084` → DGM-H `0.267`；与原 DGM 的起点/设置不应混同 |
| 学术论文审阅 | 生成审阅/判断，评价代理任务表现 | 初始 `0.000` → DGM-H `0.710` |
| **Genesis / Unitree Go2** | 通过代码生成奖励函数，再训练并评估仿真强化学习策略；测试任务为跳跃/抬身目标 | **初始 0.060 → DGM-H 0.372**；同实验的定制 DGM 为 0.348，论文未证实两者差异显著 |
| 奥赛数学判分 | 跨领域重用此前进化过的 meta agent，比较迁移与从头进化 | 支持元层经验迁移，但不等于跨机器人环境验证 |

### Go2 奖励设计协议（Genesis）

- Task Agent 获得任务说明、Genesis 环境源码路径和默认奖励函数，输出可执行 Python 奖励代码。
- 生成奖励传入 Genesis/RSL-RL 训练入口，经过 PPO 训练后，在仿真环境计算 fitness；论文讨论 Go2 正向速度跟踪与未见的跳跃/躯干高度任务。
- 论文的“机器人结果”是**仿真奖励程序设计 + RL 训练**的评估，不是一个 DGM-H 直接实时输出电机力矩，也不是已在 Go2 真机部署。
- 公开仓库的 `domains/genesis/README.md` 给出 20 s 仿真评价、4 s episode、4096 并行环境等实施条件；实际算力和实验成本需另估。

### 元层自改与迁移

论文通过查看演化出的代码与跨领域实验，讨论**持久记忆、性能追踪**等改善“未来如何改”的机制。跨领域测试使用冻结的已有 meta agent 与重新执行搜索的不同对照；不能把单一任务收益误读为“任何计算任务都已得到自加速”。

## 与相关研究区分

- **DGM**：进化 task/coding agent，本身的改进过程主要由手工固定；Hyperagents 在同一个可编辑程序里也允许进化 meta agent。
- **SelfImprove-Bench**：提出测试自改结果能否跨实例/难度/任务泛化的评测协议；该基准将 Hyperagents 列为一种 RSI 受测方法，不能用其结果替代本文自己的实验协议。
- **机器人控制**：生成奖励函数用于训练策略是“机器人研发自动化”的证据，不代表机器人策略在线自修改、物理安全或真机泛化已得到证明。

## 局限、实现与复现边界

- 模型权重冻结，依赖外部 API、Docker、评测器和多轮高成本训练；增益与基准选择、模型、随机种子有关。
- 因可修改代码会在运行环境中执行，官方 README 明示存在**破坏性行为**风险；必须在隔离环境中审查，外部安全门不应交给被进化的代码修改。
- 本次核对了论文 v2 与仓库静态入口，但**未独立运行模型 API、Genesis PPO 或全量演化实验**。
- 官方许可文件为 **CC BY-NC-SA 4.0（非商业、相同方式共享）**，不能假设商业使用与宽松 MIT 等价。

## 参考来源

- [Hyperagents：arXiv:2603.19461v2 原始论文](https://arxiv.org/pdf/2603.19461v2)
- [facebookresearch/Hyperagents README 与源码](https://github.com/facebookresearch/Hyperagents)
- [代码归档（可运行入口、评测器、许可）](../repos/hyperagents.md)
- [DGM 对照的已有论文节点](../../wiki/entities/paper-darwin-godel-machine.md)
