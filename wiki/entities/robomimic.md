---
type: entity
project_id: robomimic
project: https://robomimic.github.io/
code: https://github.com/ARISE-Initiative/robomimic
arxiv: "2108.03298"
tags: [robot-learning, imitation-learning, offline-rl, manipulation, dataset, stanford, ut-austin]
status: complete
updated: 2026-10-17
summary: "robomimic 是机器人示范学习与离线强化学习框架，提供统一的数据集格式、算法实现、训练评测工具和可复现实验；原始 CoRL 研究揭示示范质量、历史信息、观测选择与策略评估对结果的影响。"
related:
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../comparisons/online-vs-offline-rl.md
  - ../concepts/sim2real.md
  - ./mimicgen.md
sources:
  - ../../sources/repos/robomimic.md
  - ../../sources/sites/robomimic.md
  - ../../sources/papers/robomimic_arxiv_2108_03298.md
---

# robomimic

robomimic 是 ARISE Initiative 发布的机器人示范学习研究框架。它把数据集、离线模仿学习与强化学习算法、训练配置和评估工具组织在统一接口下，方便在同一批机器人操作示范上比较方法。项目不是某一种策略算法，也不是即插即用的人形机器人部署系统。

## 一句话定义

一个围绕机器人操作离线示范数据构建的模块化研究工具箱：用统一 HDF5 数据约定、算法接口和评估流程，比较行为克隆及离线强化学习策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BC | Behavioral Cloning | 直接从观测回归示范动作 |
| RNN | Recurrent Neural Network | 用历史信息建模连续决策 |
| HBC | Hierarchical Behavioral Cloning | 高层预测子目标、低层生成动作序列 |
| RL | Reinforcement Learning | 从奖励信号优化策略的学习范式 |
| HDF5 | Hierarchical Data Format 5 | 官方数据集采用的分层文件格式 |
| IRIS | Implicit Reinforcement without Interaction at Scale | 论文比较的离线强化学习算法之一 |

## 为什么重要

- **比较更可复现：** 同一数据格式和训练管线让研究者能把算法差异与数据差异分开检查，而不是每篇工作各自维护不兼容的脚本。
- **数据本身进入研究问题：** 原始 CoRL 研究不仅比较算法，还系统改变示范熟练度、数据规模、状态/图像观测和仿真/真实任务。
- **对模仿学习工程有直接启发：** 人类示范会受到未观测的操控器状态和动作历史影响；把策略做成只看当前帧的回归器，可能漏掉决策上下文。
- **边界明确：** 框架强调可复现实验与算法比较；迁移到新机器人仍需实现环境适配、观测与动作映射及控制接口。

## 核心原理

### 统一的数据与训练接口

官方兼容数据通常存为单个 HDF5 文件，包含多个轨迹组。每条轨迹可记录观测 obs、下一时刻观测 next_obs、动作 actions、奖励 rewards、终止信号 dones，并携带环境配置等元数据。图像按通道后置格式保存；官方约定动作通常归一化到 [-1, 1]。数据集可通过 mask/filter key 对轨迹切分，例如训练集与验证集。

一个常见实验路径是：

演示数据 → HDF5 预处理 / 观测抽取 → 配置算法与观测模态 → 训练策略 → 环境回放评估

```mermaid
flowchart LR
  A["示范轨迹 / 环境记录"] --> B["HDF5 数据与观测处理"]
  B --> C["算法与训练配置"]
  C --> D["策略检查点"]
  D --> E["仿真或真实环境回放评估"]
```

### 算法族与研究论文

原始研究比较六种离线方法：

| 类别 | 方法 | 核心区别 |
|------|------|----------|
| 模仿学习 | BC | 当前观测到动作的直接监督回归 |
| 模仿学习 | BC-RNN | 利用循环网络建模动作决策中的时间相关性 |
| 模仿学习 | HBC | 高层预测未来观测子目标，低层条件生成动作序列 |
| 离线 RL | BCQ | 约束动作接近数据集支持范围 |
| 离线 RL | CQL | 通过保守价值估计抑制分布外动作高估 |
| 离线 RL | IRIS | 论文纳入比较的无交互规模化离线强化学习算法 |

robomimic 后续版本继续扩展算法与功能。官方 v0.5 文档包含 IQL、TD3-BC、Diffusion Policy 等实现，并提供 Transformer、Diffusion Policy、多数据集与语言条件训练教程。复现实验时应记录使用的 robomimic 与 robosuite 版本、配置、数据版本和观测处理方式，不能把论文的旧环境结果直接视作新版逐项相同。

### 数据集范围

CoRL 研究使用 Lift、Can、Square、Tool Hang、Transport 等多阶段操作任务，包含仿真与真机版本。数据覆盖熟练操作者、多操作者混合熟练度以及机器生成策略轨迹；实际任务和数据变体因数据集而异。官方 v0.1 文档同时指出：当前新版 robosuite 处理的数据与原始 offline_study 分支数据并不保证逐项复现完全一致。

## 实验与评测

原始研究页面总结的经验包括：

1. **历史依赖策略表现突出：** BC-RNN 与 HBC 在人类数据上通常优于不建模决策历史的基线。
2. **人工数据质量混杂会挑战离线 RL：** 从机器生成数据中处理好坏差异的能力，不会自动迁移到多人类示范。
3. **验证损失不能代替策略回放：** 训练目标和任务成功率不同；检查点选择最好结合环境执行评估。
4. **观测设计会显著影响表现：** 例如腕部相机观察对操作任务可能很关键；图像视角和超参数应作为实验变量记录。
5. **仿真结论有迁移价值但不覆盖所有真机难点：** 论文的真实 Tool Hang 结果明显低于仿真，说明仿真任务成功不等于真实任务已解决。

## 工程实践

- **先确认数据版本：** 论文复现优先按相应文档选择数据与 robosuite 分支；新版数据可能采用不同 MuJoCo Python 绑定和环境实现。
- **固定轨迹划分与评估协议：** 用 filter key 明确训练/验证演示，另做独立环境 rollout；不要只报训练或验证损失。
- **先建立低维基线，再加图像：** 这样可拆分数据管线、策略结构和视觉编码带来的影响。
- **保留动作与观测契约：** 新环境包装层需明确关节/夹爪动作定义、动作归一化、观测字段、时间步和终止条件。
- **把版本和配置归档：** 数据版本、算法配置、随机种子、相机视角和环境版本都会改变结果。

| 项目 | 官方入口 |
|------|----------|
| 源码与训练示例 | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) |
| 安装、算法和教程 | [robomimic 文档](https://robomimic.github.io/docs/) |
| CoRL 数据集说明 | [数据与复现实验](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html) |
| 数据集 | [robomimic Hugging Face](https://huggingface.co/datasets/robomimic/robomimic_datasets) |

## 局限与风险

- **任务范围以操作为主：** 原始基准是机械臂操作，不能据此推断其包含人形全身运动数据或人形控制器。
- **离线数据覆盖限制策略：** 策略主要从固定数据分布中学习；未覆盖的状态和动作会带来分布偏移风险。
- **数据质量和采集过程不可忽略：** 操作者差异、失败样本、遥操作器和历史动作都可能影响学习结果。
- **训练验证与实机成功率不等价：** 需要以目标环境 rollout 检查实际行为，真实部署还需单独处理安全、延迟和硬件接口。
- **版本之间结果不保证一致：** 新版框架、仿真器和数据处理更新后，应按具体版本重新核验基线。

## 关联页面

- [模仿学习](../methods/imitation-learning.md)
- [强化学习](../methods/reinforcement-learning.md)
- [在线与离线 RL 对比](../comparisons/online-vs-offline-rl.md)
- [Sim2Real](../concepts/sim2real.md)
- [MimicGen](./mimicgen.md)

## 参考来源

- [robomimic 官方代码仓库归档](../../sources/repos/robomimic.md)
- [robomimic 官方项目与研究页归档](../../sources/sites/robomimic.md)
- [论文来源归档](../../sources/papers/robomimic_arxiv_2108_03298.md)
- [CoRL 2022 正式论文（PMLR）](https://proceedings.mlr.press/v164/mandlekar22a.html)
- [官方 v0.1 数据与复现文档](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)

## 推荐继续阅读

- [官方研究页：What Matters in Learning from Offline Human Demonstrations for Robot Manipulation](https://robomimic.github.io/study/)
- [模仿学习](../methods/imitation-learning.md)
- [在线与离线 RL 对比](../comparisons/online-vs-offline-rl.md)
