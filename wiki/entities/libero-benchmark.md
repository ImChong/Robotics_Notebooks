---
type: entity
tags:
- sim2real
- tooling
- deployment
- hmi-opensource-table
- repo
- linux-foundation
- paper
- robot-manipulation
- lifelong-learning
- benchmark
- imitation-learning
- dataset
- simulation
status: draft
updated: 2026-10-06
summary: LIBERO：用一百三十个机械臂任务控制对象、布局、目标和语言变化，专门评估终身学习与迁移中的分布偏移；固定任务套件和数据接口便于比较策略是记住训练场景还是获得可迁移能力。
related:
- ../concepts/sim2real.md
- ../entities/isaac-lab.md
- ../entities/humanoid-motion-intelligence.md
- ../entities/paper-world-action-planner.md
- ../entities/paper-why-action-chunking-improves-bc.md
- ../entities/paper-gsr-paravla.md
- ../entities/paper-actfovea.md
- ../entities/paper-neural-introspection-gating.md
- ../entities/paper-flex-pi.md
- ../entities/paper-galaxea-g05.md
- ../entities/paper-reflexvla.md
- ../entities/paper-deicticvla.md
- ../entities/paper-rift-wam.md
- ../entities/paper-odeworld.md
- ../queries/hmi-opensource-projects-coverage.md
- ../concepts/llm-robotics-control-interfaces.md
- ./anthropic-embody.md
- ../tasks/manipulation.md
sources:
- ../../sources/repos/libero-benchmark.md
- ../../sources/repos/humanoid-motion-intelligence.md
- ../../sources/papers/world_action_planner_arxiv_2607_27599.md
- ../../sources/papers/why_action_chunking_improves_bc_corl2026.md
- ../../sources/papers/neural_introspection_gating_arxiv_2608_10824.md
- ../../sources/papers/odeworld_arxiv_2607_27924.md
- ../../sources/sites/anthropic-claude-plays-robotics.md
- ../../sources/repos/libero-plus.md
- ../../sources/repos/libero-pro.md
- ../../sources/papers/rcl_awesome_wam_2306_03310_libero-benchmarking-knowledge-transfer-f.md
project_id: libero-benchmark
arxiv: '2306.03310'
code: https://github.com/Lifelong-Robot-Learning/LIBERO
venue: NeurIPS 2023 Datasets and Benchmarks Track
---

# LIBERO

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md) 的「工程与实机部署」分组，是本库为该入口建立的独立详情节点。

## 一句话定义

用一百三十个机械臂任务控制对象、布局、目标和语言变化，专门评估终身学习与迁移中的分布偏移；固定任务套件和数据接口便于比较策略是记住训练场景还是获得可迁移能力。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LIBERO | LIBERO | LIBERO 相关缩写，详见正文 |
| Sim2Real | Simulation to Real | 仿真到真机部署主线 |
| RL | Reinforcement Learning | 训练与评测常用框架 |
| API | Application Programming Interface | 仿真/中间件编程接口 |

| BC | Behavioral Cloning | 从演示轨迹监督学习策略，论文统一采用的学习范式 |
| ER | Experience Replay | 回放旧任务数据以缓解遗忘的持续学习方法 |
| EWC | Elastic Weight Consolidation | 以参数重要性正则约束旧任务关键权重的持续学习方法 |
| ViT | Vision Transformer | 论文比较的视觉编码器之一 |
| PDDL | Planning Domain Definition Language | 官方数据附带的符号化场景/任务描述格式 |

## 为什么重要

- **主表工程定位清晰**：该条目被放在「工程与实机部署」下，说明它服务的是这条人形运动智能问题链上的具体环节，而不是泛泛的链接收藏。
- **可对照开源边界**：主表已概括其可复现范围（训练/推理/部署或仅方法页）；选型时应先读本页「开源状态」，再回官方 README / 项目页核对许可证与平台支持。
- **便于知识库交叉引用**：独立节点让路线图、对比页与 ingest 日志可以稳定链接，避免只在策展列表里「点名」却无法下钻。

## 核心原理

### 在技术路线中的位置

| 字段 | 内容 |
|------|------|
| 主表分组 | 工程与实机部署 |
| 官方入口 | https://github.com/Lifelong-Robot-Learning/LIBERO |
| 开源状态（据主表） | 已开源（以官方仓库 README 为准） |

主表给出的技术定位可压缩为：

> 用一百三十个机械臂任务控制对象、布局、目标和语言变化，专门评估终身学习与迁移中的分布偏移；固定任务套件和数据接口便于比较策略是记住训练场景还是获得可迁移能力。

阅读时建议抓住三点：**(1) 输入是什么数据或观测；(2) 输出是参考轨迹、策略、数据还是中间件能力；(3) 公开材料能否支撑训练/部署复现。**

### 流程直觉（对照主表叙事）

```mermaid
flowchart LR
  A["上游数据 / 观测 / 配置"] --> B["LIBERO"]
  B --> C["下游策略 / 部署 / 评测"]
```

具体模块边界以官方文档为准；本页不替代 README。

### 扰动增强变体：LIBERO-Plus

**LIBERO-Plus** 是本库多篇论文页共同引用的 **扰动增强套件**：保持 LIBERO 的任务与数据接口，但在 **相机视角、场景布局、语言表述、观测噪声、纹理** 等维度加扰，用来把「记住训练场景」与「获得可迁移能力」的差距**显式量化**。读表时注意三点：

- **它不是一个新任务集，而是同一批任务的扰动条件**——因此 LIBERO 原榜近饱和（多篇报 97–99%）时，LIBERO-Plus 仍能拉开差距（本库已收录的报告值多在 **74–89%** 区间）。
- **分项比均值有信息量**：不同方法的强项落在不同扰动轴上，例如 [LAWA](./paper-lawa.md) 微平均 **74.4%** 但语言扰动弱于 Joint 基线、赢在相机/噪声/纹理；[StellaVLA](./paper-stellavla-structured-icl-vla.md) 零样本 **85.1%** 的主要来源是**视角扰动 +23.5**。
- **跨页数字不可直接横比**：各页的基座、训练数据与评测子集不同（如 [GaussianDream++](./paper-gaussiandream-plusplus.md) **87.8%**、[Kairos](./paper-kairos-native-world-model-stack.md) **89.0**、[Rift](./paper-rift-wam.md) **81.1%**、[Flex-π](./paper-flex-pi.md) **80.9%**、[SLIM-0.5B](./paper-slim-05b.md) **77.45%**），应回各自论文页核对协议后再比较。

具体扰动定义与划分以上游 LIBERO-Plus 发布物为准；本页只做本库交叉引用的锚点。

### 与 LIBERO-PRO 的区别

- **LIBERO-Plus**：系统测试相机视角、物体布局、初态、指令、光照、背景纹理和传感器噪声等扰动，适合按扰动维度分析鲁棒性。
- **LIBERO-PRO**：围绕对象、初始状态、任务指令和环境变化，检查模型是否理解任务并能适应合理变化，而非复现训练轨迹。
- 两者基于 LIBERO 生态，但扰动定义、样本组织和指标协议不同。报告结果时需注明版本与设置，不能将它们视为同一排行榜。

入口：[LIBERO-Plus](../../sources/repos/libero-plus.md)、[LIBERO-PRO](../../sources/repos/libero-pro.md)。

## 工程实践

1. **先核入口类型**：若是 GitHub/Gitee 仓库，从 README 的安装、训练与部署章节入手；若是项目页/论文，先确认是否已挂代码或权重。
2. **对齐本体与接口**：人形项目需核对关节顺序、控制频率、观测契约与仿真后端（Isaac / MuJoCo 等）是否与本机栈一致。
3. **按主表定位做消融**：主表强调的可分拆实验切口（例如只换重定向约束、只换部署层）应优先验证，避免一上来全链路重训。
4. **记录开源边界**：若仅有权重、Sim2Sim 或说明文档，不要假设训练管线可复现。

| 检查项 | 建议 |
|--------|------|
| 许可与星标时效 | 以官方仓库页面为准 |
| 支持机器人 / 仿真 | 读 assets 与 task 配置 |
| 真机入口 | 查找 SDK、ROS、ONNX/JIT 导出说明 |

## 局限与风险

- **主表是策展摘要**：细节、指标与许可以一手来源为准；本页只做知识库节点与导航。
- **开源状态可能变化**：标为待发布的项目后续可能放码；已开源仓库也可能拆分或迁移路径。
- **不要与同名论文页混淆**：若本库另有 `paper-*` 深读页，以论文页承载方法细节，本实体页侧重工程入口与选型。

## 项目资源与工程补充

### 基准组成

论文提出四个任务套件，共 **130 个语言条件操作任务**：

| 套件 | 任务数 | 主要考察的变化 |
|---|---:|---|
| LIBERO-Spatial | 10 | 物体之间的空间关系和摆放布局 |
| LIBERO-Object | 10 | 操作对象类别 |
| LIBERO-Goal | 10 | 任务目标 |
| LIBERO-100 | 100 | 物体、布局与目标等知识的组合迁移 |

LIBERO-100 在基准设置中进一步划分为 **LIBERO-90**（用于预训练）与 **LIBERO-10**（用于下游终身学习评测）。官方数据包括人类遥操作演示；项目说明还列出工作区与腕部 RGB 图像、本体状态、语言任务描述和 PDDL 场景描述等内容。

### 方法与评测设置

论文使用行为克隆（Behavioral Cloning, BC）从演示轨迹学习操作策略，以便在有限计算资源下比较终身学习设定。它研究三种视觉-运动策略架构：

- **ResNet-RNN：** ResNet 编码视觉输入，LSTM 汇总时间信息。
- **ResNet-T：** ResNet 视觉特征与 Transformer 时间骨干结合。
- **ViT-T：** Vision Transformer 处理视觉输入，并以 Transformer 建模时间序列。

比较的学习方案包括顺序微调和多任务学习基线，以及 Experience Replay（ER）、Elastic Weight Consolidation（EWC）和 PackNet 等终身学习方法。论文主要使用任务成功率评估，并研究任务顺序、策略结构、算法选择和预训练对迁移的影响。

### 论文报告的主要发现

1. **架构和算法都影响迁移。** Transformer 时间骨干在抽象时序信息方面表现突出；不同视觉编码器在不同类型的知识迁移上各有强项，没有一种架构对所有套件都最好。
2. **防遗忘不等于更强的前向迁移。** 在论文比较的设定中，ER、EWC、PackNet 等方法能缓解遗忘，但总体上顺序微调的前向迁移表现更好。
3. **任务语言嵌入未必带来提升。** 使用语义丰富的任务描述嵌入，表现并未优于使用任务 ID 嵌入。
4. **朴素监督预训练可能适得其反。** 在大规模离线数据上直接做监督预训练，可能降低后续终身学习表现。

以上结论对应论文的任务、策略和训练协议；复现或横向比较时，应以原文实验设置为准。

### 如何使用

1. 从官方仓库安装环境，查看任务套件、策略配置和评估脚本。
2. 使用官方脚本下载对应套件的遥操作演示数据；README 说明可选择 Hugging Face 下载来源。
3. 选定 suite、策略和终身学习算法，按统一任务顺序及成功率协议评测。
4. 对比 LIBERO-Spatial、Object、Goal 与 LIBERO-90/10 的结果，定位变化来自布局、物体、目标还是它们的组合。

本页不复述完整安装步骤，依赖版本、命令和数据文件结构以[官方 README](https://github.com/Lifelong-Robot-Learning/LIBERO#readme)及[文档](https://lifelong-robot-learning.github.io/LIBERO/)为准。

### 适用范围与限制

- LIBERO 是仿真中的机器人操作基准，适合研究终身模仿学习、知识迁移、任务顺序和策略结构。
- 在 LIBERO 上的结果不能直接等同于真实机械臂上的性能或 sim-to-real 能力。
- 不同 LIBERO 扩展版、任务子集、训练数据和评估协议可能不同；比较分数前先核对具体套件、初始状态、rollout 数和训练设置。
- **重定向就绪度（数据形态适配）：** 官方演示在 robosuite 中以单臂机械臂（Franka Panda）采集，可直接作为同形态 BC 策略的训练输入；换到其他机械臂、双臂或人形平台时需重新采集或做动作空间重定向，不能直接复用。
- 本文是 2023 年提出的基准工作。后续如使用更新的仓库版本或扩展套件，应注明版本，避免将新增设置归到原论文。

## 关联页面

- [sim2real](../concepts/sim2real.md)
- [isaac-lab](../entities/isaac-lab.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [开源主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)
- [World Action Planner](./paper-world-action-planner.md) — LIBERO-Long / Object 上用 pose-image WM + VLM 规划测组合与新布局泛化
- [ActFovea](./paper-actfovea.md) — 在本基准四套件（40 任务 / 2000 episodes）上做 VLA 运行时扰动与防护评测
- [Neural Introspection Gating](./paper-neural-introspection-gating.md) — OpenVLA / OFT 上 logit-margin 门控 KV 缓存；Long/Goal 收回盲缓存掉点（arXiv:2608.10824）
- [BooST](./paper-boost-skill-transfer.md) — DROID 预训练技能迁到本基准；LIBERO-90 10 demo **0.70**（arXiv:2608.10600；训练仓未开）
- [Flex-π](./paper-flex-pi.md) — 多流 WAM；LIBERO 柔性 ckpt 98.5%、固定模式 99.2%；LIBERO-Plus Total 80.9%（arXiv:2608.10860；代码待发布）
- [G0.5](./paper-galaxea-g05.md) — AR VLA；LIBERO 均 **98.9%** / Long **98.6%**（已开源）
- [ReflexVLA](./paper-reflexvla.md) — 动态模块后 LIBERO 仍 **97.2%**（与 VLA-Adapter 持平；代码待开放）
- [DeicticVLA](./paper-deicticvla.md) — Object/Spatial/Goal 子集上 VP-BBox Spatial-ZS 领先；分布内 mean SR **~95%**（arXiv:2608.28108；未开源）
- [Rift](./paper-rift-wam.md) — 免 rollout WAM；LIBERO **98.8%**、LIBERO-Plus **81.1%**（未开源）
- [Why Action Chunking Improves BC](./paper-why-action-chunking-improves-bc.md) — Libero-90 上 Delay / RDE 相对 action chunking 的机制消融
- [GSR / ParaVLA](./paper-gsr-paravla.md) — LIBERO-Para 改写协议；SmolVLA 4.47%→49.12%（arXiv:2608.02497）
- [SLIM-0.5B](./paper-slim-05b.md) — 0.47B latent 策略；LIBERO 97.5% / LIBERO-Plus 77.45%（开源权重）
- [Temporal GRPO](./paper-temporal-grpo.md) — LIBERO-Long 阶段信用探针 99.1%；看 \(\Delta p_k\) 落在哪一段（arXiv:2608.13026）
- [ODEWorld](./paper-odeworld.md) — 连续时间 WM；全量 LIBERO 训视频，LIBERO-LONG 序列子目标 **83.6%**（arXiv:2607.27924）
- [Embody](./anthropic-embody.md) — 用 LIBERO 厨房场景评 **LLM 直接控制 vs 监督 MolmoAct**，不是 VLA SOTA 榜
- [AtomicVLA](./paper-atomicvla.md) — SG-MoE 原子技能 VLA；LIBERO +2.4%、LIBERO-LONG +10% vs π₀（已开源）

- [机器人操作](../tasks/manipulation.md)
- [robosuite 论文（2009.12293）](robosuite.md) — LIBERO 的底层仿真框架
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 评测基准分层选型
- [LIBERO 论文来源归档](../../sources/papers/rcl_awesome_wam_2306_03310_libero-benchmarking-knowledge-transfer-f.md)
- [LIBERO 项目仓库归档](../../sources/repos/libero-benchmark.md)

- [humanoid-motion-intelligence](../entities/humanoid-motion-intelligence.md)
- [paper-world-action-planner](../entities/paper-world-action-planner.md)
- [paper-why-action-chunking-improves-bc](../entities/paper-why-action-chunking-improves-bc.md)
- [paper-gsr-paravla](../entities/paper-gsr-paravla.md)
- [paper-actfovea](../entities/paper-actfovea.md)
- [paper-neural-introspection-gating](../entities/paper-neural-introspection-gating.md)
- [paper-flex-pi](../entities/paper-flex-pi.md)
- [paper-galaxea-g05](../entities/paper-galaxea-g05.md)
- [paper-reflexvla](../entities/paper-reflexvla.md)
- [paper-deicticvla](../entities/paper-deicticvla.md)
- [paper-rift-wam](../entities/paper-rift-wam.md)
- [paper-odeworld](../entities/paper-odeworld.md)
- [llm-robotics-control-interfaces](../concepts/llm-robotics-control-interfaces.md)

## 参考来源

- [LIBERO 来源归档](../../sources/repos/libero-benchmark.md)
- [Humanoid Motion Intelligence 仓库归档](../../sources/repos/humanoid-motion-intelligence.md)
- [World Action Planner 论文策展](../../sources/papers/world_action_planner_arxiv_2607_27599.md)
- [Why Action Chunking Improves BC 论文策展](../../sources/papers/why_action_chunking_improves_bc_corl2026.md)
- [开源项目主表（上游）](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

- [论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf) 与 [arXiv 摘要](https://arxiv.org/abs/2306.03310)
- [官方 GitHub 仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [官方项目页](https://libero-project.github.io/) · [文档](https://lifelong-robot-learning.github.io/LIBERO/)
- [官方数据页](https://libero-project.github.io/datasets) · [Hugging Face 数据集](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)
- [RCL Awesome World-Action Models](https://github.com/rcl-robotics/Awesome-World-Action-Models)：第 031 项，仅作策展索引

- [neural_introspection_gating_arxiv_2608_10824](../../sources/papers/neural_introspection_gating_arxiv_2608_10824.md)

- [odeworld_arxiv_2607_27924](../../sources/papers/odeworld_arxiv_2607_27924.md)

- [anthropic-claude-plays-robotics](../../sources/sites/anthropic-claude-plays-robotics.md)

## 推荐继续阅读

- [官方入口](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [Humanoid Motion Intelligence 知识库实体页](./humanoid-motion-intelligence.md)
