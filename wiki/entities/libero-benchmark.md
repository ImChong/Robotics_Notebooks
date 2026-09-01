---
type: entity
tags: [sim2real, tooling, deployment, hmi-opensource-table, repo, linux-foundation]
status: draft
updated: 2026-08-28
summary: "LIBERO：用一百三十个机械臂任务控制对象、布局、目标和语言变化，专门评估终身学习与迁移中的分布偏移；固定任务套件和数据接口便于比较策略是记住训练场景还是获得可迁移能力。"
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
  - ../entities/paper-rift-wam.md
  - ../entities/paper-odeworld.md
  - ../queries/hmi-opensource-projects-coverage.md
  - ../concepts/llm-robotics-control-interfaces.md
  - ./anthropic-embody.md
sources:
  - ../../sources/repos/libero-benchmark.md
  - ../../sources/repos/humanoid-motion-intelligence.md
  - ../../sources/papers/world_action_planner_arxiv_2607_27599.md
  - ../../sources/papers/why_action_chunking_improves_bc_corl2026.md
  - ../../sources/papers/neural_introspection_gating_arxiv_2608_10824.md
  - ../../sources/papers/odeworld_arxiv_2607_27924.md
  - ../../sources/sites/anthropic-claude-plays-robotics.md
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
- [Rift](./paper-rift-wam.md) — 免 rollout WAM；LIBERO **98.8%**、LIBERO-Plus **81.1%**（未开源）
- [Why Action Chunking Improves BC](./paper-why-action-chunking-improves-bc.md) — Libero-90 上 Delay / RDE 相对 action chunking 的机制消融
- [GSR / ParaVLA](./paper-gsr-paravla.md) — LIBERO-Para 改写协议；SmolVLA 4.47%→49.12%（arXiv:2608.02497）
- [SLIM-0.5B](./paper-slim-05b.md) — 0.47B latent 策略；LIBERO 97.5% / LIBERO-Plus 77.45%（开源权重）
- [Temporal GRPO](./paper-temporal-grpo.md) — LIBERO-Long 阶段信用探针 99.1%；看 \(\Delta p_k\) 落在哪一段（arXiv:2608.13026）
- [ODEWorld](./paper-odeworld.md) — 连续时间 WM；全量 LIBERO 训视频，LIBERO-LONG 序列子目标 **83.6%**（arXiv:2607.27924）
- [Embody](./anthropic-embody.md) — 用 LIBERO 厨房场景评 **LLM 直接控制 vs 监督 MolmoAct**，不是 VLA SOTA 榜

## 参考来源

- [LIBERO 来源归档](../../sources/repos/libero-benchmark.md)
- [Humanoid Motion Intelligence 仓库归档](../../sources/repos/humanoid-motion-intelligence.md)
- [World Action Planner 论文策展](../../sources/papers/world_action_planner_arxiv_2607_27599.md)
- [Why Action Chunking Improves BC 论文策展](../../sources/papers/why_action_chunking_improves_bc_corl2026.md)
- [开源项目主表（上游）](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

## 推荐继续阅读

- [官方入口](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [Humanoid Motion Intelligence 知识库实体页](./humanoid-motion-intelligence.md)
