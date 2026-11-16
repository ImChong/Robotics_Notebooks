---
type: entity
tags: [benchmark, embodied-ai, mllm, vla, manipulation, evaluation, pku, baai, huggingface, open-source]
status: complete
project_id: robo-bench
arxiv: "2510.17801"
project: https://robo-bench.github.io/
code: https://github.com/yulin-luo/RoboBench
original:
  - ../../sources/papers/robo_bench_arxiv_2510_17801_full_reading_2026_11_15.md
updated: 2026-11-15
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ./esi-bench.md
  - ./paper-daily-omni.md
  - ./ewmbench.md
  - ./paper-vesta-generalist-embodied-reasoning.md
  - ./paper-hrl-stack-34-gr00t-n1.md
  - ../queries/manipulation-vla-architecture-selection.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/papers/robo_bench_arxiv_2510_17801_full_reading_2026_11_15.md
  - ../../sources/papers/robo_bench_arxiv_2510_17801.md
  - ../../sources/repos/robo-bench.md
summary: "RoboBench（arXiv:2510.17801，ECCV 2026）系统评测 MLLM 作为操纵流水线 embodied brain：五维 14 能力 25 任务 6092 QA；规划维用 MLLM-as-world-simulator 检验 embodied 可行性；CALVIN 感知相关显著，LIBERO-10 affordance 仅呈非显著正趋势。"
---

# RoboBench（MLLM 具身大脑综合评测）

**RoboBench**（*A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain*，arXiv:[2510.17801](https://arxiv.org/abs/2510.17801)，[项目页](https://robo-bench.github.io)，[代码](https://github.com/yulin-luo/RoboBench)，[数据](https://huggingface.co/datasets/LeoFan01/RoboBench)，[结果](https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results)，**ECCV 2026**）把评测焦点从「机器人最终能不能做成」推进到 **System 2 是否具备操纵所需的完整高层认知**：在双系统范式下，将 **多模态大语言模型（MLLM）** 当作 **embodied brain**，沿 **理解意图 → 感知场景 → 规划与泛化 → affordance 细化 → 失败诊断** 全流水线出题，并用 **MLLM-as-world-simulator** 检验规划是否能在物理与视觉约束下达成关键物体状态变化。

## 一句话定义

用 **五维 QA + 真机 grounding 数据 + DAG 规划世界模拟器** 量化「MLLM 作为 embodied brain 在操纵任务里究竟会推理到哪一步」，并把分数与 **下游 VLA 控制** 的可迁移性对齐。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MLLM | Multimodal Large Language Model | 多模态大语言模型，本基准主要评测对象 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略；RoboBench 分与下游 VLA 显著相关 |
| VLM | Vision-Language Model | 视觉-语言模型，常作为 VLA 的 System 2 骨干 |
| DAG | Directed Acyclic Graph | 规划评测中原子动作因果依赖图 |
| QA | Question Answering | 本基准以选择题 / 多步问答形式组织 6092 题 |
| ECCV | European Conference on Computer Vision | 论文接收会议（2026） |

## 为什么重要

- **补齐「只评执行成功率」的盲区**：许多机器人基准报告 task success，却难以分离 **高层推理失败** 与 **低层控制失败**；RoboBench 明确评 **System 2**，与 [GR00T N1](./paper-hrl-stack-34-gr00t-n1.md)、[Vesta](./paper-vesta-generalist-embodied-reasoning.md) 等 **VLM + action head** 双系统路线直接对话。
- **覆盖完整操纵认知链**：不像单一 VQA 或规划题集，五维 taxonomy 把 **隐式指令、跨本体/物体/视角泛化、affordance、执行 vs 规划失败诊断** 放在同一坐标系，便于定位模型短板。
- **规划评测超越文本匹配**：**MLLM-as-world-simulator** 在 DAG 上 rollout 原子动作，检查 **NodeCorrectness** 与 **TaskCompletion**，减少「听起来合理但不可执行」的假阳性计划。
- **与下游 VLA 有统计信号**：将开源 VLM backbone 最小微调为 VLA 后，RoboBench **object-centric 感知** 与 **CALVIN** 长程表现 **r=0.884**；**LIBERO-10** 更依赖 **静态+dynamic affordance**（r=0.677）——说明不同控制基准依赖不同认知技能，RoboBench 可作 **VLM 选型诊断**。
- **工程可复现**：代码、HF 数据集与 leaderboard 结果均已发布；ECCV 2026 版覆盖 **18** 个闭源/开源/具身 MLLM（含 GPT-5.4、Claude-Opus-4.7、Gemini-3.1-Pro、Qwen3-VL、RoboBrain-2.5、MiMo-Embodied 等）。**MiMo-V2.6** 为通用 Agent 基座，具身 MLLM 评测仍见 [MiMo-Embodied](https://github.com/XiaomiMiMo/MiMo-Embodied) 与 [MiMo-V2.6](./mimo-v2-6.md) 消歧说明。

## 核心原理与评测结构

### 五维 taxonomy（14 能力 · 25 任务 · 6092 QA）

| 维度 | 测什么 | 代表子能力 / 任务 |
|------|--------|-------------------|
| **Instruction Comprehension** | 显式 vs **隐式** 意图 | Explicit / Implicit goal |
| **Perception Reasoning** | 四视图感知与时空推理 | Robot-type & robot-view；static/functional attribute；spatial relation；temporal grounding；causality；referential comprehension |
| **Generalized Planning** | 跨场景泛化与长程结构 | Cross-embodiment / object / view / task planning；Q1 长程、Q2 下一步、Q3 子任务状态 |
| **Affordance Prediction** | 可执行接触与运动 | Static contact；dynamic trajectory；navigation base position |
| **Failure Analysis** | 失败归因 | Execution-level vs planning-level error diagnosis |

数据来自 **开源真机操纵数据 + 自采**，强调 **多样本体、属性丰富物体、多视角、记忆驱动导航**；经 **工具辅助 + 人机协同** 标注后统一 schema 自动生成 QA。

### 规划评测：MLLM-as-world-simulator

- 每个任务分解为 **参数化原子动作序列**，构成 **DAG**（因果与时序依赖）。
- **Q1 长程规划**：MLLM 模拟器 rollout 动作，评 **动作对齐** 与 **关键物体状态是否达成**（视觉 + 物理约束）。
- **Q2 下一步规划**：比较预测 skill / object / parameter 与参考。
- **Q3 任务状态估计**：二分类判断子任务是否已完成。

### 与相关基准的定位

| 基准 | 主要对象 | 交互形态 | 与 RoboBench 关系 |
|------|----------|----------|-------------------|
| **RoboBench** | MLLM 作为 **embodied brain** | 被动 QA（多图 + 语言） | 操纵流水线 **五维认知** + 规划世界模拟 |
| [ESI-Bench](./esi-bench.md) | MLLM **空间智能** | **主动探索** 闭环（OmniGibson） | 强调遮挡/容纳等 **隐藏空间结构**；少覆盖操纵 affordance 与失败诊断 |
| [Daily-Omni](./paper-daily-omni.md) | OLM/MLLM **跨模态时序对齐** | 被动 **音视频** MCQA | 日常开放域 AV Align；不覆盖操纵 affordance / 失败诊断 |
| [EWMBench](./ewmbench.md) | **具身世界模型** 视频生成 | 开环视频 rollout | 评场景/运动/语义一致性，非 MLLM 高层推理题集 |
| EmbodiedBench / EmbodiedEval | 导航+操作+QA 综合 | 具身环境交互 | 更偏 **端到端 agent**；RoboBench 更细粒度拆 **System 2 认知轴** |

## 流程总览（数据 → QA → 五维评测）

```mermaid
flowchart TB
  subgraph data [真机数据与标注]
    R[开源 + 自采机器人视频/图像]
    P[预处理 + 工具起草 + 人工 refine]
    S[统一 schema]
    R --> P --> S
  end

  subgraph dims [五维 QA 6092 题]
    IC[Instruction Comprehension]
    PR[Perception Reasoning]
    GP[Generalized Planning]
    AP[Affordance Prediction]
    FA[Failure Analysis]
    S --> IC & PR & GP & AP & FA
  end

  subgraph plan [规划维专用]
    DAG[原子动作 DAG]
    SIM[MLLM-as-world-simulator]
    GP --> DAG --> SIM
    SIM --> NC[NodeCorrectness]
    SIM --> TC[TaskCompletion]
  end

  subgraph eval [模型与下游]
    M[18 SOTA MLLM leaderboard]
    VLA[最小微调 VLA → CALVIN / LIBERO-10]
    IC & PR & AP & FA --> M
    M -.相关.- VLA
  end
```

## 源码运行时序图

依据已读 CLI、pipeline 节点与规划评测器；此图只描述公开评测代码，不含真实机器人执行。

```mermaid
sequenceDiagram
  participant User as 用户
  participant CLI as robobench.cli.cmd_pipeline
  participant Exec as PipelineExecutor
  participant Data as RoboBenchDataset / PromptBuilder
  participant Model as AsyncModelClient
  participant Eval as EvaluateNode / PlanningEvaluator
  participant Out as raw.json / evaluated.json
  User->>CLI: --config config.yaml pipeline --repeats N
  CLI->>Exec: 注册 load / build / inference / evaluate 节点
  loop 每次运行、模型与子任务
    Exec->>Data: 读 questions.json 与 system_prompt.json
    Data-->>Exec: 问题、图片路径与 messages
    Exec->>Model: run_batch 调被测模型
    Model-->>Out: 原始回答与 request_id
    Exec->>Eval: 读取回答，按题型选 evaluator
    opt Planning Q1 / Q2 / Q3
      Eval->>Model: 独立 judge、图像和参考约束
      Model-->>Eval: 节点/完成度、下一步或 yes/no 评判
    end
    Eval-->>Out: 分数、覆盖与解析状态
  end
```

## 工程实践

先准备数据与独立的被测模型/judge 配置，运行 `robobench --config config.yaml inspect-data --metadata-only` 检查记录；完整图片下载后去掉 `--metadata-only`。`pipeline --dry-run` 只列节点，不是数据或 API 连通性验证。先选择一个维度、子任务与少量样本，再运行完整评测；本次未调用模型 API。

论文的 25 任务在发布时拆成 32 个 manifest，仍为 **6,092 QA**：MCQ 1,895、Planning 4,197。Affordance 通过候选点、轨迹、底座位置的**选择题**测试，不是直接执行运动。Q1 组合分为 `(NodeCorrectness + TaskCompletion) / 20`，当前实现乘 100；不能解释为机器人闭环成功率。固定 judge、提示词、图像预处理、题型与分组方式，并保留每题原始输出和解析失败数。

当前 Q1/Q2 均值排除解析失败项，Q3 空响应按错计，需同时审计覆盖率。多轮聚合器只读取数字或顶层 `accuracy`，CLI 还使用目录名作聚合键；应逐任务核对，不能将默认聚合当作论文全部总表。`report` 子命令仍为 TODO。详见[原文与实现复核](../../sources/papers/robo_bench_arxiv_2510_17801_full_reading_2026_11_15.md)。

## 实验与评测（ECCV 2026 作者报告）

- **能力鸿沟大但前沿在进步**：Gemini-3.1-Pro 在感知（67.32）、affordance（85.70）、失败分析（52.95）最均衡；多数 MLLM 各维极不均衡。
- **闭源领先开源约 20 分**；同族内随规模/代际稳定提升。
- **隐式指令是共性难点**：最强显式目标模型在隐式指令上平均掉 **~18 分**；CoT 改写 ablation 表明非单纯 prompt 问题。
- **感知瓶颈在 embodiment 与时序**：functional attribute 可达 82+，但 **robot-view**（53.62）与 **temporal grounding**（50.38）最弱。
- **规划「听起来对、执行不了」**：规划失败 45% 为 **执行错误**（动作序列问题），24% 识别错误，25% 常识/物理约束错误。
- **视觉不可或缺**：GPT-5.4 text-only 在感知（27.12）与 affordance（25.61）接近随机，远低于最佳视觉 MLLM。
- **执行失败诊断极难**：最佳约 **43.71**，远低于规划失败诊断（最高 **80.74**）——需要细粒度空间与物理理解。

### 下游相关性的统计边界

迁移实验仅含 **8 个骨干**，全参数微调、单视角 RGB、无本体状态。CALVIN ABC-D 测 1,000 条五子任务序列，指标是平均完成子任务数；LIBERO-Long 测 10 任务各 50 次，指标是成功率。CALVIN 的 object-centric / scene-centric 相关为 **r=0.884, p=0.0035 / r=0.833, p=0.0103**；LIBERO 的 static+dynamic affordance 为 **r=0.677, p=0.0652**，未达到常用 0.05 阈值。相关性不证明因果或必要条件，也不能外推所有 VLM 与控制器。

## 结论

**RoboBench 是高层操纵认知与计划评审的诊断工具；它与真机控制的联系须按骨干、微调协议和统计证据分别验证。**

1. 分开记录 MCQ、Q1 结构化 judge、Q2 下一步与 Q3 状态判断，避免混成一种成功率。
2. 认知分用于筛选假设，控制能力仍须闭环评测；LIBERO 的正趋势不等于显著证据。
3. 冻结 judge 和提示版本，保留解析覆盖率与逐题结果，再核对聚合口径。

## 常见误区或局限

- **不是低层控制或 VLA 端到端基准**：RoboBench 评 **高层 MLLM 认知**；真机成功率仍需 CALVIN、LIBERO、SimplerEnv 等对照。
- **QA 形式 ≠ 在线闭环**：题目以选择题 / 多步问答为主，与 [ESI-Bench](./esi-bench.md) 的逐步探索 API 互补而非替代。
- **规划模拟器依赖 MLLM 能力**：世界模拟器本身有误差上界；DAG 粒度与物理约束抽象需随版本迭代校准。
- **数据与许可证**：HF 数据集与结果卡需单独核对使用条款；复现前以 GitHub README 环境为准。

## 关联页面

- [具身测评纵深路线](../../roadmap/depth-embodied-eval.md) — Stage 1 的认知评测与下游能力外推。

- [VLA（Vision-Language-Action）](../methods/vla.md) — 双系统架构与 RoboBench 所评 System 2 骨干
- [Manipulation（操作任务）](../tasks/manipulation.md) — 任务域与 CALVIN / LIBERO 等下游评测背景
- [ESI-Bench（具身空间智能基准）](./esi-bench.md) — 主动空间探索 vs 操纵流水线认知的互补基准
- [Daily-Omni（日常 AV 时序对齐）](./paper-daily-omni.md) — 音视频跨模态同步诊断 vs 操纵五维认知
- [EWMBench（具身世界模型生成评测）](./ewmbench.md) — 像素域世界模型质量评测对照
- [Vesta（通用具身推理）](./paper-vesta-generalist-embodied-reasoning.md) — 同类「高层 cognition + 低层 actor」分层路线
- [Manipulation VLA 架构选型](../queries/manipulation-vla-architecture-selection.md) — VLM 骨干选型时可参考 RoboBench 维度分数
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页是其「① 具身大脑/MLLM 认知评测层」的代表基准，双向回链
- [RoboDojo](./robodojo.md) — 策略闭环成功率（sim+real）对照；勿与本页 MLLM 认知分混比

## 参考来源

- [完整原文、附录与当前 CLI 复核](../../sources/papers/robo_bench_arxiv_2510_17801_full_reading_2026_11_15.md)

- [RoboBench 论文摘录](../../sources/papers/robo_bench_arxiv_2510_17801.md)
- [RoboBench 仓库与数据归档](../../sources/repos/robo-bench.md)
- Luo et al., *RoboBench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain*, [arXiv:2510.17801](https://arxiv.org/abs/2510.17801)

## 推荐继续阅读

- [RoboBench 项目页](https://robo-bench.github.io) — 交互 leaderboard、demo case、数据与规划管线图
- [yulin-luo/RoboBench（GitHub）](https://github.com/yulin-luo/RoboBench) — 评测脚本与复现说明
- Mees et al., *CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks*, [arXiv:2112.03227](https://arxiv.org/abs/2112.03227) — RoboBench 报告 VLM–VLA 相关性的下游长程操纵基准
