# RoboBench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain

> 来源归档（ingest）

- **标题：** RoboBench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain
- **类型：** paper / benchmark / mllm / embodied-reasoning / manipulation / evaluation
- **arXiv abs：** <https://arxiv.org/abs/2510.17801>
- **提交日期：** 2025-10
- **会议：** ECCV 2026（已接收）
- **项目页：** <https://robo-bench.github.io>
- **代码：** <https://github.com/yulin-luo/RoboBench>
- **数据集：** <https://huggingface.co/datasets/LeoFan01/RoboBench>
- **评测结果：** <https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results>
- **机构：** 北京大学、北京智源人工智能研究院（BAAI）、复旦大学、北京科技大学、北京人形机器人创新中心等（见论文脚注）
- **作者：** Yulin Luo\*、Chun-Kai Fan\*、Menghang Dong\*、Jiayu Shi\*、Xiangju Mi\*、Mengdi Zhao†、Bo-Wen Zhang\*、Cheng Chi† 等（\* 等贡献，† 项目负责人，✉ Shanghang Zhang）
- **入库日期：** 2026-07-12
- **一句话说明：** 面向 **双系统具身架构中 System 2（embodied brain）** 的 MLLM 综合基准：沿操纵流水线定义 **指令理解、感知推理、泛化规划、affordance 预测、失败分析** 五维、14 能力、25 任务、**6092** QA；规划维引入 **MLLM-as-world-simulator**（DAG 原子动作 + 节点正确性 / 任务完成度）；数据来自大规模真机操纵与自采；官方榜单覆盖 **18** 个 SOTA MLLM，且分数与 **CALVIN / LIBERO-10** 下游 VLA 表现显著相关。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 论文 | <https://arxiv.org/abs/2510.17801> | arXiv 2510.17801 |
| 项目页 | <https://robo-bench.github.io> | ECCV 2026 官方 leaderboard、demo、数据构造与规划评测管线图 |
| 代码 | <https://github.com/yulin-luo/RoboBench> | 评测脚本与数据加载 |
| 数据集 | <https://huggingface.co/datasets/LeoFan01/RoboBench> | 6092 QA 与多模态场景 |
| 结果 | <https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results> | 模型 leaderboard 归档 |

## 核心摘录（面向 wiki 编译）

### 1) 问题动机：执行成功率 ≠ 高层认知

- **摘录要点：** 近年具身系统常采用 **双系统范式**：System 2 负责高层推理，System 1 执行低层控制。本文将 System 2 称为 **embodied brain**，强调其在操纵任务中的认知核心地位。现有基准要么只看 **最终执行成功**，要么在高层推理上 **维度不全、任务不够真实**，无法系统刻画 MLLM 作为 embodied brain 的能力边界。
- **对 wiki 的映射：**
  - [RoboBench](../../wiki/entities/robo-bench.md) — 「为什么需要专门评 System 2」。
  - [VLA](../../wiki/methods/vla.md) — 双系统 VLA 语境下高层 VLM/MLLM 与低层 action head 的分工。

### 2) 五维评测 taxonomy 与规模

- **摘录要点：** RoboBench 沿 **完整操纵流水线** 定义五维：
  1. **Instruction Comprehension**（显式 / 隐式意图）
  2. **Perception Reasoning**（robot-centric、object-centric、scene-centric、task-centric 四视图）
  3. **Generalized Planning**（跨 embodiment / object / view / task 泛化；Q1 长程、Q2 下一步、Q3 子任务状态）
  4. **Affordance Prediction**（静态接触点、动态轨迹、移动底座）
  5. **Failure Analysis**（执行级 vs 规划级错误诊断）
  合计 **14** 能力、**25** 任务类型、**6092** QA（二选一 / 多选 / 多步）。
- **对 wiki 的映射：**
  - [RoboBench](../../wiki/entities/robo-bench.md) — 五维主表与能力树。

### 3) 数据构造：真机 grounding + 人机协同标注

- **摘录要点：** 整合开源真机数据集与自采数据，覆盖 **多样本体、属性丰富物体、多视角场景、记忆驱动导航**。统一流程：**预处理 → 工具辅助 + 人机协同标注 → 统一 schema → 自动生成 QA**。各维专用构造：隐式指令由 LLM 改写显式指令；感知维用 caption/detection/segmentation 起草后人工 refine；规划维从机器人视频建 planning pool；affordance 在关键帧标静态/动态/导航 affordance；失败分析从真机 trial 挖掘执行失败并扰动正确指令合成规划错误。
- **对 wiki 的映射：**
  - [RoboBench](../../wiki/entities/robo-bench.md) — 数据构造管线图。

### 4) MLLM-as-world-simulator 规划评测

- **摘录要点：** 规划维（Q1–Q3）将任务分解为 **参数化原子动作 DAG**，编码因果与时序依赖。**Q1（长程规划）**：用 MLLM 世界模拟器评估 **NodeCorrectness**（动作对齐）与 **TaskCompletion**（关键物体状态是否达成），在视觉与物理约束下 rollout，超越纯文本匹配。**Q2** 比较下一步 skill/object/parameter；**Q3** 判断子任务是否完成。
- **对 wiki 的映射：**
  - [RoboBench](../../wiki/entities/robo-bench.md) — 规划评测框架与 mermaid 流程图。

### 5) 主要发现与 VLM–VLA 一致性

- **摘录要点（ECCV 2026 leaderboard，18 MLLM）：**
  - **闭源领先开源约 20 分**；Gemini-3.1-Pro 在感知（67.32）、affordance（85.70）、失败分析（52.95）最均衡。
  - **隐式指令** 显著难于显式（最强模型 79.94→61.38）；**robot-view** 与 **时序 grounding** 为感知瓶颈。
  - **规划失败** 中 45% 为执行错误（动作序列缺失/错误），24% 识别错误，25% 常识/物理约束错误。
  - **文本-only 基线**（GPT-5.4 无图）在感知（27.12）与 affordance（25.61）接近随机，说明必须视觉 grounding。
  - **执行失败诊断** 极难（最佳约 43.71），规划失败诊断相对容易（最高 80.74）。
  - **VLM–VLA 一致性：** 将开源 VLM backbone 最小微调为 VLA，在 CALVIN / LIBERO-10 评测；RoboBench 感知分与 CALVIN 长程表现强相关（object-centric r=0.884，scene-centric r=0.833）；LIBERO-10 上静态+dynamic affordance r=0.677。
- **对 wiki 的映射：**
  - [RoboBench](../../wiki/entities/robo-bench.md) — 主要发现与下游迁移信号。
  - [ESI-Bench](../../wiki/entities/esi-bench.md) — 对照：ESI 评主动空间探索，RoboBench 评操纵流水线高层认知。

## 当前提炼状态

- [x] arXiv 摘要、项目页五维 taxonomy、数据与规划管线、ECCV 2026 leaderboard 要点已摘录
- [x] 与 `sources/repos/robo-bench.md`（代码/HF 入口）分工明确
- [x] wiki 映射：`wiki/entities/robo-bench.md` 新建，并与 VLA / 同类基准交叉引用
