# RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies

> 来源归档（ingest）

- **标题：** RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies
- **类型：** paper / benchmark / manipulation / evaluation / vla
- **arXiv abs：** <https://arxiv.org/abs/2604.09860>
- **会议：** Robotics: Science and Systems (RSS) 2026，Sydney
- **项目页：** <https://research.nvidia.com/labs/srl/projects/robolab/>
- **代码：** <https://github.com/NVLabs/RoboLab>
- **Leaderboard：** <https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html>
- **机构：** NVIDIA（Jenai Xuning Yang、Alex Zook、Stan Birchfield、Jonathan Tremblay 等）；多伦多大学（Rishit Dagli）；悉尼大学（Fabio Ramos）
- **入库日期：** 2026-09-14
- **一句话说明：** 高保真 Isaac Lab 仿真评测框架 + RoboLab-120：面向**仅用真机数据训练、零样本进仿真**的通用策略，回答「仿真能否理解真机策略行为」与「哪些环境因子最影响成功率」。

## 核心摘录（面向 wiki 编译）

### 1) 问题：仿真基准饱和与训练–评测重叠

- **摘录要点：** 通用机器人基础模型进展快，但仿真评测易饱和；现有基准与训练数据域重叠大，成功率虚高，难以测**真实泛化**与**鲁棒性**。
- **对 wiki 的映射：**
  - [RoboLab](../../wiki/entities/robolab.md) — 问题设定。
  - [仿真 vs 真机评测 gap](../../wiki/concepts/sim-vs-real-eval-gap.md) — 外推校准语境。

### 2) 框架：场景–任务–环境三阶段生成

- **摘录要点：** 人工或 LLM agent 在仿真中物理摆放物体建场景；为场景加语言指令成任务；再指定机器人、策略、观测/动作配置生成可运行环境。任务库与机器人/策略解耦。
- **对 wiki 的映射：**
  - [RoboLab](../../wiki/entities/robolab.md) — 流程图与 agent 工作流。

### 3) RoboLab-120 与三能力轴

- **摘录要点：** **120** 任务；三轴 — **Visual**（颜色/语义/尺寸）、**Relational**（时序/数量/空间关系）、**Procedural**（affordance/重定向/堆叠）；各轴三难度。平均 **2.02** 子任务/任务、**9.0** 物体/任务。
- **对 wiki 的映射：**
  - [RoboLab](../../wiki/entities/robolab.md) — 任务与能力维表。
  - [RoboDojo](../../wiki/entities/robodojo.md) — 对照另一 sim-and-real 通用操纵榜。

### 4) 与 DROID 分布差异

- **摘录要点：** 相对 DROID 训练分布，更强调多步与两步任务；仅 **68.7%** 基准物体出现在 DROID 训练词表——刻意降低域重叠。
- **对 wiki 的映射：**
  - [RoboLab](../../wiki/entities/robolab.md) — 选型读法。

### 5) 榜单与 sim–real 相关

- **摘录要点：** Leaderboard 上 π0.5 Default 指令约 **28%** SR / **43.4** Score（N 随策略完成度变化）；语言越模糊成功率越低。整体排名与 RoboArena 真机 Elo **Spearman ρ=0.94**。
- **对 wiki 的映射：**
  - [RoboLab](../../wiki/entities/robolab.md) — 评测表。
  - [π0.5](../../wiki/entities/paper-pi05-open-world-vla.md) — 代表 VLA 基线。

### 6) 敏感性分析（NPE）

- **摘录要点：** Neural Posterior Estimation 量化环境参数对任务结果的后验；**腕部相机**扰动对成功率影响显著。
- **对 wiki 的映射：**
  - [RoboLab](../../wiki/entities/robolab.md) — 鲁棒性读法。

### 7) 开源状态（截至 2026-09-14，项目页核查）

- **摘录要点：** **已开源** Apache-2.0：完整评测栈、120 任务、资产、Dashboard、`/robolab-scenegen` 与 `/robolab-taskgen` skills；策略权重依各模型自行发布。
- **对 wiki 的映射：**
  - [robolab 仓库](../repos/robolab.md)
  - [robolab 项目页](../sites/robolab-nvidia.md)
