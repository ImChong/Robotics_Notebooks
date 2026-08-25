# PhysCaP（arXiv:2608.21031）

> 来源归档（ingest）

- **标题：** PhysCaP: Grounding Code-as-Policy Agent with Physics-Informed Exploration
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.21031>
  - <https://physcap.github.io/>
- **机构：** 台湾大学（NTU Taiwan）；英伟达研究院（NVIDIA Research）；谷歌 DeepMind；阳明交大（NYCU）等
- **入库日期：** 2026-08-25
- **一句话说明：** 在 Code-as-Policy 上叠加免训练 PhysX 模块（本体感觉估计质量/刚度）与双代理 Planner/Prioritizer，主动探索隐藏物理属性；真机三任务 SR 8–9/10，交互次数与执行时间低于朴素交互基线。

## 核心摘录（MVP）

### 1) 被动 VLA/CaP 的局限

- **摘录要点：** 模仿学习擅长复现示范，但难以推断隐藏物理状态（空罐、成熟度、遮挡物）；朴素加交互易 over-explore。
- **对 wiki 的映射：**
  - [PhysCaP](../../wiki/entities/paper-physcap.md) — 问题定义。
  - [VLA](../../wiki/methods/vla.md) — 对照语境。

### 2) PhysX 模块 + 双代理探索

- **摘录要点：** `get_mass` 用固定抬升轨迹与雅可比力矩差估计质量；`get_stiffness` 用夹爪位移与归一化电机力重复测量+投票；Planner 决定何时探索/停止，Prioritizer 过滤并按 VLM 启发式排序候选交互。
- **对 wiki 的映射：**
  - [PhysCaP](../../wiki/entities/paper-physcap.md) — 架构与流程图。

### 3) 真机与 LIBERO 结果

- **摘录要点：** Find Blue Cube **9/10**、Identify Empty Can **8/10**、Pick Ripe Avocado **9/10**；PhysCaP 成功试验平均探索交互 **1.33–2.5** 次，执行时间 **40–300 s**，优于 CaP 与 CaP+PhysX 变体。
- **对 wiki 的映射：**
  - [PhysCaP](../../wiki/entities/paper-physcap.md) — 评测表。
  - [manipulation](../../wiki/tasks/manipulation.md) — 桌面操作任务。

### 4) 开源状态（截至 2026-08-25）

- **摘录要点：** 项目页 **未列 GitHub** 或代码链 → **确认未开源**（论文方法可复现描述，无官方可运行仓）。
- **对 wiki 的映射：**
  - [physcap 项目页](../sites/physcap-github-io.md) — 步骤 2.5 核查。

## 当前提炼状态

- [x] arXiv + 项目页已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-physcap.md` 新建
