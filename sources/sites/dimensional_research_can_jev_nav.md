# Can Jev Nav?（Dimensional Research · Nav Arena）

> 来源归档

- **标题：** Can Jev Nav?
- **类型：** site / benchmark-report
- **机构：** Dimensional（Dimensional Research）
- **链接：** <https://research.dimensional.org/system-one-navigation>
- **关联代码：** [dimensionalOS/dimos](https://github.com/dimensionalOS/dimos)（评测与 `WorldState` / Habitat 套件；复现分支 `feat/typesafe-world-state`）
- **入库日期：** 2026-09-30
- **一句话说明：** Dimensional 在 **133 个 Habitat 家庭场景、327 项 object-goal 导航任务** 上对比 **Dimcode 导航技能**、**TypeSafe Jev（2 Hz typed drive）** 与 **Pi 系 coding agent（Astra / Fable / GPT-5.6 / Opus）**；统一 **文本 WorldState** 与 **SPL / SoftSPL** 轨迹评分，结论为 **工具化导航仍显著领先纯 LLM/Jev 闭环**，但 Jev 在 **短路径、成本与延迟** 上接近部分 coding agent。

## 核心摘录（MVP）

### 1) 评测规模与问题设定

- **链接：** 项目页 § Intro / Test matrix
- **核心内容：** **327** 导航任务、**133** 环境、**6** 种 driver、**1,962** 次录制 run；任务为 **起点 → 目标物体** 的 object-goal navigation；场景来自 **Habitat / HSSD**（去门后全房间可达），按面积分 small / medium / large，再分 cramped / open。
- **对 wiki 的映射：**
  - [Can Jev Nav? 基准实体](../../wiki/entities/dimensional-can-jev-nav-benchmark.md)
  - [DimOS（Dimensional）](../../wiki/entities/dimensionalos-dimos.md)

### 2) Driver 矩阵（Nav Arena）

- **链接：** 项目页表格
- **核心内容：**
  - **Dimensional + Dimcode：** 带 **navigation & path planning tool calls** 的 harness；典型 **单次 Navigate()**。
  - **TypeSafe Jev：** `TypeSafeAgent`，**无 tools**，2 Hz；输出 **drive.x/y/yaw、stop、task、target** typed choices → `Twist`。
  - **Astra / Fable / GPT-5.6 / Opus + Pi：** **无 dimOS tools**，仅 Zenoh 三 topic（`world_state` / `cmd_vel` / `finished`）；自写 Python zenoh 客户端。
- **对 wiki 的映射：**
  - [Jev（TypeSafe）](../../wiki/entities/typesafe-jev.md)
  - [Can Jev Nav? 基准实体](../../wiki/entities/dimensional-can-jev-nav-benchmark.md)

### 3) WorldState 与 Jev 输入格式敏感性

- **链接：** 项目页 § Perception / Jev needs language
- **核心内容：** 全体 agent 仅收 **语言/JSON 字符串**（对齐 Jev 文本接口）；**robot-frame + 自然语言 helper**（`ahead_left`、`blocked`、`near` 等）相对 **world-frame 数值坐标** 将 Jev 完成率从 **40% → 90%**（84 任务子集）。
- **对 wiki 的映射：**
  - [Can Jev Nav? 基准实体](../../wiki/entities/dimensional-can-jev-nav-benchmark.md)

### 4) 总体结果（Mean SPL / Arrived）

- **链接：** 项目页 § Results at a glance
- **核心内容（327 任务）：**

| Driver | Mean SPL | Arrived |
|--------|----------|---------|
| Dimensional Dimcode | 0.743 | 88.4% |
| Astra (Pi) | 0.522 | 76.1% |
| Fable 5.1 (Pi) | 0.330 | 50.8% |
| TypeSafe Jev | 0.263 | 45.9% |
| GPT-5.6 (Pi) | 0.213 | 32.1% |
| Opus 4.7 (Pi) | 0.125 | 18.7% |

- **对 wiki 的映射：**
  - [Can Jev Nav? 基准实体](../../wiki/entities/dimensional-can-jev-nav-benchmark.md)
  - [Jev](../../wiki/entities/typesafe-jev.md)

### 5) 复现命令

- **链接：** 项目页 § Reproduction
- **核心内容：**
  ```bash
  git clone https://github.com/dimensionalOS/dimos.git && cd dimos && git checkout feat/typesafe-world-state
  dimos evals run dimos.evals.suites.habitat_nav --agent dimos.evals.agents.topic --set 'modules=["type-safe-agent"]' --trace TypeSafeAgent --case 102343992_chair
  ```
- **对 wiki 的映射：**
  - [DimOS 仓库归档](../repos/dimensionalos_dimos.md)

## 参考链接

- 报告：<https://research.dimensional.org/system-one-navigation>
- dimOS Navigation 文档：<https://github.com/dimensionalOS/dimos/blob/main/docs/capabilities/navigation/index.md>
- BibTeX：页面 § Cite this work
