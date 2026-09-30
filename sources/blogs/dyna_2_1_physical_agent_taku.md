# Dyna-2.1: A Physical Agent for End-to-End Workflows

> 来源归档（blog / research post · Dyna Robotics 官方）

- **标题：** Dyna-2.1: A Physical Agent for End-to-End Workflows
- **类型：** blog / research technical report（公司站，非 arXiv）
- **作者 / 组织：** Dyna Robotics
- **原始链接：** <https://www.dyna.co/dyna-2.1>
- **发表日期：** 2026-09-29
- **入库日期：** 2026-09-30
- **抓取方式：** 官方页 WebFetch（`www.dyna.co/dyna-2.1`）
- **一句话说明：** **Dyna-2.1** 是 Dyna 首个宣称可 **自主完成约一小时、非线性 loco-dexterous 工作流** 的 physical agent：硬件为轮式半人形 **Taku**，软件为 **全身 RL 控制器 + DYNA-2 WAM + VLM 编排器**；洗衣房为 running example（洗烘机操作、毛巾流、折叠上架、进度记忆与可恢复错误）。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-09-30） |
|----|-------------------------|
| 研究入口 | <https://www.dyna.co/dyna-2.1> |
| 代码 / 权重 | **确认未开源** — 未见 GitHub / Hugging Face |
| arXiv | **未见** |
| 可信度边界 | 产业官方长文 + 演示视频；定量多为内部工作流与 teachability 叙事 |

## 核心摘录（归纳，非全文）

### 产品主张：从 task 到 workflow

- 静止任务（如 DYNA-1 折餐巾 24h）仍要人补料/清栈 → **工作流** 才对应「整班员工」。
- 工作流需要：**全角色工作空间覆盖**（底座+躯干+双臂）、**可教性**（客户流程差异大）、**非线性推理**（机器异步完成、可中断折叠去卸烘等）。

### 硬件：Taku

| 要素 | 要点 |
|------|------|
| 形态 | 腰上类人、下体可折叠、**四轮转向** 底座、**双 7-DoF** 臂 |
| 动机 | 洗衣等 vertical 以 reach/精度为主，轮式换站；尺寸对齐平均成人以便人轨迹重定向 |
| 执行器 | 低减速比行星关节 → 更高臂速/加速度（相对谐波臂） |

### 三层模型栈（Figure 2.4）

| 层 | 角色 | 频率/时钟 |
|----|------|-----------|
| Whole-body controller | 仿真 RL；URR 腕/肘/胸/ footprint 目标 → 关节+轮速 | **100 Hz** |
| **DYNA-2**（改进 WAM） | 当前子步 → 全身 URR 目标轨迹 | 高于编排器 |
| Workflow orchestrator | VLM；13 决策点、文本长期记忆、指令扩展 | 低于策略 |

### Unified Robot Representation（URR）

- 局部坐标系下 **腕、肘、胸、footprint** 位姿序列；人与 Taku 均可产出 → 人数据可直接教策略层命令接口。

### 可教性与可靠性

- **子任务复合可靠性：** 单步 95% 时 ~182 步循环几乎无法无辅助完成 → 强调 per-step 9's。
- **Teachability 三板斧：** URR 吃人数据；控制器随 demo 变好并改善后续 demo 质量；硬件改版 **重训控制器、保留策略**。
- 另示 **server servicing**、**retrieving a drink** 等新技能快速上手。

### 编排与记忆

- 洗衣 cycle **13 个决策点**；折叠可中断 attend 烘干机。
- 策略内做步骤完成检查与恢复分支；编排器维护 **文本长期记忆**（门状态、计数、哪台机器在跑等）。

## 对 wiki 的映射

- 站点归档：[`sources/sites/dyna-co-dyna-2-1.md`](../sites/dyna-co-dyna-2-1.md)
- 实体页：[`wiki/entities/dyna-2-1.md`](../../wiki/entities/dyna-2-1.md)
- 前代：[`wiki/entities/dyna-2.md`](../../wiki/entities/dyna-2.md)、[`sources/blogs/dyna_2_million_hour_wam.md`](./dyna_2_million_hour_wam.md)
