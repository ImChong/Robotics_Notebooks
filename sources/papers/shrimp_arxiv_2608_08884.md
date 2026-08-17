# SHRIMP: Iterative Refinement of Robot Task Plans（arXiv:2608.08884）

> 来源归档（ingest）

- **标题：** SHRIMP: Iterative Refinement of Robot Task Plans
- **全称：** Simulation-driven Human-in-the-loop Refinement Interface for Manipulation Planning
- **类型：** paper / hri / task-planning / llm
- **arXiv：** <https://arxiv.org/abs/2608.08884>
- **会议：** UIST '26（DOI [10.1145/3830398.3830644](https://doi.org/10.1145/3830398.3830644)）
- **项目页：** <https://wisc-hci.github.io/SHRIMP/>（归档见 [`sources/sites/shrimp-wisc-hci.md`](../sites/shrimp-wisc-hci.md)）
- **代码：** <https://github.com/Wisc-HCI/SHRIMP>（归档见 [`sources/repos/shrimp.md`](../repos/shrimp.md)）
- **作者：** Mya Schroder、Yuna Hwang、Callie Y. Kim、Leqian Cheng、Jeffrey Li-cheng Liu、Chenchen Zheng、Xinning He、Bilge Mutlu
- **机构：** 威斯康星大学麦迪逊分校 HCI（UW–Madison）
- **入库日期：** 2026-08-17
- **一句话说明：** 自然语言生成层级 primitive 计划，用户在物理数字孪生里重提示/改参数，满意后再上双臂 Franka。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-17）：** 有系统示意图、N=35 用户研究结论；Code 指向 `Wisc-HCI/SHRIMP`。
- **代码仓：** Python + Docker；子模块 `Wisc-HCI/robot-stack`；入口 `setup_scripts/start_desktop.sh`（Isaac Sim / 界面）与 `start_laptop.sh`（Franka FCI）。硬件：2× Panda + Tesollo Dg-3F + RealSense。
- **结论：** **已开源、可运行**（强依赖双机 Docker + 真机硬件）。无 SPDX LICENSE 文件。

## 摘录 1：闭环

语言任务描述有歧义，LLM 计划缺少执行前验证。SHRIMP：语言 + 场景跟踪 → 高层 primitive 序列（含参数化低层原语）→ 抓取/碰撞校验 → 仿真逐步执行 → 重提示或改参数 → 任务历史 → 真机。

## 摘录 2：评测

桌面厨房任务用户研究 **N=35**：提升感知控制感与机器人透明度。不是成功率榜，而是 HRI 指标。

**对 wiki 的映射：** [`wiki/entities/paper-shrimp.md`](../../wiki/entities/paper-shrimp.md)；交叉 [VLA](../../wiki/methods/vla.md)、[模仿学习](../../wiki/methods/imitation-learning.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（Docker 真机栈）
