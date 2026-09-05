---
type: entity
tags: [robotis, mujoco, simulation, assets, physical-ai, open-source]
status: complete
updated: 2026-09-05
summary: "ROBOTIS robotis_mujoco_menagerie：官方 MuJoCo 模型集，含 AI Worker（FFW）、OMY/OMX、OpenMANIPULATOR-X、OP3、TurtleBot3 等，供仿真与 LeRobot 教程。"
related:
  - ./robotis.md
  - ./robotis-ai-worker.md
  - ./robotis-open-manipulator-line.md
  - ./turtlebot3.md
  - ./robotis-op3.md
  - ./cyclo-lab.md
  - ./lerobot.md
  - ./mujoco.md
sources:
  - ../../sources/repos/robotis_mujoco_menagerie.md
---

# ROBOTIS MuJoCo Menagerie

**robotis_mujoco_menagerie**（[`ROBOTIS-GIT/robotis_mujoco_menagerie`](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)，~69★）是 ROBOTIS 维护的 **MuJoCo XML/网格资产集**，覆盖 Physical AI 与教育平台机型，供仿真、演示与 LeRobot 类教程使用。

## 一句话定义

把 AI Worker（FFW 系列）、OpenMANIPULATOR（OMY/OMX/X）、OP3、TurtleBot3 等 ROBOTIS 机型整理成可在 MuJoCo 中加载的官方模型目录，作为 Isaac Lab / 真机栈之外的轻量仿真入口。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MuJoCo | Multi-Joint dynamics with Contact | 本资产目标物理引擎 |
| FFW | Freedom From Work | AI Worker 型号前缀（SH5/SG2/BG2） |
| OMY / OMX | OpenMANIPULATOR-Y / X | 机械臂资产 |
| MJCF | MuJoCo XML Format | 模型描述格式 |
| IL | Imitation Learning | 常与 LeRobot MuJoCo 教程联用 |

## 为什么重要

- **官方资产可信度**：相对第三方 URDF 转换，厂商 menagerie 更适合做演示与数据收集教程的默认场景。
- **跨产品线一仓**：半人形、臂、小轮式、小型人形同仓，便于写「多机型 MuJoCo 示例」。
- **与 Cyclo 文档互链**：ai_worker / physical_ai_tools README 均指向本仓作为仿真模型入口。

## 核心原理

| 目录/机型 | 内容 |
|-----------|------|
| `robotis_ffw` | FFW-SH5 / SG2 / BG2 |
| `robotis_omy` / `robotis_omx` | OMY / OMX |
| `robotis_open_manipulator_x` | OpenMANIPULATOR-X |
| `robotis_op3` | OP3（致谢 DeepMind menagerie 参考） |
| `robotis_tb3` | TurtleBot3 Burger / Waffle Pi |

根目录 Apache-2.0；**各模型子目录可能有独立 LICENSE**——再分发前必读。

## 工程实践

1. 克隆仓库，按目标机型进入子目录加载 MJCF。
2. LeRobot / 演示数据：社区示例 [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)（README 引用）。
3. 需要 GPU 并行 RL / Mimic 时转 [cyclo_lab](./cyclo-lab.md)（Isaac Lab），本仓不替代 Lab 任务定义。
4. 真机部署仍走 [ai_worker](./robotis-ai-worker.md) 等 ROS 2 包，勿把 MuJoCo 执行器增益直接当真机增益。

## 局限与风险

- **开源状态：已开源**（资产公开；注意逐模型许可证）。
- **保真度边界**：教学与 IL 数据收集足够；高保真接触/驱动非线性仍需系统辨识或 Isaac 侧域随机。
- **与 DeepMind mujoco_menagerie 关系**：OP3 等有历史参考链接；以本仓当前文件为准。

## 关联页面

- [ROBOTIS hub](./robotis.md)
- [AI Worker](./robotis-ai-worker.md) · [OpenMANIPULATOR 线](./robotis-open-manipulator-line.md)
- [TurtleBot3](./turtlebot3.md) · [OP3](./robotis-op3.md)
- [cyclo_lab](./cyclo-lab.md) · [MuJoCo](./mujoco.md)

## 参考来源

- [sources/repos/robotis_mujoco_menagerie.md](../../sources/repos/robotis_mujoco_menagerie.md)
- 上游：<https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie>

## 推荐继续阅读

- [MuJoCo 文档](https://mujoco.readthedocs.io/)
- [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)（社区对照）
