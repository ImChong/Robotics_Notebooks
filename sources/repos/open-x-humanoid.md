# Open-X-Humanoid（GitHub 组织）

> 来源归档

- **标题：** Open X-Humanoid — GitHub Organization
- **类型：** repo（组织级多仓生态）
- **来源：** 北京人形机器人创新中心（X-Humanoid）
- **组织页：** https://github.com/Open-X-Humanoid
- **组织描述：** the first innovation center in China that focuses on the core technology, product development, and application ecosystem construction of humanoid robots.
- **Website 字段：** https://opensource.x-humanoid-cloud.com/
- **Email：** opensource@x-humanoid.com
- **公开仓库数（核查日）：** 23
- **组织创建：** 2025-06-25
- **入库日期：** 2026-07-21
- **一句话说明：** X-Humanoid 官方公开代码组织：覆盖天工本体（URDF/ROS/Docs/SDK）、IsaacLab RL（TienKung-Lab）、VLA/VLM（HEX、XR-1、Pelican）、数据工具（RoboMIND / 训练工具链）、地图可视化（BICMap）等。
- **开源状态：** **已开源** — 组织内多数仓可克隆；权重常另挂 Hugging Face `X-Humanoid`；各仓许可证以仓库 LICENSE 为准（常见 BSD-3 / Apache-2.0 / MIT）。
- **沉淀到 wiki：** 是 → [`wiki/entities/x-humanoid.md`](../../wiki/entities/x-humanoid.md)

---

## 为什么值得保留

- **可复现主入口**：相对官网与 Discuz 社区，GitHub 是训练/部署/二次开发的可执行源。
- **栈完整**：从 URDF → ROS/SDK → IsaacLab 运控 → VLA 微调 → 数据集工具链 → Web 地图 SDK，适合做「中心制开源人形」对照样本。
- **与历史 org 并存**：部分 README / 文档仍指向 `x-humanoid-robomind`；现行公开开发以 **`Open-X-Humanoid`** 为准。

## 仓库矩阵（2026-07-21 `gh api` 抽样）

### A. 本体 / 底层软件

| 仓库 | Stars（约） | 角色 |
|------|-------------|------|
| [TienKung_Docs](https://github.com/Open-X-Humanoid/TienKung_Docs) | 5 | Lite/Pro 手册与 SDK PDF |
| [TienKung_URDF](https://github.com/Open-X-Humanoid/TienKung_URDF) | 39 | URDF + mesh |
| [TienKung_ROS](https://github.com/Open-X-Humanoid/TienKung_ROS) | 19 | ROS 底层：`body_control` 等 |
| [Deploy_Tienkung](https://github.com/Open-X-Humanoid/Deploy_Tienkung) | 38 | 真机部署（C++；TienKung-Lab 链出） |
| [xhumanoid_sdk](https://github.com/Open-X-Humanoid/xhumanoid_sdk) | 1 | **具身天工 3.0** ROS 2 Jazzy 示例（Ubuntu 24.04） |

### B. 运动控制 / 仿真

| 仓库 | Stars（约） | 角色 |
|------|-------------|------|
| [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) | 820 | Isaac Sim 4.5 / Isaac Lab 2.1 + RSL-RL；AMP + 周期步态；Sim2Sim→MuJoCo；真机验证 |
| [xSIM_MUJOCO](https://github.com/Open-X-Humanoid/xSIM_MUJOCO) | 11 | MuJoCo 仿真相关 |
| [x-humanoid-vla-simulation-benchmark](https://github.com/Open-X-Humanoid/x-humanoid-vla-simulation-benchmark) | 5 | Isaac Sim 上 VLA 评测 |

### C. 数据 / 训练工具链

| 仓库 | Stars（约） | 角色 |
|------|-------------|------|
| [x-humanoid-training-toolchain](https://github.com/Open-X-Humanoid/x-humanoid-training-toolchain) | 70 | 适配天工 + **RoboMIND** → LeRobot v2.1 |
| [RoboMIND-Sim](https://github.com/Open-X-Humanoid/RoboMIND-Sim) | 14 | RoboMIND 仿真 |
| [RoboMIND-dataset-utils](https://github.com/Open-X-Humanoid/RoboMIND-dataset-utils) | 2 | 数据集工具 |

**RoboMIND 规模（训练工具链 README）：** 约 **107k** 真实演示；**479** 任务；**96** 物体类；项目页 [x-humanoid-robomind.github.io](https://x-humanoid-robomind.github.io/)；HF [`x-humanoid-robomind/RoboMIND`](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)。

### D. 具身模型（VLA / VLM / RL）

| 仓库 | Stars（约） | 角色 |
|------|-------------|------|
| [HEX](https://github.com/Open-X-Humanoid/HEX) | 335 | 全身 VLA；跨本体身体部位对齐；HF 模型/数据；预训练与微调已释出（2026-05） |
| [XR-1](https://github.com/Open-X-Humanoid/XR-1) | 186 | 统一视觉–运动表征的 VLA（ICML 2026 Oral）；HF/ModelScope；真机部署样例 TODO |
| [pelican-vl](https://github.com/Open-X-Humanoid/pelican-vl) | 86 | Pelican-VL 1.0 具身脑（7B–72B）；HF/ModelScope |
| [Pelican-VLA05](https://github.com/Open-X-Humanoid/Pelican-VLA05) | 16 | Pelican VLA 相关 |
| [Robo-ValueRL](https://github.com/Open-X-Humanoid/Robo-ValueRL) | 18 | 价值引导 offline→online 操作 RL；天工双臂真机验证 |
| [xMimic](https://github.com/Open-X-Humanoid/xMimic) / [xGMR](https://github.com/Open-X-Humanoid/xGMR) | — | 模仿 / 运动重定向相关 |
| [Humanoid-Occupancy](https://github.com/Open-X-Humanoid/Humanoid-Occupancy) | 92 | 占用 / 感知相关 |

> **命名注意：** 本组织的 **XR-1**（北京人形 VLA）与本库已收录的小米 **Xiaomi-Robotics-1（官网代号 XR-1）** 不是同一项目。

### E. 可视化 / 地图

| 仓库 | Stars（约） | 角色 |
|------|-------------|------|
| [BICMap](https://github.com/Open-X-Humanoid/BICMap) | 38 | WebGL 机器人地图 SDK（MapLibre + Three.js）；npm `@x-humanoid-cloud/bic-map`；文档 https://bicmap.x-humanoid-cloud.com |
| [BICMap-Mobile](https://github.com/Open-X-Humanoid/BICMap-Mobile) / [BICMap-Python](https://github.com/Open-X-Humanoid/BICMap-Python) | — | 移动端 / Python 配套 |

## TienKung-Lab 运行入口（摘录）

依赖：Isaac Sim **4.5.0**、Isaac Lab **2.1.0**、Python 3.10、`rsl_rl` 子树。

```bash
cd TienKung-Lab && pip install -e .
cd TienKung-Lab/rsl_rl && pip install -e .
python legged_lab/scripts/train.py --task=walk --logger=tensorboard --headless --num_envs=64
```

运动重定向：经 [GMR](https://github.com/YanjieZe/GMR)，当前支持 SMPL-X 类（AMASS、OMOMO）。部署仓：[Deploy_Tienkung](https://github.com/Open-X-Humanoid/Deploy_Tienkung)。

## Hugging Face 组织

公开模型/数据集合常挂在 [`huggingface.co/X-Humanoid`](https://huggingface.co/X-Humanoid)（HEX、XR-1、Pelican、Robo-ValueRL 等）。

## 与本仓库现有资料的关系

- 官网 / 社区：[`sources/sites/x-humanoid.md`](../sites/x-humanoid.md)、[`sources/sites/x-humanoid-opensource-cloud.md`](../sites/x-humanoid-opensource-cloud.md)
- wiki 实体：[天工](../../wiki/entities/tienkung-humanoid-open-source.md)、[HEX 161#038](../../wiki/entities/paper-loco-manip-161-038-hex.md)、[Pelican-Unified](../../wiki/methods/pelican-unified-1.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/x-humanoid.md`](../../wiki/entities/x-humanoid.md)「开源仓库矩阵」与「工程实践」节。
