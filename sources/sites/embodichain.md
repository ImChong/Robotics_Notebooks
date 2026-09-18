# EmbodiChain 官方文档与产品页

- **标题：** EmbodiChain — GPU-Accelerated Robotics Simulation for Embodied AI
- **类型：** site / 官方文档 + 产品页
- **文档：** https://dexforce.github.io/EmbodiChain/main/index.html
- **官网：** https://dexforce.com/embodichain/index.html#/EmbodiChain
- **代码：** https://github.com/DexForce/EmbodiChain（**已开源**，Apache 2.0）
- **机构：** 灵巧智能（DexForce Technology Co., Ltd.）
- **入库日期：** 2026-09-18
- **一句话说明：** DexForce 发布的端到端 GPU 具身智能平台文档站：安装、教程、架构、Gym 任务、IL/RL 与 Sim2Real 部署指南。
- **沉淀到 wiki：** 是 → [`wiki/entities/embodichain.md`](../../wiki/entities/embodichain.md)

---

## 文档结构（GitHub Pages）

| 区块 | 内容 |
|------|------|
| Introduction | 框架定位：资产/场景构建、无头或浏览器检视、机器人学习环境、专家示范、IL/RL 训练、真机部署 |
| Getting Started | [Installation](https://dexforce.github.io/EmbodiChain/main/quick_start/install.html)、Tutorials、How-to Guides |
| Architecture Explorer | DexSim 之上的 Simulation Framework、Embodied Task Program、Gym、RL |
| Resources | [Roadmap](https://dexforce.github.io/EmbodiChain/main/resources/roadmap.html)、[Publications](https://dexforce.github.io/EmbodiChain/main/resources/publications/README.html) |

底层引擎：**DexSim** — 高性能物理与渲染，面向具身 AI 研究与生产。

---

## 产品页要点（dexforce.com）

- 与 GitHub / 文档互链；强调 **GPU 加速**、**模块化**、**Sim2Real**
- 当前版本标注 **Alpha**，功能按 roadmap 持续扩展

---

## 安装与依赖（文档核查 2026-09-18）

- **OS / GPU：** Ubuntu 20.04+、NVIDIA CC 7.0+、驱动 ≥535、CUDA 12.x
- **Python：** 3.10 / 3.11 / 3.12（核心）；`gensim` extra 需 **3.11**（Blender `bpy` ABI）
- **包索引：** DexForce index `http://pyp.open3dv.site:2345/simple/`（`embodichain`、`dexsim_engine`）
- **Docker：** `dexforce/embodichain:ubuntu22.04-cuda12.8`
- **可选：** `[policy-deploy]` ONNX GPU、`[gensim]` SimReady/Scene Engine、独立安装 cuRobo V2

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 项目页 Code 链接 | **有** → GitHub DexForce/EmbodiChain |
| 文档 Footer / 安装页 | 指向同一仓库与 PyPI + DexForce index |
| 宣称开源 | README + LICENSE Apache 2.0；**已开源** |
| 边界 | `dexsim_engine` wheel 经 DexForce index 分发，非默认 PyPI；框架本体代码在 GitHub |

---

## 对 wiki 的映射

- [EmbodiChain](../../wiki/entities/embodichain.md)
- [EmbodiChain 仓库归档](../repos/embodichain.md)
- [RoboSynChallenge](../../wiki/entities/paper-robosynchallenge.md)
