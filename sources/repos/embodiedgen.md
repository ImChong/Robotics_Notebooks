# EmbodiedGen（Horizon Robotics 官方仓库）

> 来源归档

- **标题：** EmbodiedGen V2 — An Agentic, Simulation-Ready 3D World Engine for Embodied AI
- **类型：** repo
- **机构：** 地平线（Horizon Robotics）+ 无问芯穹（WuwenAI）
- **项目页：** <https://horizonrobotics.github.io/EmbodiedGen/>
- **文档：** <https://horizonrobotics.github.io/EmbodiedGen/docs/>
- **代码：** <https://github.com/HorizonRobotics/EmbodiedGen>
- **数据集：** <https://huggingface.co/datasets/HorizonRobotics/EmbodiedGenData>
- **论文：**
  - V2：[arXiv:2607.07459](https://arxiv.org/abs/2607.07459)
  - V1：[arXiv:2506.10600](https://arxiv.org/abs/2506.10600)
- **入库日期：** 2026-07-14
- **一句话说明：** 开源 **sim-ready 3D 世界引擎**：把语言、图像与编辑指令编译为带碰撞/物理/affordance 的三维资产、多房间场景与任务驱动交互世界，可导出至 SAPIEN / Isaac Sim / Isaac Gym / MuJoCo / Genesis / PyBullet，并作为在线 RL 训练环境。
- **沉淀到 wiki：** [EmbodiedGen V2（Simulation-Ready 3D World Engine）](../../wiki/entities/paper-embodiedgen-v2-sim-ready-world-engine.md)

---

## 仓库状态（README / 文档，ingest 快照）

| 项 | 状态 |
|----|------|
| 代码 | 已开源（`v2.0.0` tag；`git clone` + `bash install.sh basic`） |
| 文档站 | 已上线（安装、教程、Services 在线 demo 索引） |
| 数据集 | [EmbodiedGenData](https://huggingface.co/datasets/HorizonRobotics/EmbodiedGenData)（Apache-2.0，~346 GB） |
| Docker | [wangxinjie/embodiedgen](https://hub.docker.com/repository/docker/wangxinjie/embodiedgen) |
| HF Spaces | Image-to-3D / Text-to-3D / Texture-Gen / Gallery-Explorer 等 |

## V2 能力栈（README 归纳）

| 模块 | CLI / 入口 | 要点 |
|------|------------|------|
| **Sim-Ready 资产** | `img3d-cli` / `text3d-cli` / `texture-cli` | 图/文 → URDF + mesh + 3DGS；后端可切换 **SAM3D / TRELLIS / Hunyuan3D** |
| **Affordance** | `affordance-cli` | 部件级语义 + 仿真验证 6-DoF 抓取 |
| **大场景** | `room-cli` / `scene3d-cli` | 多房间可编辑房屋；3DGS 背景场景 |
| **任务世界** | `layout-cli` + `sim-cli` | NL 任务 → scene graph → 可加载 layout.json |
| **Vibe Coding** | Claude Code `/embodiedgen:*` | 有状态 NL 编辑；physics-validated skill call |
| **跨模拟器** | `MeshtoMJCFConverter` / `MeshtoUSDConverter` | 六引擎一致几何/碰撞/物理元数据 |
| **机器人学习** | `parallel_sim.py` / `eval_collision_success.py` | 并行 gym 环境、抓取质量评测 |

## 安装与依赖（Quick Start）

```sh
git clone https://github.com/HorizonRobotics/EmbodiedGen.git
cd EmbodiedGen && git checkout v2.0.0
conda create -n embodiedgen python=3.10.13 -y && conda activate embodiedgen
bash install.sh basic   # 可选 cu126 / affordance / room / scene3d 等 profile
```

- 多数管线需配置 `embodied_gen/utils/gpt_config.yaml`（GPT agent）。
- 许可：**Apache-2.0**。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Generative World Models](../../wiki/methods/generative-world-models.md) | **环境生成式世界建模**；产出可 RL 的 3D 状态而非单帧视频 |
| [wm-action-consequence-category-03-geometry-4d](../../wiki/overview/wm-action-consequence-category-03-geometry-4d.md) | 策展「环境层」支柱：sim-ready 场景 vs 像素–几何预测 |
| [GigaWorld-1 论文实体](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md) | **WM 策略评估** vs EmbodiedGen **3D 可执行环境** |
| [Sim2Real](../../wiki/concepts/sim2real.md) | 伴随 sim-to-real RL 研究：生成环境训练策略迁移真机 |

## 对 wiki 的映射

- 主实体页：**`wiki/entities/paper-embodiedgen-v2-sim-ready-world-engine.md`**（增补工程实践、文档与数据集链接）
- 数据集归档：**`sources/datasets/embodiedgen-data.md`**
- 交叉：**`wiki/overview/wm-action-consequence-category-04-eval-posttrain.md`**、**`wiki/methods/generative-world-models.md`**

## 外部参考（便于复核）

- [EmbodiedGen 项目页](https://horizonrobotics.github.io/EmbodiedGen/)
- [EmbodiedGen 文档](https://horizonrobotics.github.io/EmbodiedGen/docs/)
- [GitHub 仓库](https://github.com/HorizonRobotics/EmbodiedGen)
- [EmbodiedGenData（HuggingFace）](https://huggingface.co/datasets/HorizonRobotics/EmbodiedGenData)
- [EmbodiedGen V2 论文（arXiv:2607.07459）](https://arxiv.org/abs/2607.07459)
- [EmbodiedGen V1 论文（arXiv:2506.10600）](https://arxiv.org/abs/2506.10600)
