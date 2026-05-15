# Source: ProtoMotions (NVlabs/ProtoMotions)

- **Title**: ProtoMotions3: An Open-source Framework for Humanoid Simulation and Control
- **URL**: https://github.com/NVlabs/ProtoMotions
- **Authors**: Chen Tessler, Yifeng Jiang, Xue Bin Peng, et al.
- **Year**: 2025
- **Type**: Research Framework / Simulation Engine
- **License**: Apache-2.0
- **Documentation**: https://protomotions.github.io/

## 核心内容（仓库 README 归纳）

- **定位**：GPU 加速的 **仿真 + 学习** 框架，面向 **数字人** 与 **人形机器人** 的物理可仿真控制；强调 **模块化、可扩展、可规模化**，Apache-2.0，社区驱动。
- **姊妹项目**：轻量运动模仿学习栈见 [MimicKit](https://github.com/xbpeng/MimicKit/tree/main)。

### 大规模运动学习

- 公开 [**AMASS**](https://amass.is.tue.mpg.de/) 全量约 **40+ 小时** 动捕：官方示例宣称约 **12 小时 / 4×A100** 可训出覆盖大量技能的单策略类结果（以官方 Quick Start 与论文级复现配置为准）。
- 多 GPU 分片数据：示例配置可达 **24×A100**，每卡子集约 **13K motions**，配合 [**BONES**](https://huggingface.co/datasets/bones-studio/seed) 与 [**SOMA**](https://github.com/NVlabs/SOMA-X) 骨架格式（见文档 *SEED BVH Data Preparation*）。

### 重定向（Retargeting）

- **v3 起**：内置基于 [**PyRoki**](https://github.com/chungmin99/pyroki) 的优化式重定向，宣称可 **一条命令** 将 AMASS 类数据转到目标机器人；早期版本使用 [Mink](https://github.com/kevinzakka/mink)。
- 教程入口：[Retargeting with PyRoki](https://protomotions.github.io/tutorials/workflows/retargeting_pyroki.html)。

### 多仿真后端与 Sim2Sim

- README 徽章所示后端组合（版本以仓库为准）：**Newton**、**IsaacLab 2.3.0**、**IsaacGym Preview 4**、**MuJoCo 3.0+**；**Genesis** 标记为 *untested*，仓库内提供社区向 `protomotions/simulator/genesis/` 示例接口。
- **Sim2Sim**：同一策略可在 **Newton / IsaacGym / MuJoCo** 等之间切换 `--simulator=...` 做一键对照；强调观测设计贴近 **真机可得的传感器**。

### Sim2Real 与部署

- 在 **BONES-SEED**（约 **142K motions**）上训练 **General Tracking Policy**，宣称可 **零样本** 迁移到 **Unitree G1**（以官方复现与硬件条件为准）。
- 导出 **单个 ONNX**（观测计算 baked-in），部署侧只需喂原始传感器；官方示例基于 [**RoboJuDo**](https://github.com/HansZ8/RoboJuDo) 集成，声称「仅增加策略文件、无需改 RoboJuDo 核心」。
- 完整教程：[G1 Deployment: Data to Real Robot](https://protomotions.github.io/tutorials/workflows/g1_deployment.html)。
- 机器人切换示例：`--robot-name=smpl` → `--robot-name=h1_2` 等，需先准备对应重定向动作。

### 与生成式运动（Kimodo）衔接

- 与 [**Kimodo**](https://research.nvidia.com/labs/sil/projects/kimodo/) 文生运动衔接：生成动作 → ProtoMotions 训练物理策略 → 可上真机；数据准备见 [Kimodo Data Preparation](https://protomotions.github.io/getting_started/kimodo_preparation.html)。

### 其他能力（README 列举）

- **程序化场景 / SDG**：从种子动作集出发，用 RL 适配增强场景以扩数据。
- **生成式策略**：例如与 **MaskedMimic** 等生成式控制路线结合（见 NVIDIA 对应项目页）。
- **地形导航**、**高保真渲染**（Isaac Sim 5.0+，含 Gaussian splatting 背景等展示向能力）。
- **自定义环境**：以 **Control / Observation / Reward / Experiment** 模块化组合（官方以 steering 任务为表格式示例）。
- **新 RL 算法**：模块化 agent 设计，README 以 **ADD** 实现约 50 行量级为例。
- **自定义仿真器**：实现 `protomotions/simulator/base_simulator/` 约定 API。
- **接入新机器人**：添加 MuJoCo `.xml`、填写 `robot_configs`、在 `factory.py` 注册（官方 README 三步）。

## 文档索引（官方）

- [Installation](https://protomotions.github.io/getting_started/installation.html)
- [Quick Start](https://protomotions.github.io/getting_started/quickstart.html)
- [AMASS / PHUMA / SEED BVH / SEED G1 CSV / Kimodo 数据准备](https://protomotions.github.io/getting_started/) 各子页
- [Tutorials](https://protomotions.github.io/tutorials/)
- [API Reference](https://protomotions.github.io/api_reference/)

## 与 MimicKit 的关系

- **MimicKit**：更偏 **算法与论文复现** 的轻量框架，运动模仿方法族谱集中。
- **ProtoMotions3**：更偏 **大规模并行仿真、多后端、数据管线与部署导出** 的「全栈」研究平台。

## BibTeX（仓库提供）

```bibtex
@misc{ProtoMotions,
  title = {ProtoMotions3: An Open-source Framework for Humanoid Simulation and Control},
  author = {Tessler*, Chen and Jiang*, Yifeng and Peng, Xue Bin and Coumans, Erwin and Shi, Yi and Zhang, Haotian and Rempe, Davis and Chechik†, Gal and Fidler†, Sanja},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVlabs/ProtoMotions/}},
}
```
