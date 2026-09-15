# MGDP（Advanced Science 2026）

> 来源归档（ingest）

- **标题：** MGDP：面向四足行走的通用深度感知模型
- **英文标题：** MGDP: Mastering a Generalized Depth Perception Model for Quadruped Locomotion
- **类型：** paper / quadruped / perceptive-locomotion / reinforcement-learning / contrastive-learning / sim2real
- **期刊：** *Advanced Science* **13**, e24345（2026）
- **DOI：** <https://doi.org/10.1002/advs.202524345>
- **PDF：** <https://advanced.onlinelibrary.wiley.com/doi/pdf/10.1002/advs.202524345>
- **项目页：** <https://arclab-hku.github.io/MGDP/>
- **代码：** <https://github.com/arclab-hku/MGDP>（步骤 2.5 已开源，见 [`sources/repos/mgdp.md`](../repos/mgdp.md)）
- **机构：** 香港大学（HKU）；北京理工大学（BIT）
- **作者：** Yinzhao Dong†, Ji Ma†, Yidan Lu, Jiahui Zhang, Wanyue Li, Yeke Chen, Teng Zhang, Xuechao Chen, Zhangguo Yu, Peng Lu（† 同等贡献；Peng Lu 通讯）
- **仿真栈：** Isaac Gym；NVIDIA Warp 并行深度渲染（`warp_sensor`）
- **开源：** **已开源**
- **入库日期：** 2026-09-15

## 核心论文摘录

### 1) 通用深度感知模型（对比学习 + 去噪）

- 多模态输入：**深度图 + 高程图（height map）**；对比学习提取**低维、可跨地形泛化**的地形特征。
- **显式深度去噪** 提升对传感器噪声/伪影的鲁棒性；感知与动力学**解耦**，降低训练显存占用。
- **NVIDIA Warp** 并行计算深度图，缓解感知 DRL 的高算力开销。
- **对 wiki 的映射：** [paper-mgdp-generalized-depth-perception](../../wiki/entities/paper-mgdp-generalized-depth-perception.md)

### 2) 两阶段训练：感知预训练 → 行走控制器

- **Stage 1：** 训练 Generalized Depth Perception Model（`train.py`，如 `random_dog_stage1`）。
- **Stage 2：** 冻结/复用感知特征，训练 Generalized Perception-Based Locomotion Controller（`resume.py`，如 `random_dog_stage2`）；支持 `DOG_NAMES` 多构型混训。
- **地形自适应奖励：** 按地形特性调节惩罚强度，单阶段习得攀爬、跳跃、匍匐、挤压等技能，**无需蒸馏流水线**。
- **对 wiki 的映射：** 同上

### 3) 跨构型与跨地形评测

- **9** 种四足：**A1、B1、Go1、Go2、Lite3、Spot、Aliengo、ANYmal C、Mini Cheetah**。
- **10** 类极端地形 traversal 定量表（各构型可达最大难度 / 地形最大难度之比）。
- 仿真：离散缝隙、踏脚石、连续崎岖等；真机：楼梯、坡道、户外非结构化，**直接部署** sim-to-real。
- **对 wiki 的映射：** 同上；交叉 [pie-perceptive-locomotion](../../wiki/methods/pie-perceptive-locomotion.md)、[paper-apt-rl-agile-perceptive-quadruped-locomotion](../../wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md)

## 步骤 2.5 开源核查（2026-09-15）

- 项目页列 **Code** → `arclab-hku/MGDP`。
- 仓库含 `legged_gym/scripts/train.py`、`resume.py`、`vis_stage1.py`、`vis_stage2.py`、`play_terrain.py`；`warp_sensor` 子包；**vendor 内嵌 Isaac Gym**。
- 仓库根目录 **未单独列出 LICENSE 文件**；以 GitHub 页面与 README 安装/训练步骤为准。
- **结论：** **已开源** — 训练与可视化入口完整；权重需自行训练。

## 当前提炼状态

- [x] 项目页 + 仓库步骤 2.5 核查
- [x] wiki 实体页 + 源码运行时序图
- [x] `sources/repos/mgdp.md`
