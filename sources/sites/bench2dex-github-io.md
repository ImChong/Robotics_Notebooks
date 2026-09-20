# Bench2Dex 项目页（bench2dex.github.io）

> 来源归档

- **标题：** Bench2Dex: Benchmarking Visuo-Tactile Bimanual Dexterous Manipulation Across Dexterous Hands
- **类型：** site（项目页 + 文档）
- **URL：** <https://bench2dex.github.io/>
- **文档：** <https://bench2dex.github.io/doc/>
- **论文：** <https://arxiv.org/abs/2609.15726>
- **代码：** <https://github.com/Bench2Dex/Bench2Dex>
- **机构：** 上海交通大学（SJTU）· 复旦大学（Fudan）· 香港大学（HKU）· Inspire Robots · 中关村学院 · COWARobot · 南洋理工大学（NTU）
- **入库日期：** 2026-09-20
- **一句话说明：** Isaac Lab 上跨 **12 种灵巧手** 的 visuotactile **双臂** 仿真基准：26 长程任务、~1.3K 遥操作 demo、统一 8-bit 触觉表征、四通道七类扰动泛化轴；评测 ACT / DP / π₀.₅ / GR00T N1.5。

## 开源核查（步骤 2.5，2026-09-20）

| 资源 | 状态 | 说明 |
|------|------|------|
| 训练 / 推理 / 遥操作代码 | **已开源** | [Bench2Dex/Bench2Dex](https://github.com/Bench2Dex/Bench2Dex)：Isaac Sim 5.1 + Isaac Lab v2.3.2；`main.py --teleop --collect`、replay、四策略管线 |
| 文档 | **已开放** | [bench2dex.github.io/doc](https://bench2dex.github.io/doc/)：环境、任务目录、Policy Usage |
| 资产 / 遥操作数据 / 权重 | **已发布** | HF：[Assets](https://huggingface.co/datasets/Bench2Dex/Assets)、[teleopdata](https://huggingface.co/datasets/Bench2Dex/teleopdata)、[policy_ckpt](https://huggingface.co/Bench2Dex/policy_ckpt)；ModelScope 镜像 |
| 硬件依赖 | **部分** | 遥操作采集需 Manus 手套 + iPhone ARKit 流；仿真评测可不依赖真机采集栈 |

**判定：已开源。** 项目页明确写「All code for training, inference, and teleoperation is open-sourced」；GitHub README 给出完整安装、采集、replay 与文档入口。

## 公开要点（编译自项目页，2026-09-20）

### 定位

- 触觉硬件未收敛、仿真触觉与真机传感器不一致 → 需要 **跨手型一致实验设定** 研究 visuotactile 双臂操作。
- Bench2Dex **不声称** 仿真触觉可替代真机触觉；提供算法开发共享底座。

### 规模与模态

| 项 | 数值 |
|----|------|
| 灵巧手 embodiment | 12 |
| 双臂 manipulation 任务 | 26（工具使用、铰接物体、多阶段） |
| 人类遥操作 demo | ~1.3K |
| 同步观测 | RGB、深度、本体、动作、物体状态、**统一 visuotactile 8-bit 图**、2D/3D box、occupancy |
| 评测策略 | ACT、Diffusion Policy、π₀.₅、GR00T N1.5 |

### 流水线

Manus 手套 + ARKit 腕流 → MediaPipe 21 点 → DexPilot 手部重定向 + Pinocchio 臂 IK → 逐步写入 HDF5（动作先于下一步仿真）；离线 replay 生成相机与触觉。

### 触觉表征

 embodiment 接触面 ray-cast 距离 → 分段量化 8-bit 图（近接触 0.005 mm/级，远距 0.03 mm/级，饱和 ~5.15 mm → 255）→ 高斯平滑；**统一格式、不拟合特定物理传感器**。

### 评测协议

- **Stable SR：** 终态 predicate 持续 dwell（默认 0.5 s）。
- **LSCR：** 曾到达且依赖有效的阶段里程碑（抗故意回退）。
- **效率 / 安全：** 成功 episode 平均用时；SafeSR、违规与掉落率等。
- **四通道：** None（锚定复现）/ Equi.（仅 equivariance 重采样）/ Inv.（仅 invariance 重采样）/ Full（独立全场景采样）。
- **七类扰动：** invariance 五类（背景、桌面纹理、光照、干扰物、相机位姿）+ equivariance 两类（物体初姿、桌面高度）。

### 与代表 benchmark 对比（项目页矩阵）

Bench2Dex 在「多 embodiment + 遥操作 + vision-based tactile + 工具/铰接 + 长程双臂灵巧」上同时勾选；LIBERO / RoboCasa / DROID 等多为单 gripper；DexMimicGen / DexVerse 缺 visuotactile 或双臂横评维度。

## 对 wiki 的映射

- [paper-bench2dex](../../wiki/entities/paper-bench2dex.md)
- [bench2dex 仓库](../repos/bench2dex.md)
- [bench2dex_arxiv_2609_15726](../papers/bench2dex_arxiv_2609_15726.md)
- 交叉：[isaac-lab](../../wiki/entities/isaac-lab.md)、[manipulation](../../wiki/tasks/manipulation.md)、[vla](../../wiki/methods/vla.md)
