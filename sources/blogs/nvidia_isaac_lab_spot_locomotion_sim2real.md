# Closing the Sim-to-Real Gap: Training Spot Quadruped Locomotion with NVIDIA Isaac Lab

> 来源归档（blog / NVIDIA Developer Blog）

- **标题：** Closing the Sim-to-Real Gap: Training Spot Quadruped Locomotion with NVIDIA Isaac Lab
- **类型：** blog
- **原始链接：** https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/
- **机构：** Boston Dynamics × NVIDIA × The AI Institute
- **入库日期：** 2026-08-30
- **一句话说明：** **Spot RL Researcher Kit**（关节级 API + Jetson AGX Orin 载荷）配合 Isaac Lab **`Isaac-Velocity-Flat-Spot-v0`**，用 **RSL-rl PPO** 训平地速度跟踪，经 **ONNX + Spot Python SDK** 在真机 **零样本** 部署，PS4 手柄发速度指令。

## 开源与项目页核查（步骤 2.5）

| 组件 | 开放程度 | 说明 |
|------|----------|------|
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | **已开源** | 含 `Isaac-Velocity-Flat-Spot-v0` 等 Spot velocity 任务 |
| [spot-rl-example](https://github.com/boston-dynamics/spot-rl-example) | **已开源** | BD 发布的 Jetson 部署示例（AI Institute 原开发） |
| Spot Python SDK（joint-level API） | **部分 / 需 Kit** | 须从 BD 获取带 joint API 的 SDK 并装入 `external/spot_python_sdk` |
| RL Researcher Kit 硬件 | **商业套件** | Spot + Orin 支架/线缆等；Orin 另购 |
| Isaac Lab 训练脚本 | **已开源** | `source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-Spot-v0` |

## 核心摘录（归纳，非全文）

### 训练目标与 MDP

- **Goal：** 平地跟踪随机采样的 **x / y / yaw 线速度与角速度**
- **观测：** 目标速度 + 博客图 1 所列状态（与仿真一致）
- **动作：** **12 DoF 关节位置** → 低层关节控制器参考位置
- **Domain randomization：** 图 1 所列多阶段随机化参数
- **网络：** MLP [512,256,128]；**PPO（RSL-rl）**
- **规模：** 4096 env × 15000 iter ≈ **4 h @ RTX 4090**；**85k–95k FPS**

### 训练命令（博客原文）

```bash
cd <path_to_isaac_lab>
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Spot-v0 --num_envs 4096 --headless
```

### 真机部署栈

- **算力：** Jetson AGX Orin 作 Spot 自定义 payload（以太网 + 供电 + 支架）
- **推理：** 训练 PC 上 `play.py` 导出 **`.onnx` + env 配置** → scp 到 Orin
- **控制：** `spot_rl_demo.py` + Boston Dynamics **State API** 构造与仿真相同观测；**PS4 手柄** 发速度指令
- **网络：** Spot 转发 **20022** 到 payload；Jetson 与 Spot 有线网段（示例 192.168.50.x）
- **前置：** Spot app **Release Control**；Orin 刷 JetPack 6

### 与 arXiv:2504.17857 的关系

- 同一 **Spot RL Researcher Kit + Isaac Lab** 生态；本篇为 **官方教程级平地 velocity 零样本部署**
- [分布距离 Sim2Real 标定论文](../../wiki/entities/paper-spot-rl-distributional-sim2real.md) 侧重 **Wasserstein/MMD + CMA-ES 仿真参数优化** 与 **>5.2 m/s 高性能步态**——方法论互补，非同一实验

## 对 wiki 的映射

- [NVIDIA Isaac Lab Spot  locomotion Sim2Real](../../wiki/entities/nvidia-isaac-lab-spot-locomotion-sim2real.md) — 本篇博客编译页
- [Spot 分布距离 Sim2Real（论文实体）](../../wiki/entities/paper-spot-rl-distributional-sim2real.md) — 同 Kit 的研究向标定管线
- [Isaac Lab 默认环境](../../wiki/entities/isaac-lab-default-environments.md) — `Isaac-Velocity-Flat-Spot-v0`
- [Boston Dynamics](../../wiki/entities/boston-dynamics.md) — Spot 平台
- [Locomotion](../../wiki/tasks/locomotion.md) — 四足速度跟踪上下文
