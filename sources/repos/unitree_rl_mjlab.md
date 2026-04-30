# unitree_rl_mjlab

> 来源归档

- **标题：** unitree_rl_mjlab
- **类型：** repo
- **来源：** unitreerobotics（Unitree 官方 GitHub 组织）
- **链接：** https://github.com/unitreerobotics/unitree_rl_mjlab
- **入库日期：** 2026-04-29
- **一句话说明：** Unitree 官方基于 mjlab + MuJoCo Warp 的人形/四足机器人 RL 训练框架，覆盖 Go2/G1/H1_2/H2 等 7 款机型，支持速度跟踪与动作模仿，内建 Sim2Real 部署链路（ONNX → C++ 部署）。
- **沉淀到 wiki：** 是 → [`wiki/entities/unitree-rl-mjlab.md`](../../wiki/entities/unitree-rl-mjlab.md)

---

## 核心定位

unitree_rl_mjlab 是 Unitree 官方维护的、面向旗下机器人的 RL 训练框架，与第三方框架（legged_gym、robot_lab、AMP_mjlab）的关系：

| 框架 | 底层仿真 | 官方维护 | 支持机型 |
|------|---------|---------|---------|
| legged_gym | Isaac Gym | 否（ETH） | 通用 |
| robot_lab | Isaac Lab | 否（个人） | 26+ |
| AMP_mjlab | mjlab | 否（社区） | G1 |
| **unitree_rl_mjlab** | **mjlab** | **是（Unitree）** | **Go2/A2/G1/H1_2/H2 等 7 款** |

---

## 支持机器人（7 款）

Go2、A2、As2、G1、R1、H1_2、H2

---

## 任务与流程

### 主要任务

| 任务 | 示例命令 |
|------|---------|
| 速度跟踪（locomotion） | `python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096` |
| 动作模仿（motion imitation） | `Unitree-G1-Tracking-No-State-Estimation` |

### Sim2Real 部署链路

```
MuJoCo 训练
    ↓
ONNX 策略导出
    ↓
C++ 控制程序编译（cyclonedds + unitree_sdk2）
    ↓
unitree_mujoco 仿真验证
    ↓
真机部署（以太网 192.168.123.222，调试模式启动）
```

---

## 技术栈

- **训练框架**：mjlab（Isaac Lab API + MuJoCo Warp）
- **语言组成**：C++ 63%、C 19%、Python 15%、CMake 3%
- **通信依赖**：cyclonedds、unitree_sdk2
- **部署格式**：ONNX

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [mjlab.md](mjlab.md) | unitree_rl_mjlab 直接以 mjlab 为底层 |
| [unitree.md](unitree.md) | Unitree 品牌下的官方 RL 训练实现 |
| [amp_mjlab.md](amp_mjlab.md) | 同基于 mjlab，AMP_mjlab 侧重 AMP + recovery，本框架侧重官方标准任务 |
| [legged_gym.md](legged_gym.md) | 同类竞品，legged_gym 基于 IsaacGym，unitree_rl_mjlab 基于 MuJoCo |
