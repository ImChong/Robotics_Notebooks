# Motphys/MotrixLab

> 来源归档

- **标题：** MotrixLab
- **类型：** repo
- **组织：** Motphys
- **链接：** https://github.com/Motphys/MotrixLab
- **文档：** https://motrixlab.readthedocs.io/zh-cn/stable/
- **仿真引擎：** https://github.com/Motphys/motrixsim-docs
- **许可：** Apache-2.0（仓库 LICENSE）
- **语言：** Python 3.10.x + Rust（MotrixSim 后端）
- **Stars / Forks：** ~145 / —（2026-09-13，GitHub 首页）
- **入库日期：** 2026-09-13
- **一句话说明：** 基于 MotrixSim 的通用机器人 RL 训练平台：环境一次定义，SKRL / RSL-RL / 自研 FastSAC 多算法训练，支持 Sim2Sim（MuJoCo）与真机部署 CLI。
- **开源状态：** **已开源** — `install.sh` + `uv` 工作区；`scripts/train.py` / `play.py` / `view.py` / `export_onnx.py` 可运行；内置 50+ 环境与 7 款机器人模型。
- **平台归档：** [motphys-motrix.md](./motphys-motrix.md)
- **沉淀到 wiki：** [motrix](../../wiki/entities/motrix.md)、[microduck-ball-balance](../../wiki/tasks/microduck-ball-balance.md)

---

## 定位

MotrixLab 把 **MotrixSim**（Rust CPU/GPU 批量物理）与 **motrix_envs**（观测/奖励/终止 Manager 工作流）和 **motrix_rl**（SKRL PPO、RSL-RL PPO、自研 FastSAC）拼成单一 CLI。与 Pollen 官方 [microduck_rl](../repos/microduck_rl.md)（mjlab + PPO）并列，是 Microduck 在 Motrix 栈上的第二套可复现训练入口。

## 可运行入口

```bash
git clone https://github.com/Motphys/MotrixLab && cd MotrixLab && git lfs pull
sh install.sh
source .venv/bin/activate

# 预览环境（不训练）
python scripts/view.py env=microduck-ball-balance

# 球平衡 + FastSAC（2048 并行 env，约 5–10 min 可见策略；play=true 边训边渲染）
python scripts/train.py task=microduck-ball-balance/motrix.fastsac play=true

# 评估 checkpoint
python scripts/play.py env=microduck-ball-balance
```

依赖：Python 3.10、`uv`、Git LFS、NVIDIA CUDA 或 AMD ROCm GPU（`install.sh` 自动选 PyTorch wheel）。

## Microduck 相关任务（MotrixLab 注册名）

| 环境 id | 算法配方 | 说明 |
|---------|----------|------|
| `microduck-walk-flat` | `skrl.ppo` / `rslrl.ppo` / `motrix.fastsac` | 速度跟踪平地行走 |
| `microduck-walk-rough` | 同上 | 粗糙地形行走 |
| `microduck-ball-balance` | **`motrix.fastsac` only** | 14-DoF 双足站在自由篮球（r=0.14 m）上保持平衡；Manager 工作流，无物理域随机化 |

`microduck-ball-balance` 任务配置见 `configs/task/microduck-ball-balance/motrix.fastsac.yaml`：`num_envs=2048`，`num_learning_iterations=20000`，FastSAC 异步默认开启（`algo.asynchronous=true`）。

## 球平衡环境要点（摘自官方中文文档）

- **动作：** 14 维关节位置目标，`目标 = 默认站姿 + action × 0.5`。
- **观测：** Actor 54 维（投影重力、基座角速度、球相对位姿/速度、关节状态、上一步动作 + 均匀噪声）；Critic 63 维（额外基座线速度、球世界坐标/速度特权）。
- **奖励：** `alive` + `upright` + `ball_under_feet`（核心）+ `base_height`（目标 z≈0.40 m）+ `dof_default`；惩罚 `action_rate_l2`、`limits_dof_pos`、`undesired_contacts`。
- **终止：** 基座过低、大倾角、球滚出脚底 0.20 m、关节异常；20 s 截断。
- **资产：** Microduck MJCF 自 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) 移植（Apache-2.0）；篮球为 `basketball.xml` 自由球体。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 平台总览 | `wiki/entities/motrix.md` |
| 球平衡任务 | `wiki/tasks/microduck-ball-balance.md` |
| Microduck 整机 | `wiki/entities/pollen-microduck.md` |
| 官方 mjlab 训练栈 | `wiki/entities/pollen-microduck-rl.md` |
| RL 框架导航 | `references/repos/rl-frameworks.md` |
