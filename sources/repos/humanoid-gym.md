# humanoid-gym

> 来源归档

- **标题：** humanoid-gym
- **类型：** repo
- **来源：** RobotEra（roboterax）
- **链接：** <https://github.com/roboterax/humanoid-gym>
- **Stars：** ~2.0k（2026-06）
- **入库日期：** 2026-06-09
- **一句话说明：** **Humanoid-Gym** 官方开源：Isaac Gym Preview 4 上人形 **PPO locomotion** 训练栈，含 **JIT 导出、MuJoCo sim2sim** 与 XBot 真机零样本迁移文档；论文 [arXiv:2404.05695](https://arxiv.org/abs/2404.05695)。
- **沉淀到 wiki：** [`wiki/entities/humanoid-gym.md`](../../wiki/entities/humanoid-gym.md)

---

## 核心定位

人形机器人 **双足行走 RL** 的社区基线之一：在 ETH **legged_gym / rsl_rl** 工程范式上，针对人形加入 **步态相位奖励、非对称特权训练、MuJoCo 校准 sim2sim**，并以 **RobotEra XBot** 系列完成真机验证叙事。

## 依赖栈

| 组件 | 版本 / 说明 |
|------|-------------|
| Python | 3.8（conda） |
| PyTorch | 1.13.1 + cu117 |
| numpy | 1.23 |
| NVIDIA 驱动 | 推荐 525，最低 515 |
| Isaac Gym | Preview 4（`pip install -e isaacgym/python`） |
| 上游致谢 | legged_gym、rsl_rl（ETH RSL） |

## 仓库结构（要点）

```
humanoid-gym/
├── humanoid/
│   ├── envs/                 # humanoid_ppo 等任务注册
│   │   ├── legged_robot.py   # 环境基类
│   │   └── *_config.py       # 机器人/奖励/训练超参
│   ├── scripts/
│   │   ├── train.py          # PPO 训练
│   │   ├── play.py           # 评估 + 自动导出 JIT
│   │   └── sim2sim.py        # MuJoCo 部署验证
│   ├── resources/            # URDF/MJCF 资产（XBot 等）
│   └── logs/                 # checkpoint 与 exported policies
└── images/
```

## 默认任务（README）

| 任务 | 说明 |
|------|------|
| `humanoid_ppo` | 基线 PPO；观测 $47\times H$（$H$ 为历史帧数），特权 73 维 |
| `humanoid_dwl` | Denoising World Model Learning（Coming Soon） |

## 典型命令

```bash
# 训练（示例 4096 env）
python scripts/train.py --task=humanoid_ppo --run_name v1 --headless --num_envs 4096

# 评估并导出 JIT
python scripts/play.py --task=humanoid_ppo --run_name v1

# MuJoCo sim2sim（需先 play 导出 policy_*.pt）
python scripts/sim2sim.py --load_model /path/to/logs/.../policy_1.pt
```

## 扩展新机器人（README 摘要）

1. 在 `envs/` 新建 `<robot>_config.py`，继承现有配置。
2. 将 URDF/MJCF 放入 `resources/`，配置 body 名、默认关节角、PD 增益。
3. 在 `humanoid/envs/__init__.py` 注册 `task_registry.register(...)`。
4. 若需 sim2sim，检查 `sim2sim.py` 中 **MJCF↔URDF 关节映射** 与初始姿态。

## 与本仓库其他资料的关系

| 资料 | 关系 |
|------|------|
| [humanoid_gym_arxiv_2404_05695.md](../papers/humanoid_gym_arxiv_2404_05695.md) | 配套论文摘录 |
| [legged_gym.md](legged_gym.md) | 上游四足/足式训练框架；Humanoid-Gym 基于其 `LeggedRobot` |
| [humanoid-gym-modified.md](humanoid-gym-modified.md) | 社区 fork：Pandaman 模型 + Gazebo sim2sim |
| [robot_lab.md](robot_lab.md) | 新一代 Isaac Lab 扩展栈；长期选型可对照 |

## 为何值得保留

- **人形 sim2real 开源标杆**：2k+ stars，XBot 真机视频与 **MuJoCo 校准 sim2sim** 降低「无真机先验验策略」门槛。
- **工程可复用模板**：步态相位掩码、多帧观测、特权 AC、DR 表与人形 reward 拆项可直接对照 [legged_gym](../../wiki/entities/legged-gym.md) 学习差异。
- **论文笔记本深读入口**：[Humanoid-Gym Zero-Shot Sim2Real Transfer](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid-Gym_Zero-Shot_Sim2Real_Transfer/Humanoid-Gym_Zero-Shot_Sim2Real_Transfer.html) 与本页互补（单篇深读 vs 跨主题知识组织）。
