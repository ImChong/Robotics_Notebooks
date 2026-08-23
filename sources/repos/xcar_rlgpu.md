# xcar-rlgpu

> 来源归档

- **标题：** XCAR RL Drift Control（xcar-rlgpu）
- **类型：** repo
- **来源：** zhou-yh19（GitHub）
- **链接：** https://github.com/zhou-yh19/xcar-rlgpu
- **Stars：** ~33（2026-08-23）
- **许可证：** MIT
- **入库日期：** 2026-08-23
- **一句话说明：** PyTorch GPU 向量化单车动力学 + rl_games，训练独立轮驱（IWD）自主漂移策略；含域随机化、多轨迹漂移任务与 Sim2Real 导向分析工具。
- **代码：** https://github.com/zhou-yh19/xcar-rlgpu（**已开源**）
- **沉淀到 wiki：** 是 → [`wiki/entities/xcar-rlgpu.md`](../../wiki/entities/xcar-rlgpu.md)

---

## 核心定位

- **动力学：** Individual Wheel Drive（IWD）模型，四轮独立速度与扭矩控制
- **训练：** GPU 并行向量化环境；子模块 **rl_games**（Denys88 fork）
- **任务：** 圆环、变曲率、八字等漂移轨迹族
- **工程：** `environment.yml` + CUDA 11.8+ / PyTorch 2.0.1+

---

## 典型入口

| 路径 / 命令 | 用途 |
|-------------|------|
| `conda env create -f environment.yml` | 环境 |
| `git submodule update --init` | 拉取 rl_games |
| 训练脚本（见 README Training 节） | PPO 等基线漂移策略 |

---

## 关联档案

- 景观策展：[`racing_drift_rl_open_source_landscape.md`](../papers/racing_drift_rl_open_source_landscape.md)
- 对照：[f1tenth_gym.md](./f1tenth_gym.md)、[gym_khana.md](./gym_khana.md)
