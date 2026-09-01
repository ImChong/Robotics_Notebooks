# 赛车漂移强化学习开源景观（仓库策展）

- **标题：** Racing / Drift RL Open-Source Landscape
- **类型：** repo-collection
- **来源：** 用户指定 GitHub 仓库列表 + 仓库 README / 项目页核查（2026-08-23）
- **入库日期：** 2026-08-23
- **一句话说明：** 按仿真后端、控制范式与硬件栈梳理赛车/漂移方向 RL、MPC、F1TENTH 与浏览器纽北驾驶引擎的开源入口，服务选型与复现路径判断。

---

## 策展清单与 wiki 映射

| # | 仓库 | 范式 | 仿真/硬件 | 开源结论 | wiki / source |
|---|------|------|-----------|----------|---------------|
| 1 | [zhou-yh19/xcar-rlgpu](../repos/xcar_rlgpu.md) | GPU 并行 RL（IWD 漂移） | 自研向量化动力学 + rl_games | **已开源**（MIT） | [`wiki/entities/xcar-rlgpu.md`](../../wiki/entities/xcar-rlgpu.md) |
| 2 | [caipeide/drift_drl](../repos/drift_drl.md) | 深度 RL 高速漂移（ICRA 2020） | CARLA 0.9.5 定制 build | **已开源**（MIT；需自备 CARLA 包） | [`wiki/entities/drift-drl.md`](../../wiki/entities/drift-drl.md) |
| 3 | [mlab-upenn/LearningMPC](../repos/learning_mpc.md) | 学习 MPC（LMPC）在线迭代 | F1/10 ROS racecar_simulator | **已开源**（代码在仓；依赖 UPenn 仿真栈） | 景观页 |
| 4 | [TeoIlie/Gym-Khana](../repos/gym_khana.md) | SB3 + 课程学习漂移/竞速 | f1tenth_gym | **已开源**（MIT；PyPI `gymkhana`） | 景观页 |
| 5 | [MPC-Berkeley/barc](../repos/barc.md) | 漂移/换道/避障全栈 | 1/10 RC + ROS + 云数据 | **已开源**（MIT） | [`wiki/entities/barc.md`](../../wiki/entities/barc.md) |
| 6 | [ustcly/DOA](../repos/doa.md) | DRL 突发障碍漂移避障 | CARLA 0.9.14 | **已开源**（MIT；含 expert_traj） | 景观页 |
| 7 | [Gelminaio/drift-mpc-ackermann](../repos/drift_mpc_ackermann.md) | 非线性 MPC + 在线摩擦估计 | 1:10 Ackermann + ROS 2 | **已开源**（MIT） | 景观页 |
| 8 | [f1tenth/f1tenth_gym](../repos/f1tenth_gym.md) | 1/10 竞速 Gym 仿真 | 纯 Python 单车动力学 | **已开源**（MIT） | [`wiki/entities/f1tenth-gym.md`](../../wiki/entities/f1tenth-gym.md) |
| 9 | [UoA-CARES/autonomous_f1tenth](../repos/autonomous_f1tenth.md) | CARES RL + Gazebo Garden | F1TENTH 真机/仿真 | **已开源**（仓内可运行；依赖较多） | 景观页 |
| 10 | [carla-simulator/carla](../repos/carla.md) | 城市驾驶仿真平台 | UE 城市场景 | **已开源**（MIT） | [`wiki/entities/carla.md`](../../wiki/entities/carla.md) |

### 补充：浏览器纽北 / 赛道驾驶引擎（2026-08-23）

| # | 仓库 | 范式 | 运行形态 | 开源结论 | wiki / source |
|---|------|------|----------|----------|---------------|
| 11 | [esc5221/drive-game](../repos/drive_game.md) | 240 Hz Pacejka 模拟器 | Web + Android；OSM/DEM 真几何 | **已开源**（MIT；可本地 build） | [`wiki/entities/drive-game.md`](../../wiki/entities/drive-game.md) |
| 12 | [yassinsolim/nordschleife-racer](../repos/nordschleife_racer.md) | arcade-sim + 多人 | 引擎 TS 模块；线上 yassin.app | **部分开源**（引擎 MIT；GLB/Supabase 未入库） | [`wiki/entities/nordschleife-racer.md`](../../wiki/entities/nordschleife-racer.md) |
| 13 | [mrdoob/starter-kit-racing](../repos/starter_kit_racing.md) | Godot→JS 街机移植 | GridMap + crashcat；CDN 零构建 | **已开源**（MIT；Kenney CC0 资产） | [`wiki/entities/starter-kit-racing.md`](../../wiki/entities/starter-kit-racing.md) |

---

## 交叉主题（写回 wiki）

- **总览地图：** [`wiki/overview/racing-drift-rl-open-source-landscape.md`](../../wiki/overview/racing-drift-rl-open-source-landscape.md)
- **MPC 方法对照：** [`wiki/methods/model-predictive-control.md`](../../wiki/methods/model-predictive-control.md)
- **RL 训练栈：** [`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md)
- **仿真平台十年图：** [`wiki/overview/sim-platforms-decade-technology-map.md`](../../wiki/overview/sim-platforms-decade-technology-map.md)
