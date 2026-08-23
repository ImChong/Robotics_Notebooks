# drift_drl

> 来源归档

- **标题：** High-speed Autonomous Drifting with Deep Reinforcement Learning
- **类型：** repo
- **来源：** caipeide（GitHub）；论文 ICRA 2020 / RA-L
- **链接：** https://github.com/caipeide/drift_drl
- **论文：** https://arxiv.org/abs/2001.01377
- **项目页：** https://sites.google.com/view/autonomous-drifting-with-drl → [`sources/sites/drift_drl_google_sites.md`](../sites/drift_drl_google_sites.md)
- **Stars：** ~140（2026-08-23）
- **许可证：** MIT
- **入库日期：** 2026-08-23
- **一句话说明：** 经典 CARLA 0.9.5 定制仿真 + 深度 RL 高速漂移控制器；七张地图分阶段训练，含参考轨迹与评测脚本。
- **代码：** https://github.com/caipeide/drift_drl（**已开源**）
- **数据/仿真：** 需下载作者提供的 CARLA 0.9.5 build（Google Drive）；非官方 CARLA 主线版本
- **沉淀到 wiki：** 是 → [`wiki/entities/drift-drl.md`](../../wiki/entities/drift-drl.md)

---

## 核心定位

- **仿真：** 基于 CARLA 0.9.5 的定制 build（README 提供 Drive 链接）
- **训练：** `conda env create -f environment_drift.yaml` → 环境名 `drift`
- **地图：** 7 张（a–g）；`code/ref_trajectory` 存参考轨迹（traj_0…traj_6 分阶段/评测）
- **硬件：** 论文期 NVIDIA GPU（GTX 1080Ti 测试）

---

## 典型入口

| 步骤 | 说明 |
|------|------|
| 安装 CARLA build | README「Start the Simulator」 |
| `conda activate drift` | Python 依赖 |
| 启动仿真 + 训练/测试脚本 | 见 `code/` |

---

## 关联档案

- 项目页核查：[`drift_drl_google_sites.md`](../sites/drift_drl_google_sites.md)
- CARLA 主线：[`carla.md`](./carla.md)、[`wiki/entities/carla.md`](../../wiki/entities/carla.md)
- 后继 CARLA 漂移：[`doa.md`](./doa.md)
