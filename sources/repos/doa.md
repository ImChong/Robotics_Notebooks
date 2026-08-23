# DOA — Drift Obstacle Avoidance

> 来源归档

- **标题：** Deep RL Based Autonomous Drift System for Abrupt Obstacle Avoidance（DOA）
- **类型：** repo
- **来源：** ustcly（中国科学技术大学相关）
- **链接：** https://github.com/ustcly/DOA
- **演示：** https://youtu.be/HBmjn-uZzoc
- **Stars：** ~1（2026-08-23）
- **许可证：** MIT
- **入库日期：** 2026-08-23
- **一句话说明：** CARLA 0.9.14 上训练/测试 DRL 漂移避障；含 expert_traj 专家示范数据集；论文标注 RA-L submitted。
- **代码：** https://github.com/ustcly/DOA（**已开源**）
- **沉淀到 wiki：** 景观页

---

## 核心定位

- **仿真：** 官方 [CARLA 0.9.14](https://carla.readthedocs.io/en/0.9.14/)（较 drift_drl 的 0.9.5 定制包更易对齐主线）
- **环境：** `conda env create -f environment.yaml` → 环境名 `carla`
- **数据：** `expert_traj/` 专家轨迹
- **测试：** `code/test.py`（需先启动 CarlaUE4）

---

## 关联

- 先驱：[`drift_drl.md`](./drift_drl.md)
- CARLA：[`carla.md`](./carla.md)
