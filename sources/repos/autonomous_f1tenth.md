# autonomous_f1tenth

> 来源归档

- **标题：** Autonomous F1TENTH（UoA CARES）
- **类型：** repo
- **来源：** UoA-CARES（奥克兰大学）
- **链接：** https://github.com/UoA-CARES/autonomous_f1tenth
- **Stars：** ~22（2026-08-23）
- **入库日期：** 2026-08-23
- **一句话说明：** 基于 Gazebo Garden + ROS 2 Humble 的 F1TENTH RL 栈：集成 [cares_reinforcement_learning](https://github.com/UoA-CARES/cares_reinforcement_learning) 与 forked gz-sim，支持仿真与真车（urg_node）。
- **代码：** https://github.com/UoA-CARES/autonomous_f1tenth（**已开源**；仓内无 license 文件，以组织惯例为准）
- **沉淀到 wiki：** 景观页

---

## 依赖

| 组件 | 版本/链接 |
|------|-----------|
| Gazebo | Garden（源码构建；UoA fork `gz-sim`） |
| ROS 2 | Humble |
| CARES RL | github.com/UoA-CARES/cares_reinforcement_learning |
| F1TENTH | github.com/UoA-CARES/f1tenth |

---

## 安装要点

```bash
git clone --recurse-submodules https://github.com/UoA-CARES/autonomous_f1tenth.git
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```

---

## 关联

- 轻量仿真对照：[`f1tenth_gym.md`](./f1tenth_gym.md)
- 景观：[`racing_drift_rl_open_source_landscape.md`](../papers/racing_drift_rl_open_source_landscape.md)
