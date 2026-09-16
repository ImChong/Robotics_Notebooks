# SwarmNxt

> 来源归档

- **标题：** SwarmNxt
- **类型：** repo / platform / ROS 2 / Ansible / aerial swarm
- **链接：** https://github.com/lis-epfl/swarm-nxt
- **项目文档：** https://lis-epfl.github.io/swarm-nxt/
- **Stars：** ~11（2026-09-16）
- **机构：** EPFL Laboratory of Intelligent Systems（lis-epfl）
- **入库日期：** 2026-09-16
- **一句话说明：** EPFL 开源的 **OmniNxt 蜂群研究平台**：Ansible 并行部署/更新/预飞/日志回收 + ROS 2 自主栈（HDSM 规划、自适应 MPC、S2M2 深度）+ Web 仪表盘。
- **沉淀到 wiki：** [paper-swarmnxt](../../wiki/entities/paper-swarmnxt.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

**SwarmNxt** 解决「从硬件组装到多机真机实验」的工程断层：

- **硬件层：** 基于 [OmniNxt](omninxt.md)（Jetson Orin NX + 全向鱼眼 + PX4）
- **编排层：** `ansible/` 下 playbook 管理 fleet（setup / update / preflight / postflight / shutdown）
- **软件层：** `ros_packages/` — `omninxt_bridge_ros2`、`drone_state_manager_ros2`、`safety_checker_ros2`、`drone_gui_ros2`（`:8080` 仪表盘）、`latency_checker_ros2`、`swarmnxt_msgs`
- **文档：** MkDocs 站点（硬件 BOM、IT 基础设施、飞行检查单）

典型复现路径：按文档组装 OmniNxt → `ansible-playbook drone_setup.yml` → 配置 `inventory.ini` → `drones_update.yml` → 动捕 + `drones_preflight.yml` → 仪表盘单机起飞验证 → 群体任务。

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [omninxt.md](omninxt.md) | 硬件基座（HKUST IROS 2024） |
| [swarmnxt_arxiv_2609_11382.md](../papers/swarmnxt_arxiv_2609_11382.md) | 平台论文 |
| [crazyswarm2.md](crazyswarm2.md) | 同为室内动捕蜂群；SwarmNxt 算力与感知栈更重 |
| [ego_planner_swarm.md](ego_planner_swarm.md) | 均为多机规划；SwarmNxt 集成 HDSM + fleet 工具链 |
| [px4_autopilot.md](px4_autopilot.md) | 底层飞控，经 micro XRCE DDS 桥接 ROS 2 |
