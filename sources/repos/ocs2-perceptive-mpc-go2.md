# OCS2 Perceptive Nonlinear-MPC on the Unitree Go2（源码适配）

> 来源归档：项目作者公开代码及使用文档

- **标题：** OCS2 Perceptive Nonlinear-MPC on the Unitree Go2
- **类型：** repo
- **仓库：** https://github.com/Khaledsweid1/ocs2-perceptive-mpc-go2
- **代码：** https://github.com/Khaledsweid1/ocs2-perceptive-mpc-go2
- **许可：** 本仓原创的 `patches/`、`tools/`、`docs/` 为 Apache-2.0；被 patch 的上游项目保留原许可
- **项目身份：** Unitree Go2 上 ROS 2 Jazzy + MuJoCo DDS 的 OCS2 感知型 NMPC 工程适配；**不是**另一个独立发表的 NMPC 算法
- **原始算法：** [Grandia et al. 2023（DOI:10.1109/TRO.2023.3275384）](https://doi.org/10.1109/TRO.2023.3275384)；实验机器人是 ANYmal
- **上游：** [OCS2](https://github.com/leggedrobotics/ocs2)、[legubiao/quadruped_ros2_control](https://github.com/legubiao/quadruped_ros2_control)、[legubiao/ocs2_ros2](https://github.com/legubiao/ocs2_ros2)、[grid_map](https://github.com/ANYbotics/grid_map)
- **核对范围：** 项目 README 全文、`docs/PERCEPTIVE_MPC_GUIDE.md` 的部署/现状/感知章节、`scripts/setup.sh`、`tools/launch/perceptive_mujoco.launch.py`、`tools/launch/perception_viz.launch.py`、`tools/terrain_elevation_pub.py`、`tools/drive_ocs2.py`、`tools/capture_perception.py`、`tools/config/perceptive_convex.yaml` 文件入口；未在本环境执行 colcon、MuJoCo 或控制闭环
- **通读：** 已完成
- **入库日期：** 2026-11-14
- **开源状态：** 已公开可重建工作区的补丁、启动脚本和感知辅助工具；需要下载上游依赖并编译
- **映射：** [项目实体节点](../../wiki/entities/ocs2-perceptive-mpc-go2.md)

## 官方运行入口（README / launch 已核对）

```bash
git clone https://github.com/Khaledsweid1/ocs2-perceptive-mpc-go2.git
cd ocs2-perceptive-mpc-go2
./scripts/setup.sh ~/ocs2_ws
cd ~/ocs2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash

# 先运行 unitree_mujoco 仿真：场景文件需由使用者自行准备
DISPLAY=:1 ./unitree_mujoco -r go2 -s scene_rough_varied.xml
# 再另开终端：
ros2 launch ~/ocs2_ws/perceptive_mujoco.launch.py scene:=/abs/path/to/scene_rough_varied.xml rviz:=true
# 再另开终端：
python3 ~/ocs2_ws/drive_ocs2.py --stand=8 --speed=0.4
```

- `setup.sh` 按 `patches/UPSTREAM_COMMITS.txt` 锁定三个上游仓库提交后施加 patch、复制工具，不负责一键完整构建 MuJoCo。
- `perceptive_mujoco.launch.py` 启动 `ros2_control_node`、`ocs2_quadruped_controller`、合成高程图节点、凸平面分解及 RViz；按启动事件顺序加载状态广播与控制器。
- `terrain_elevation_pub.py` 的启动参数缺省 5.0 m、0.04 m、5 Hz；launch 明确覆盖 `--size=6.0`，即 150 × 150 栅格；地图发布在 `/elevation_mapping/elevation_map_raw`。
- `drive_ocs2.py` 在 20 Hz 向 `/control_input` 发布站立/小跑指令并基于 `/odom` 实现横向保持。
- `capture_perception.py` 订阅高程、分割、边界与里程计数据，落盘 NPZ 供 `plot_perception.py` 绘图；可独立验证感知链路。

## 工程边界

1. `fp:=0` 为 launch 默认配置（落足/碰撞实验开关未开启）；使用 `fp:=1`、`fp:=fc`、`fp:=both` 为不同实验选项，不应视为已稳定可用。
2. 项目指南写明：**落足区域约束默认关闭（over-constrain Go2）**；本配置能站立、小跑、适应部分粗糙地形，但**不能完整爬完楼梯**。因此 README 对“engage rough terrain”的表述不等于所有硬约束功能已验证。
3. 地形来自 MuJoCo 射线采样的模拟几何，与真机激光/深度传感器的遮挡、噪声、时延、漂移不同；README 仅提出可扩展至真机，并未提供本项目真机验收数据。
4. 原文附件里的耗时/可视化取自作者所述记录；本次未独立基准测量或完整复现。
