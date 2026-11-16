# elephantrobotics/myAGV（大象机器人移动机器人底座文档与 ROS 使用指南）

> 来源归档

- **标题：** elephantrobotics/myAGV
- **类型：** repo + docs
- **机构：** 大象机器人（Elephant Robotics）× 华蔚联合开发
- **版本 / 仓库：** `https://github.com/elephantrobotics/myAGV`（`main` 分支）
- **官方文档：** <https://docs.elephantrobotics.com/docs/gitbook-cn/2-myAGV/README.html>（`myagv_docs.txt`）
- **核对范围：** 核查 `elephantrobotics/myAGV` 仓库全部文件（`README.md`、`Summary.md`、`1-introduce/初次使用.md`、`1-introduce/VNC.md`、`2-development/{3.MYAGV小车建图.md,4.MYAGV小车地图导航.md,5.MYAGV小车PS4手柄控制.md,8.MYAGV连接屏幕.md,小车机械臂联动.md}`、`3-compound/{6.MYAGV摄像头精准定位.md,7.MYAGV与机械臂连接.md}`）以及大象机器人官方 GitBook 文档 `2 myAGV` 章节（`myagv_docs.txt`）。
- **通读：** 已完成
- **沉淀到 wiki：** [`wiki/entities/elephantrobotics-myagv.md`](../../wiki/entities/elephantrobotics-myagv.md)

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|------|------|
| GitHub 仓库 `elephantrobotics/myAGV` | **文档/操作手册仓库（非直接 C++/ROS 源码包）** — 仓内主要包含中文 GitBook Markdown 文档与工程截图，记录机载 ROS 工作空间启动指令与节点交互流程 |
| 机载 ROS 工作空间说明 | 文档明确指出机载树莓派 4B 内置 ROS 工作空间 `/home/ubuntu/Documents/aibot_ws/`，包含 `navigation_demo`、`ds4_driver`、`mycobot_ros` 等功能包及 `aibot_detect_marker.py`、`reach_pump.py` 等脚本 |
| 官方文档与社区镜像 | GitBook 在线中文手册完整公开参数、硬件引脚图、建图/导航/手柄/复合抓取操作流程；实际运行时依赖官方出厂 TF 卡系统镜像与机载驱动包 |

## 仓库与官方文档核心要点编译

1. **硬件平台与接口（GitBook + `README.md`）：**
   - 紧凑型全向移动底盘（GitBook 表记 `311×236×130 mm`、重 `3.6 kg`、额定最大负载 `2 kg`、树莓派 4B 2GB、霍尔编码器电机、最大速度 `0.9 m/s`、360° 全向激光雷达测距 `8 m`、500 万像素前置摄像头、12V2A 充电；`README.md` 早版页面表记 `300×205×145 mm` / `3.02 kg` / `24V 1A`）。
   - 机身配有 XT30UPB-F 12V 5A 机械臂供电口（用于驱动 myCobot 等复合机械臂）、树莓派 14 针 GPIO/2.54mm 杜邦 IO 组、HDMI-D 口、USB 2.0、以太网口及车身双侧扩展电池仓。
2. **ROS 建图、导航与遥操作命令链（`2-development/*.md`）：**
   - **SLAM 建图（`3.MYAGV小车建图.md`）：** `roslaunch navigation_demo myagv_slam.launch` 启动 IMU、轮速里程计（`/odom`）与激光建图节点（支持 Gmapping 与 Cartographer 两种 SLAM 方案）及 RViz；配合 `rosrun teleop_twist_keyboard teleop_twist_keyboard.py` 遥控建图后，由 `rosrun map_server map_saver` 生成 `map.pgm` 与 `map.yaml`。
   - **地图导航（`4.MYAGV小车地图导航.md`）：** 将地图复制至 `aibot_ws/src/aibot/navigation_demo/maps/`，修改 `myagv_navigation.launch`（文档第 3 节命令行亦有拼写为 `myagv_navigetion.launch`）加载地图后，通过 RViz `2D Nav Goal` 发送目标点自主避障导航。
   - **PS4 蓝牙手柄控制（`5.MYAGV小车PS4手柄控制.md`）：** 通过 `hciattach /dev/ttyAMA0 bcm43xx` 与 `bluetoothctl` 配对后，并行启动 `roslaunch navigation_demo myagv_navigation.launch` 与 `roslaunch ds4_driver ds4_twist.launch`。
3. **复合机器人视觉对准与移动抓取（`3-compound/6.MYAGV摄像头精准定位.md`、`2-development/小车机械臂联动.md`）：**
   - **双 ArUco 精密停靠：** 目标台两侧等距粘贴 `6×6`、`50 mm` 字典的 ArUco 标记（ID `3` 与 `4`），在 `myagv_navigation.launch` 运行基础上执行 `rosrun navigation_demo aibot_detect_marker.py`（建议初始距目标 `>40 cm` 使双码同时入画）实现视觉微调停靠。
   - **车-臂协同取放：** 依次启动 `myagv_navigation.launch`、`roslaunch mycobot_ros mycobot_detect_marker_with_topic.launch` 与 `python2 reach_pump.py`（位于 `Thirtyparty/mycobot_ros/scripts`），节点 `my_xy_pub` 顺序调度 `Movement.goto_target(1, 0)` → `aibot_adjust()` → `Controller.get_cargo()`（手眼 ArUco 识别 + 吸泵抓取放到车身背板）→ `Movement.goto_target(0, 0)`。
