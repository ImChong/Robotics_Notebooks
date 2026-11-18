# Panthera-HT_lerobot：高擎机械臂的 LeRobot 适配

- **类型：** repo / 官方软件集成
- **机构：** 高擎动力（HighTorque Robotics）
- **链接：** https://github.com/HighTorque-Robotics/Panthera-HT_lerobot
- **项目页：** https://hightorque.cn/Panthera-HT_Hub/
- **入库日期：** 2026-11-18
- **核对范围：** 通读官方 README 中英使用说明和运行命令；阅读 Panthera-HT_robot_lerobot/src/lerobot_robot_panthera/ 下的 panthera_leader.py（connect/get_action）、panthera_follower.py（connect/get_observation/send_action）、motors/panthera_motors_bus.py（connect/sync_read/sync_write）相关实现。
- **通读：** 已完成
- **对应 wiki：** [Panthera-HT 项目总览](../../wiki/entities/cn-os-panthera-ht-main.md)

## 项目适配边界

- 仓库将 LeRobot 框架源码与独立 Panthera-HT_robot_lerobot 适配包并列放置。
- panthera_leader 提供主臂关节位置动作，含重力/摩擦补偿线程；panthera_follower 对接关节和相机观测、执行从臂动作；panthera_dual_follower 用于双臂任务扩展。
- 底层 PantheraMotorsBus 使用 Panthera-HT_SDK 的 Panthera / hightorque_robot 控制对象；独立安装 LeRobot 不等于已安装电机驱动。
- 官方建议 Python 3.10、旁置 Panthera-HT_SDK、安装 RealSense 驱动；配置默认关联 SDK 的 Follower.yaml / Leader.yaml。

## 官方最短执行路径

1. 环境安装：pip install -e lerobot；pip install -e Panthera-HT_robot_lerobot；另装 pynput、pin、pyrealsense2。
2. 采集：在 lerobot/src/lerobot/scripts 中运行 lerobot_record.py，参数 robot.type=panthera_follower、teleop.type=panthera_leader、dataset.repo_id=local/panthera_demo。
3. 训练：lerobot_train.py，指定 dataset.repo_id=local/panthera_demo、policy.type=act。
4. 评估：lerobot_record.py，指定 policy.path 指向训练出的 ACT checkpoint，并另存评估 episodes。

## 局限与注意

- 上述是 README 给出的可运行命令与源码入口，**本次未连机械臂、未实测数据采集频率或任务成功率**。
- 在 leader 源码里 send_feedback() 为 no-op；**不可声称已经实现双边力反馈遥操作**。
- 默认相机型号、序列号和参数文件路径带硬件/实验环境假设；需要按自己的实物替换。
- README 的旧文案称尚无 LICENSE，但仓库当前目录列有 LICENSE；以当前仓库实际文件和适用范围为准。
