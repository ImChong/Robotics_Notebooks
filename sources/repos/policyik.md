# PolicyIK 官方代码仓库

> 一手 GitHub 仓库核验；运行路径根据仓库 README、执行脚本与相关源码逐项对照。

- **标题：** PolicyIK: High-Precision Inverse Kinematics for Robot Arms with Reinforcement Learning
- **类型：** repo / reinforcement-learning / inverse-kinematics
- **链接：** <https://github.com/Tinaliu0123/PolicyIK>
- **项目页：** <https://tinaliu0123.github.io/PolicyIK/>
- **arXiv：** <https://arxiv.org/abs/2611.05372>
- **入库日期：** 2026-11-08（源码入口复核）
- **开源状态：** 已开源（仿真、真机 OpenArm、checkpoint、数据）
- **一句话说明：** 同一个 6D 目标位姿控制思路有 PPO/Isaac Lab 仿真路线和 FlashSAC/HER/ROS 2 真机路线。
- **核对范围：** 主 README、hardware/README.md；仿真训练与评测 shell、policyik_sim.train/evaluate、tasks/actions.py、tasks/mdp.py、tasks/env_cfg.py；真机训练与评测 shell、policyik_hw.train/evaluate、envs/pose_env.py、robot/ros_io.py、rl/learner_hooks.py、FlashSAC 的 lan_critic.py。
- **通读：** 已完成
- **沉淀到 wiki：** [PolicyIK 统一实体](../../wiki/entities/paper-policyik.md)

## 仿真运行链（公开代码核验）

1. simulation/scripts/train.sh 运行 python -m policyik_sim.train，以 Isaac Lab 注册任务 PolicyIK-OpenArm 训练 4,096 并行环境，PPO 分三段到 13,000 epochs。
2. policyik_sim.tasks.env_cfg 配置多尺度姿态误差、进度与保持奖励；tasks.actions.AccumulatedJointReferenceAction 把动作乘以误差相关步长，积分为关节位置参考。
3. policyik_sim.average_checkpoints 对近期 checkpoint 作指数加权平均，仓库在 simulation/checkpoints/ 提供三个发布权重。
4. simulation/scripts/evaluate.sh 先以 policyik_sim.evaluate 做单次测试，再通过 corrected_goals 修正目标，并分别进行多种 transient-goal 重试；policyik_sim.metrics 汇总，区分一次执行与部署重试的成功率。
5. 数据包含 4,120 个 eval_tasks，训练关节构型库和目标池；末端位姿在底座系中表达，四元数顺序 w,x,y,z。

## 真实 OpenArm 运行链（公开代码核验）

1. hardware/README.md 要求 Ubuntu 24.04 / ROS 2 Jazzy；ros2/openarm_right_bringup 的 launch 启动控制器，robot/ros_io.ArmIO 经 /joint_states 读取状态、向 /forward_position_controller/commands 写 7 关节位置。
2. hardware/scripts/train.sh 调用 python3 -m policyik_hw.train；构造 envs.pose_env.OpenArmPoseEnv、FlashSACAgent、PoseHERRLPDBuffer，加载 22,500 条示范并用 HER 目标重新标注训练。
3. envs.pose_env.OpenArmPoseEnv.step 执行误差增益缩放、关节参考积分、关节位置指令下发与传感反馈；训练总计 120,000 步；支持 checkpoint、buffer 和故障 watchdog。
4. hardware/scripts/evaluate.sh 以三个位置位移区间各 20 个任务启动 policyik_hw.evaluate，最终调用 policyik_hw.metrics 汇总 60 个任务；评测实际使用 --no-collision-guard。
5. 真机安装和安全约束依赖 CAN 总线、gravity_node.py 的重力矢量、单 controller_manager、故障监测；**不可把评测参数直接视为安全可部署配置**。

## 复现与开源边界

- 源码/权重：有仿真发布权重和评估数据，真实 OpenArm 有运行/训练/评测代码及示范数据；公开范围按实际目录核对。
- xArm7：论文报道了扩展实验，但本次未核实到仓库提供等价的 xArm7 一键复现入口；不要将论文结果误读为已完整发布对应硬件驱动。
- 若仅阅读此索引仍不能确保物理平台可直接复现，实际运行前检查 mounting、重力补偿、手臂关节界限、周围环境、安全策略。

## 参考来源

- [官方仓库 README](https://github.com/Tinaliu0123/PolicyIK/blob/main/README.md)
- [真机复现说明](https://github.com/Tinaliu0123/PolicyIK/blob/main/hardware/README.md)
- [仿真动作积分实现](https://github.com/Tinaliu0123/PolicyIK/blob/main/simulation/policyik_sim/tasks/actions.py)
- [真机环境实现](https://github.com/Tinaliu0123/PolicyIK/blob/main/hardware/policyik_hw/envs/pose_env.py)
