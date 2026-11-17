# allenai/MolmoBot 官方模型、训练和真机推理仓

- **仓库：** https://github.com/allenai/MolmoBot
- **数据生成引擎：** https://github.com/allenai/molmospaces
- **论文：** https://arxiv.org/abs/2603.16861
- **项目：** https://allenai.github.io/MolmoBot/
- **核查日期：** 2026-11-17
- **核对范围：** 官方根 README、MolmoBot/README.md（安装、Franka / RB-Y1 训练与部署）、MolmoBot/launch_scripts/serve_molmo.py（含 main、load_molmo、WebsocketPolicyServer）、robot_eval/scripts/droid/run_policy.py（含 main、get_obs、PolicyClient.infer、RobotInterface / GripperInterface 执行）、molmospaces README（数据生成与评测入口）。未运行大型依赖环境。
- **通读：** 已完成
- **开源结论：** 已提供可辨识训练/推理/数据生成及真机执行入口；代码可能持续更新，复现应固定 commit、环境和 checkpoint。

## 实际源码入口与运行边界

1. **训练：** `MolmoBot/launch_scripts/train_molmobot.py` 由官方 README 用 `torchrun --nproc-per-node=8` 启动，输入解包/校验后的 MolmoBot-Data、统计量、已下载并 unshard 的 Molmo2-4B checkpoint；`--action_preset franka_joint`、`--camera_preset franka_one_random_then_wrist`、`--model.num_flow_timestamps=8`。README 给出的实验配方使用 8–64 H100，不能认定普通单卡可复现实验成绩。
2. **模型服务：** `MolmoBot/launch_scripts/serve_molmo.py --hf-repo allenai/MolmoBot-DROID`，代码 `download_model_from_hf → load_molmo → RealRobotVLAPolicy → WebsocketPolicyServer.serve_forever()`；服务端绑定 8000 端口。
3. **Franka 真机客户端：** `robot_eval/scripts/droid/run_policy.py`（Hydra 配置）从 `RobotInterface` / `GripperInterface` 和相机收集图像、qpos，`PolicyClient.infer(obs)` 请求服务，按 `action_scale` 对绝对关节位置插值后调用 `robot.update_desired_joint_positions`，夹爪变化时用 `gripper.goto(..., blocking=False)`。
4. **仿真评测：** `MolmoBot/launch_scripts/run_eval.py --checkpoint_path ... --benchmark_path ... --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig`。
5. **数据生成：** `molmospaces` README 列出 `molmo_spaces/data_generation/main.py`、`molmo_spaces/evaluation/eval_main.py` 与 `scripts/datagen/run_pipeline.py --seed 1`；演示路径为可调试入口，不保证直接生成论文相同完整数据分布。

## 对知识库的映射

- [MolmoB0T 论文归档](../papers/molmobot_arxiv_2603_16861.md)
- [项目入口核查](../sites/molmobot-ai2.md)
- [数据集归档](../datasets/allenai-molmobot-data.md)
- [论文独立节点](../../wiki/entities/paper-molmobot.md)
