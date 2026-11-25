# SANA-Robot Code | 训练与部署源码归档

- **类型：** repo
- **代码：** https://github.com/NVlabs/Sana/tree/robot
- **固定提交：** [d92b529](https://github.com/NVlabs/Sana/tree/d92b52903598acc5e03d1ab0d7c7d1d46609b955)
- **代码许可：** Apache-2.0；权重、Gemma 文本编码器、训练数据及外部基准各自检查许可和访问要求。
- **模型：** [AR Pretrain](https://huggingface.co/Efficient-Large-Model/SANA-Robot-AR-Pretrain)，及官方集合的 RoboDojo、RoboTwin 2.0、LIBERO 权重。README 训练配方、三个模拟评测与双臂 YAM 示例可查；不能据此认为 23 份原始数据或所有论文实验专用权重均附仓。
- **项目：** https://nvlabs.github.io/Sana/Robot/
- **核对范围：** 官方 README 的安装、权重、数据、微调、评测、部署与许可；追踪 train shell、RoboTwin 评测入口、deploy README、PolicyServer、ChunkRunner、MemoryRTCController、PhaseMemory、ARSession 和低位推理入口。静态检查，不执行 GPU/真机。
- **通读：** 已完成
- **入库日期：** 2026-11-25
- **对应节点：** [SANA-Robot](../../wiki/entities/paper-sana-robot-2611-28943.md)

| 使用层 | 官方入口 | 实际职责 |
|---|---|---|
| 数据与微调 | `scripts/data/convert_*.sh` → `scripts/train/train_ar.sh configs/robotwin_ar.yaml` | 转统一数据行，加载 AR 权重与任务配置后微调 |
| 基准 | `eval/{robodojo,robotwin,libero}/README.md` | 分开配置环境、评测端和策略端 |
| 部署入口 | `scripts/deploy/server.sh` → `sana_robot.deploy.server`，机器人端 `deploy/client.py` 或 `deploy/yam/client.py` | GPU 主机持有策略，机器人侧读取状态/视频并执行动作；控制端模板不提供任意机器人所需的安全限幅和停机 |
| 分块 | `PolicyServer.handle` → `Deployment.runner`、`ChunkRunner` | `start/push_frame/next_plan`，将本体状态转 robot80，再把动作转换回本体命令 |
| 相位与生成 | `MemoryRTCController` → `PhaseMemory.commit_window/generate` → `ARSession.generate` | 保存/切换观测历史（含视频流状态），前缀预填后联合动作/视频去噪 |
