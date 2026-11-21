# RoboColosseum：Robot Client、Policy Server 与 Fine-tune 工具

> 来源归档（repo）

- **类型：** repo
- **代码：** https://github.com/RoboColosseum/RoboColosseum
- **机器人 Client：** https://github.com/RoboColosseum/robocolosseum-client
- **策略 Server：** https://github.com/RoboColosseum/robocolosseum-policy-server
- **Franka / DROID Fine-tune：** https://github.com/RoboColosseum/robocolosseum-finetune-franka
- **G1 Fine-tune：** https://github.com/RoboColosseum/robocolosseum-finetune-g1
- **YAM Fine-tune：** https://github.com/RoboColosseum/robocolosseum-finetune-yam
- **SO-101 Fine-tune：** https://github.com/RoboColosseum/robocolosseum-finetune-so101
- **核对日期：** 2026-11-21
- **核对范围：** 阅读官方汇总 README、Client README、Policy Server README，并核对 Client 源码 src/colosseum_client/client.py、robot_runner.py 及 Policy Server 源码 src/colosseum_policy_server/server.py、sdk.py 的观测封装、WebSocket/Protobuf 配对、推理和 action chunk 数据流。未实际连接真机或托管 Router；不宣称全套代码实测运行。
- **通读：** 已完成
- **关联实体：** [RoboColosseum](../../wiki/entities/paper-robocolosseum.md)

## 真实入口和模块分工

| 模块 | 证实入口 | 职责 |
|---|---|---|
| Robot Client | README: uv run --extra <robot_type> colosseum-robot configs/robot.yaml | 操作员录入任务、场景试验，录制评价及导出 |
| 网络通信 | colosseum_client/client.py: ColosseumClient.connect/infer | 以 Protobuf RelayFrame 通过 WebSocket 发送 HELLO / MATCH_REQUEST / OBSERVATION，接收 ACTION_PLAN |
| 观测与动作执行 | colosseum_client/robot_runner.py: protobuf_observation/run_robot/action_chunk | RGB 图像、关节/夹爪状态封装，chunk 维度、step、截止时间检查与机器指令执行 |
| Policy Server | colosseum_policy_server/server.py: PolicyServer.run/_handle | 注册服务、接收 OBSERVATION、调用 policy.infer 并回发 ACTION_PLAN |
| 自定义策略接入 | colosseum_policy_server/sdk.py 与 README: ColosseumPolicySDK.get_obs/send_action | 用户把图像+状态+instruction 送入自有模型，返回 action horizon×dim |
| 结果发布 | Client README / lerobot_export.py / hub_upload.py | 生成 LeRobot 结构、视频与元数据并上送 HF |

## 工程复现要点与边界

- Python 3.10+、uv；Client 录像导出需 FFmpeg；本地测试可先使用 policy server 的 colosseum-policy-test-backend 验证收发而不加载模型。
- 官方 policy server README 的最小自定义循环为 with ColosseumPolicySDK.from_yaml(...) as sdk → sdk.get_obs() → model.infer(obs) → sdk.send_action(actions, observation=obs)。输出为 NumPy 的 (horizon, action_dim) chunk，SDK 检查维度、有效数值、deadline 和长度。
- 官方 SDK 默认 joint_position、15 Hz、最大 16-step chunk；这些是默认配置而非所有本体统一的测量频率，部署时须与 robot spec 对齐。
- 客户端 Franka 机器人底层示例包含 protobuf_observation 和 execute_robot_action；YAM 有 GUI 和实际驱动文档；Client README 对 G1 adapter 仍标 TODO，**不能因仓库有目录就断言 G1 Client 已完整公开可运行**。
- Router 服务端和真实机器人连接/现场安全操作还需项目团队权限，不可把本地 SDK 测试写成在线真机评测已复现。

## 对应站点

- [官网归档](../sites/robocolosseum.md)
- [论文原文归档](../papers/robocolosseum_arxiv_2611_02497.md)