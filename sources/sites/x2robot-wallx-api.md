# Wall-X API 申请与 WebSocket 推理接入（官方文档）

> 来源归档；属于现行 Wall-X 服务文档，**不是** WALL-OSS-1.0 权重发布证明。

- **标题：** Wallx API + 推理教程 / API 接口使用说明
- **类型：** 官方开发者文档
- **机构：** 自变量机器人（X Square Robot）
- **文档 URL：** <https://x2robot.com/docs/quantax1pro/v1/model-development/wallx-api/api-usage-instruction>
- **文档版本：** 量子一号 Pro 1.0 开发文档，2026-11-23 核对
- **服务地址：** `wss://maas-api.x2robot.com/v1/vla/wallx`
- **模型版本：** 握手示例返回 `wall-x`，文档未公开 1.0 或其他精确后端 checkpoint 版本
- **核对范围：** 请求链接、密钥申请、G/H 夹爪与各本体支持、同步推理示例、RTC 异步推理、Quanta X1 Pro、YAM/AgileX、Franka 及入参/出参约定
- **通读：** 已完成

## API Key 申请

- 官方申请邮箱：**wallxapi@x2robot.com**；建议邮件主题 `Wall-X API Key 申请 - 姓名 - 机构`。
- 官方要求填写：真实姓名、电话、机构、学历、职业、Google Scholar（可无）、GitHub、技术博客（可无）、测试环境（真机/仿真和平台）、测试本体。
- **不要**把网页中的演示用 `sk-...` 当作可用密钥，也不要将申请得到的密钥提交进 Git 仓库。

## 推理协议与形状

- 依赖：Python `websockets` / `msgpack` / `msgpack-numpy`；WebSocket Authorization Bearer Key；连接后先读取 MessagePack 握手帧，再发送请求。
- 典型双臂请求：`embodiment`、`normalization: provided`、`instruction`、三路同时间 JPEG（front/left_wrist/right_wrist）和左右末端 `position` / `rotation_rpy` / `gripper`。
- 末端位姿是本体坐标系的绝对量，位置米、欧拉角弧度，旋转顺序 `Rz(yaw)·Ry(pitch)·Rx(roll)`；服务内部计算相对变化。G 夹爪值 [0,1.526]，H 夹爪 [0,4.6]。
- 文档双臂动作输出长度 `action_horizon=32`，`row_dt_s=0.0337` 秒。首轮 CUDA Graph 预热可能更久，不能把预热延时误判为稳态推理延时。
- RTC：首轮不传前一段；后续传 `use_rtc=true`、`inference_delay`（1–4，桌面主从示例用 4）和 `prev_action_chunk`。桌面主从每轮执行 6 帧后重新观测；上一段需保留索引 0 作为协议哨兵，再拼接未执行部分，服务端跳过该首帧。
- 本体：TeleTwin Desktop（双臂）；Quanta X1 Pro（文档指出头/升降/底盘动作测试不足，建议先仅双臂，`return_body` 可屏蔽）；YAM、AgileX（双臂）；Franka（单右臂、前视+右腕、禁用 RTC/history、文档按 15Hz 执行）。
- 网络调用可返回模型动作，但机械安全联锁、控制周期和控制器动作限幅仍须在客户端与机器人侧实现。

## 范围与资源

- 这是可运行客户端**文档示例**，不是官方开源的服务端实现，也没有 WALL-OSS-1.0 专属源码运行时序可供核查。
- 见 [WALL-OSS-1.0 技术博客](x2robot-wall-oss-1-0.md) 与 [旧版 wall-x 仓库](../repos/wall-x.md)。

## 对 wiki 的映射

- [WALL-OSS-1.0](../../wiki/entities/wall-oss-1-0.md)（作为独立 API 资源，未混同模型版本）
