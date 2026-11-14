# GitHub 仓库：robo-arena/roboarena

- **仓库：** https://github.com/robo-arena/roboarena
- **关联项目：** RoboArena（DROID-RoboArena Distributed Evaluation）
- **开源状态：** 已开源（包含政策提交服务端接口 `WebsocketPolicyServer`、客户端协议 `WebsocketClientPolicy`、接口格式一致性检测脚本 `test_policy_server.py`，并对接 DROID 仿真预检 `sim-evals` 与 `openpi` 训练指南）
- **核对范围：** 已通读官方仓库 `roboarena` 中的 `README.md`、`pyproject.toml`、`roboarena/policy.py`、`roboarena/policy_server.py`、`roboarena/policy_client.py` 与 `scripts/test_policy_server.py`
- **通读：** 已完成

## 代码结构与核心模块

1. **统一策略抽象与通信契约（`roboarena/policy.py` & `PolicyServerConfig`）**
   - `BasePolicy` 定义仅需实现的两个标准方法：`infer(obs: Dict) -> Dict` 与 `reset(reset_info: Dict) -> None`（按 `session_id` 清空有状态策略历史）。
   - `PolicyServerConfig` dataclass 在 WebSocket 握手首包由服务端主动下发给客户端，按需声明客户端图像分辨率（默认 `(224, 224)`）、是否发送手腕图 `needs_wrist_camera`、外部相机路数 `n_external_cameras in {0, 1, 2}`、是否双目 `needs_stereo_camera`、是否携带 `needs_session_id` 以及动作空间枚举（`joint_position`、`joint_velocity`、`cartesian_position`、`cartesian_velocity`），最小化跨广域网传输载荷。

2. **异步 WebSocket 服务端与客户端实现（`policy_server.py` & `policy_client.py`）**
   - `WebsocketPolicyServer` 基于 `websockets` + `msgpack_numpy` 在默认 `0.0.0.0:8000` 启动异步事件循环：建立连接后首先打包发送 `PolicyServerConfig`，随后依请求字典 `endpoint` 字段分发至 `reset` 或 `infer`，异常时回传完整栈信息并关闭连接。
   - `WebsocketClientPolicy` 源自 `Physical-Intelligence/openpi`，支持 `ws://` 自动回退 `wss://` 隧道建连，实现客户端观测上传与返回二维 `actions` 块（`(N, 8)` 关节+夹爪或 `(N, 7)` 笛卡尔+夹爪，整块一次发往真机执行再发起下一轮请求）。

3. **准入接口校验与生态协同（`scripts/test_policy_server.py`）**
   - `test_policy_server.py` 作为论文所述安全防线第一关：连接策略服务端后校验 `PolicyServerConfig` 字段合法性，依据配置生成合规哑观测 `_make_dummy_observation` 发起一次 `infer` 断言返回 ndarray 形状匹配动作空间（`joint` 末维为 `8`、笛卡尔为 `7`），最后校验 `reset` 接口可用性。
   - `README.md` 指引提交者先通过 `arhanjain/sim-evals` 在仿真环境下检验动作合理性，再挂接公网端口提交真实机器人网络评估。
