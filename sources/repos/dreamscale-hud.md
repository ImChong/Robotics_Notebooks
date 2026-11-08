# Dreamscale × HUD — 官方评测适配仓库与真实调用链

- **类型：** 官方公开集成代码（并非云端服务后端）
- **源码：** https://github.com/Dreamscale-Labs/dreamscale-hud
- **官方 README：** https://github.com/Dreamscale-Labs/dreamscale-hud/blob/main/README.md
- **协议：** MIT（该仓库集成代码；上游模型权重、仿真资产及第三方依赖遵循各自许可，详见 THIRD_PARTY.md）
- **核查日期：** 2026-11-08
- **核对范围：** 阅读 README 完整的实机 YAM/SO-101、LIBERO、RoboLab/RoboTwin/RoboDojo、Python agent 及验证说明；审阅 `src/dreamscale_hud/cli.py` 中 `evaluate` / `tasks` / `job_runtime`、`agent.py` 中 `DreamscaleRobotAgent` / `DreamscaleModel.ainfer`、`adapter.py` 中 `LiberoAdapter`、`contract.py` 中数据形状和控制频率合同。不声称已审完整个仓库或云端未公开服务代码。
- **通读：** 已完成
- **实体：** [Dreamscale Labs](../../wiki/entities/dreamscale-labs.md)

## 公开可运行入口

```bash
# 实机双臂 YAM + HUD 轨迹记录
uv tool install dreamscale
dreamscale login
dreamscale robots yam setup
dreamscale doctor yam
dreamscale integrations hud setup
dreamscale run yam --model molmoact2-bimanual-yam --hud --hud-job "MolmoAct2-BimanualYAM"

# 仿真：一局本地 LIBERO + HUD
uv sync
uv run dreamscale login
uv run hud login
docker build -f Dockerfile.hud -t dreamscale-hud-libero:local .
uv run dreamscale-hud --task-ids 0 --init-state-ids 0

# 静态代码质量检查（尚未在本次环境实际运行）
uv run ruff check .
uv run pytest -q
```

HUD 的每条 task trace 可以记录指令、相机视频、关节/动作与 grade；真实机器人 rollout 需要有实际机械臂、相机、Linux 控制机和付费/可用的云端连接。

## 已审阅的源码实际数据/调用链

1. `cli.py:evaluate` 创建 `DockerRuntime`、`HUDRuntime` 或既有 Runtime，构造 `Taskset`，通过 `job_runtime` 管理共享仿真会话。
2. 进入 `async with DreamscaleRobotAgent(...)`：`agent.py:__aenter__` 创建一个 _connect task；`_connect` 调用公开 `dreamscale.aconnect(model=MODEL, rtc="off", calibration="off", control_hz=..., keep_warm=0)`，校验动作 Hz 与 chunk 长度。
3. 每个 HUD rollout 经 `DreamscaleRobotAgent.__call__` 取得环境协议合同，确认 `control_rate` 与目标一致，等待服务会话 ready，随后调用 HUD RobotAgent 的执行循环。
4. `LiberoAdapter.adapt_observation` 校验 256×256×3 RGB uint8 双相机、8 维 state，把观测经 `dreamscale.libero.observe` 封装；`DreamscaleModel.ainfer` 调用 `policy.predict`，检查动作 chunk 形状并写 telemetry；`adapt_chunk` / `adapt_action` 将动作映射回 HUD。
5. `Taskset(...).run(agent,...)` 结束后汇总 reward、trace、timings，关闭会话。该适配案例按源码 `max_concurrent=1`，不可将官方博客 128 仿真并发宣称套到**该单个 agent 实例**。

## 模型/版本边界（重要）

- 这个仓库的 `contract.py` 固定 `MODEL="molmoact2-libero"`、`CONTROL_HZ=20`、`CHUNK_SIZE=10`，即评测适配器中的 pinned 合同；**不等于**官网目前共享模型列表为每个组织公开可用，也不应擅自替换 model ID。
- 该仓库适配的是 HUD 环境侧与 Dreamscale **公开 SDK 接口**；并不包括云端模型权重/serving backend 的完整实现。
- README 的 RoboTwin/RoboDojo 示例明确指出包含 hold-pose agent；不是已证明这些基准可调用同一 Dreamscale policy 成功执行。
- 评测费用、模型限制、机器可用性与真机安全条件需分别检查；未在本次会话启动 GPU 环境、下载仿真镜像或执行真机 rollout。

## 运行时序图依据

在 [Dreamscale 实体页](../../wiki/entities/dreamscale-labs.md) 中仅绘制 README + 上述已核对文件的 LIBERO/HUD 评测调用链；**不伪造未开放的 Dreamscale 服务器内部模块**。

## 链接

- [官方文档摘录](../sites/dreamscale-labs.md)
- [官方技术博客通读](../blogs/dreamscale_cloud_robot_inference_2026.md)
