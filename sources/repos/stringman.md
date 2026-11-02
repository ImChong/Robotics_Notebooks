# cranebot3-firmware

- **类型：** 官方代码仓库
- **链接：** https://github.com/nhnifong/cranebot3-firmware
- **核对版本：** [79f2b695f885e2b09a0a7d1e0fac3546431e9bc1](https://github.com/nhnifong/cranebot3-firmware/tree/79f2b695f885e2b09a0a7d1e0fac3546431e9bc1)
- **许可证：** Apache-2.0（软件仓库；硬件许可需分别核实）
- **项目页：** https://neufangled.com/stringman
- **入库日期：** 2026-11-01
- **对应实体：** [Stringman](../../wiki/entities/stringman.md)

## 已核对的代码入口

- `pyproject.toml` 定义 `stringman-headless = nf_robot.host.observer:main`、`component-server = nf_robot.robot.i2c_dispatcher:main` 及 `mcp` 可选依赖。
- `src/nf_robot/host/observer.py` 中 `--mcp` 注册 `McpBridge`；`--mcp_allow_code` 控制代码工具。
- `src/nf_robot/host/mcp_bridge/bridge.py` 暴露状态、图像、运动、抓取、结果等待、停止等工具；运动任务异步返回，使用 `wait` 查询结果。
- `src/nf_robot/host/mcp_bridge/GUIDE.md` 说明操作、安全约束及实际抓取局限。

仓库公开可读；本次没有运行或连接真机。控制器到设备的完整低层调用路径尚需进一步逐函数核查。
