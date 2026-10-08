# xbot_sdk_api

> 来源归档（国内具身开源全景）

- **标题：** xbot_sdk_api
- **类型：** repo
- **机构：** 星动纪元
- **链接：** https://github.com/roboterax/xbot_sdk_api
- **分类：** SDK/驱动
- **入库日期：** 2026-09-06
- **一句话说明：** 星动纪元 开源项目 xbot_sdk_api（SDK/驱动），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/cn-os-xbot-sdk-api.md`](../../wiki/entities/cn-os-xbot-sdk-api.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/cn-os-xbot-sdk-api.md](../../wiki/entities/cn-os-xbot-sdk-api.md)

## 官方 README 补核（2026-10-08）

- **README：** https://github.com/roboterax/xbot_sdk_api/blob/main/README.md

公开 Python ROS 2 API，类入口为 `RobotController`、`TrajectoryController`、`MPCController`，覆盖关节状态、轨迹、ServoPose、XHAND 与底盘命令。README 要求厂商 developer 环境或 ROS 2 Humble + CycloneDDS，构建 `teleop_client` 消息，设置 `ROS_DOMAIN_ID=211`。

示例轨迹机型支持 Q5 / L3 / L7。API 封装公开不等于下层 MPC 求解器、控制服务和完整运控策略均在该仓实现；本轮未做真机调用。未确认单一首发日期。
