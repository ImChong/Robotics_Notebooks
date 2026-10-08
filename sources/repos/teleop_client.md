# teleop_client

> 来源归档（国内具身开源全景）

- **标题：** teleop_client
- **类型：** repo
- **机构：** 星动纪元
- **链接：** https://github.com/roboterax/teleop_client
- **分类：** 遥操作与数据采集
- **入库日期：** 2026-09-06
- **一句话说明：** 星动纪元 开源项目 teleop_client（遥操作与数据采集），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/cn-os-teleop-client.md`](../../wiki/entities/cn-os-teleop-client.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/cn-os-teleop-client.md](../../wiki/entities/cn-os-teleop-client.md)

## 官方 README 补核（2026-10-08）

- **README：** https://github.com/roboterax/teleop_client/blob/main/README.md

README 命令序列为 `start_sdk → init_teleop → start_teleop → stop_teleop → stop_sdk`。初始化包含授权文件 `--verify`、XHAND/Lite、VR/gamepad、dummy/realsense/stereo 选择。开发环境还引用厂商 GitLab 的 `rbclient` 和 `pub_client.py`；公开仓入口不保证服务端、recording 实现和授权环境全部可公开复现。SDK README 另引用此仓构建消息定义。未确认单一首发日期，未连接真机。
