# Low Cost Mocap（for drones）

> 来源归档（ingest）

- **标题：** Low Cost Mocap
- **类型：** repo
- **来源：** Joshua Bird（GitHub）
- **链接：** <https://github.com/jyjblrd/Low-Cost-Mocap>
- **原仓库链接：** <https://github.com/jyjblrd/Mocap-Drones>（现重定向至上述仓库）
- **许可证：** MIT
- **入库日期：** 2026-10-19
- **项目说明：** 面向房间尺度多目标定位的低成本光学动捕系统，用于室内无人机跟踪与自主飞行。
- **开源状态：** 已开源，仓库包含 `computer_code/`、`sender_esp32/`、`receiver_esp32/` 与 `3d_files/`。README 提供启动说明，但作者指出文档尚不完善。
- **关联项目页：** <https://joshuabird.com/blog/post/mocap-drones>
- **沉淀到 wiki：** [Low Cost Mocap 实体页](../../wiki/entities/low-cost-mocap.md)

## 仓库结构与运行入口

| 路径 | 作用 |
|------|------|
| `computer_code/` | 相机采集、视觉定位计算与 Web 界面 |
| `sender_esp32/` | 向无人机发送控制数据的 ESP32 固件 |
| `receiver_esp32/` | 无人机侧接收控制数据的 ESP32 固件 |
| `3d_files/` | 相机外壳、安装件等可打印文件 |

README 的启动步骤为：安装 Node 依赖并执行 `yarn run dev` 启动界面；另开终端运行 `python3 api/index.py` 启动处理相机流的后端。项目依赖 pseyepy 与包含 SFM 模块的 OpenCV。

## 项目关系

- [项目博客与技术摘录](../sites/joshuabird-low-cost-mocap.md)
- [Low Cost Mocap 实体页](../../wiki/entities/low-cost-mocap.md)
