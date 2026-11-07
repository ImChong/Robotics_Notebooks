# RetargetingKit 官方源码归档

- **类型：** repo / teleoperation / motion retargeting
- **链接：** https://github.com/openrobokit/retargetingkit
- **文档：** https://github.com/openrobokit/retargetingkit/blob/main/docs/concepts/pipeline.md
- **遥操作指南：** https://github.com/openrobokit/retargetingkit/blob/main/docs/guide/teleoperation.md
- **源码入口：** https://github.com/openrobokit/retargetingkit/blob/main/scripts/teleop/hand.py
- **关联项目页：** https://openrobokit.com/retargetingkit/
- **许可证：** MIT（已核对 LICENSE）
- **核查日期：** 2026-11-07
- **核对范围：** 官方 README、docs/concepts/pipeline.md、docs/guide/teleoperation.md、完整的 scripts/teleop/hand.py。
- **通读：** 已完成

## 工程事实

- Source 读取 VR、手套、动捕或录制人体帧；Retargeter 通过 RoboKit 优化或神经手模型求解；Target 向 Viser 或硬件发送命令。
- 手部命令是 `QPosCommand` 类关节位置；人形还可携带浮动基座位姿，SONIC 是独立接收控制进程。
- `scripts/teleop/hand.py` 的 `main` 用 tyro 根据 rig 构造配置；`run` 初始化 Warp、source、retargeter 及 targets，循环执行 `get_frame → solve → set_qpos`；`mode` 支持 vis、real 或 both。
- 实际遥操作所需设备依赖、校准、网络与安全必须另行检查，示例不是任意灵巧手开箱即用保证。

## 对 wiki 的映射

- [RetargetingKit](../../wiki/entities/retargetingkit.md)
