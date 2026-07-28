# Bunny-VisionPro（Dingry/BunnyVisionPro）

> 来源归档

- **标题：** Bunny-VisionPro
- **类型：** repo
- **来源：** HKU / UC San Diego
- **链接：** <https://github.com/Dingry/BunnyVisionPro>
- **项目页：** <https://dingry.github.io/BunnyVisionPro/>
- **论文：** <https://arxiv.org/abs/2407.03162>
- **许可：** MIT
- **入库日期：** 2026-07-28
- **一句话说明：** Vision Pro 双手追踪客户端、容器化重定向服务端、XArm7+Ability Hand 真机入口及振动触觉硬件实现。
- **开源状态：** **部分开源**；可运行基础遥操作链已发布，论文安全优化模块未随主仓库完整发布。
- **沉淀到 wiki：** [`paper-notebook-bunny-visionpro-real-time-bimanual-dexterous-tel.md`](../../wiki/entities/paper-notebook-bunny-visionpro-real-time-bimanual-dexterous-tel.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 服务端 | Docker 镜像 `yzqin/bunny_teleop_server` 或 `Dingry/bunny_teleop_server` |
| 客户端 | `pip install bunny_teleop` |
| 最小示例 | `python examples/minimal/minimal.py` |
| 真机 | `real_control/`（XArm7 + Ability Hand） |
| 状态边界 | collision/singularity/collision-free retargeting 在 README 中仍为 TODO |

## 对 wiki 的映射

- 项目页：[`bunny-visionpro.md`](../sites/bunny-visionpro.md)
- 论文来源：[`humanoid_pnb_bunny-visionpro.md`](../papers/humanoid_pnb_bunny-visionpro.md)
- 遥操作路线：[`depth-teleoperation.md`](../../roadmap/depth-teleoperation.md)
