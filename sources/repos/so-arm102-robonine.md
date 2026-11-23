# SO-ARM 102（Robonine）开源硬件、URDF 与 LeRobot 插件
- **类型：** repo / robot arm hardware / teleoperation
- **硬件仓库：** https://github.com/roboninecom/SO-ARM102
- **独立软件插件：** https://github.com/roboninecom/SO-ARM-102-Lerobot
- **资料：** models/3mf、models/stl、models/step、docs/assembly-guide.md、docs/bom.md、simulation/urdf/
- **许可证：** 硬件 CERN-OHL-P-2.0；软件 Apache-2.0；文档 CC BY 4.0
- **核对范围：** 硬件 README、docs/lerobot.md、独立插件 README 的舵机设置/校准/遥操作命令和性能测试条件。
- **通读：** 已完成
- **入库日期：** 2026-11-23

## 核心摘录
- 单臂 5 DoF + 夹爪，可配一对或双对领从臂，兼容 LeRobot。
- 领端 5V，从端 12V；平行夹爪 85 mm。
- 300g 刚度对照中不同机械臂使用不同打印材料；额定负载和空载速度为官方测试或规格，不是独立验证。
- 插件 README 用 0.1.0a3，硬件 docs/lerobot.md 用 0.1.0a1，版本有差异，需锁定版本。

映射：[独立实体](../../wiki/entities/so-arm102-robonine.md)和[遥操作路线](../../roadmap/depth-teleoperation.md)。
