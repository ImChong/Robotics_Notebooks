# PCB 路线官方仓库 README 通读复核（2026-11-16）

- **类型：** repo / 原始技术文档复核
- **核对日期：** 2026-11-16
- **核对范围：** 下列官方仓库由 GitHub README API 返回的完整 README 正文（包括安装、示例、功能与许可节）；版本以正文 SHA-256 标识。未通读全部文档站、论文或源码，未编译及运行硬件；已有源码图仍保留原核对提交与范围。
- **通读：** 已完成

本记录只证明所列 README 已读；不能把它外推为所有链接文档或实验均已复现。

## 原文与结论

| 项目与原文 | 正文 SHA-256 | 核对结论 |
|---|---|---|
| [atopile](https://github.com/atopile/atopile#readme) | `853509b9fc392839df53d2e66a74078e9dc23cd3c6c9aa23b5eec0b90bcb47ad` | .ato 声明设计意图，编译器求解、选料和检查并更新 KiCad；KiCad 对初始构建可选，对后续布局推荐。 |
| [konnect](https://github.com/mixelpixx/Konnect#readme) | `e5611ba854999d04ab78a5902391e07924b571c4d2b45c74b87ecd51ca85f1b8` | README 顶部宣布本仓进入归档，后续 Konnect-EDA 为闭源商业产品；旧仓 AGPL 源码仍保留。原理图文件编辑、PCB IPC 与 kicad-cli 是三条不同执行路径。 |
| [part-db](https://github.com/Part-DB/Part-DB-server#readme) | `03d68200052ead73925f212cab05488e54b3eb81f636c30dc00f653614159806` | Web 服务管理元件、库存、权限、BOM 与 KiCad 数据源。安装需 PHP/Web 服务与数据库，前端构建需 Node 22+；数据库迁移前备份。 |
| [qucs-s](https://github.com/ra3xdh/qucs_s#readme) | `885500911ea2d8182e1667fdf2aeea80ed47bf72b739d98863fba6462f1c3b3c` | GUI 与数值后端分离，推荐 Ngspice；Qt5 已自 v25.1.0 移除，Qt6 为默认；安装 GUI 并不自动满足后端依赖。 |
| [opentac](https://github.com/opentac/opentac#readme) | `ef2850a44f4ddfce84b68117ddea977c1d5b603f9c3f5577490bdb739097d357` | 共享 manifest 贯穿 CAD、读出、仿真与 transfer；uv run opentac cad 启动设计器，flash/setup/view 分别烧录、配置与读取。硬件与代码许可证分开。 |
| [pcb-motor](https://github.com/ziteh/pcb-motor#readme) | `da188461b544e5a70cb3b04ac6177b22febc9fce96f8e41443c4ad02d358ca20` | README 对槽极比与五类绕组进行文献比较；20 极、六层、140 μm、1 mm 气隙是所引研究[3]的参数，不应当作本仓实测成品指标。 |
| [wolfiemouse](https://github.com/kbumsik/WolfieMouse#readme) | `3b9881dc6829112c22d949a1d05d333ff772c135ec1f37353e3671e577cdd4db` | 固件、桌面仿真与采集工具分目录，GNU Arm/Make/OpenOCD 是固件构建部署路径；第三方库保留各自许可证。 |
| [ukmarsbot](https://github.com/ukmars/ukmarsbot#readme) | `cfbe33f31cda33f99e9d26b443e5823177e31b582a80747c15487415804e850f` | 主仓负责硬件、制造文件、机械与文档，软件在姊妹仓；Nano 是默认处理器，供电与传感器可选，部分硬件交付文件可能不完整。 |
| [moteus](https://github.com/mjbots/moteus#readme) | `b30f2ca3d31b4b9751bdcb14c151b8652efcb01d7d2db57aebc989297959a7be` | 公开固件与多板型 PCB；规格表明确区分无散热、冷却及峰值相电流。hw/fw/lib/utils 分别对应硬件、固件、客户端与诊断。 |
| [tinymovr](https://github.com/motionlayer/Tinymovr#readme) | `f11b5198b0c3cb9c46c3f2dd65cfdc4f537114137752839f1dd050443cddfe2d` | 公开代码停留于 v3.0.0；v3.1 起仅专有二进制。固件、硬件、文档为 MIT，Studio Python 与整体项目 GPLv3，须按目录判断。 |
| [vesc](https://github.com/vedderb/bldc#readme) | `d9cb6f5d853516132fa6b322667333dd35f76a6ff68996370ddc1875111a67dd` | make 列举板型，先安装 ARM SDK 再按板型构建；SWD 与 USB/VESC Tool 两条烧录路径。固件 GPLv3，硬件在单独仓库，本次未核对该硬件仓。 |
| [simplefoc](https://github.com/simplefoc/Arduino-FOC#readme) | `f0560d47db4d359874d6afcdbf8810ac76e1da0a9c353bf44a575a98e6a2aa16` | Arduino/PlatformIO 组合 motor、sensor、driver、current sense；README 完整示例在 setup 初始化并 initFOC，loop 中先 loopFOC 再 move。估算电流模式与真实采样模式须区分。 |
| [tscircuit](https://github.com/tscircuit/tscircuit#readme) | `816864ac4c228f53abb7363a6f1dbf814d0c222eca32007f2c11680a4676a937` | TypeScript/React 描述电路，tsci dev 预览并可导出制造文件；core、CLI 和 viewers 分项目。README 明确建议指定 pcbX/pcbY，自动布局与路由仍在发展。 |
| [serial-studio](https://github.com/Serial-Studio/Serial-Studio#readme) | `a9da466418ecc74176d9054433f76c6b4703211f11bafa74658f17bd48fc81fa` | Quick Plot 可自动识别数值行；GPL 构建仅免费功能，官方二进制与 Pro 权限不同。README 当前构建基线为 Qt 6.11.2；GUI 遥测不是实时控制环。 |

## 对 wiki 的映射

- [atopile](../../wiki/entities/atopile.md)
- [konnect](../../wiki/entities/konnect.md)
- [part-db](../../wiki/entities/part-db.md)
- [qucs-s](../../wiki/entities/qucs-s.md)
- [opentac](../../wiki/entities/opentac.md)
- [pcb-motor](../../wiki/entities/pcb-motor.md)
- [wolfiemouse](../../wiki/entities/wolfiemouse.md)
- [ukmarsbot](../../wiki/entities/ukmarsbot.md)
- [moteus](../../wiki/entities/moteus.md)
- [tinymovr](../../wiki/entities/tinymovr.md)
- [vesc](../../wiki/entities/vesc.md)
- [simplefoc](../../wiki/entities/simplefoc.md)
- [tscircuit](../../wiki/entities/tscircuit.md)
- [serial-studio](../../wiki/entities/serial-studio.md)
