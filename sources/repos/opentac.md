# OpenTac 官方代码仓库

- **类型：** repo / 触觉传感器工具链
- **标题：** OpenTac
- **链接：** <https://github.com/opentac/opentac>
- **官方项目页：** <https://opentac.github.io/>
- **在线设计器：** <https://opentac.github.io/cad/>
- **代码许可证：** MIT（官方项目页标注）
- **硬件许可证：** CERN-OHL-P-2.0（官方项目页标注）
- **入库日期：** 2026-10-23
- **一句话说明：** OpenTac 仓库提供定制柔性触觉皮肤所需的 CAD、固件与数据工具；传感器布局 manifest 连接设计、读取、仿真、数据迁移及自适应采样环节。
- **沉淀到 wiki：** 是 → [`wiki/entities/opentac.md`](../../wiki/entities/opentac.md)

## 仓库用途

- OpenTac-CAD：从 2D 轮廓、DXF 或可展开的 STL 表面设计传感器，并输出柔性 PCB 与打印外壳相关制造文件。
- 读出固件与 Python 工具：配置读出板、读取 taxel 数据、可视化并记录触觉数据。
- OpenTac-Sim / Transfer / Xampling：用共享 manifest 连接触觉仿真、不同皮肤间的空间数据映射和自适应采样。

## 开源状态

- 官方项目页公开链接到本仓库，并提供代码与硬件的许可证信息；仓库可公开访问。
- 完成可用传感器仍需柔性 PCB、压阻材料、打印外壳与读出板等硬件，软件仓库本身不是即插即用的成品传感器。

## 对 wiki 的映射

- [OpenTac（可定制 3D 触觉皮肤生态）](../../wiki/entities/opentac.md)
- [触觉感知](../../wiki/concepts/tactile-sensing.md)
