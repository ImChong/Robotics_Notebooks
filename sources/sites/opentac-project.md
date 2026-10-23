# OpenTac 官方项目页

- **类型：** 官方项目页 / 技术文档
- **标题：** OpenTac: An Open Ecosystem for Customizable, Scalable, and Transferable 3D Tactile Sensing
- **链接：** <https://opentac.github.io/>
- **设计工具：** <https://opentac.github.io/cad/>
- **论文页：** <https://opentac.github.io/paper/>
- **代码：** <https://github.com/opentac/opentac>
- **机构：** Johns Hopkins University；Massachusetts Institute of Technology
- **入库日期：** 2026-10-23
- **一句话说明：** OpenTac 是用于机器人表面定制化触觉皮肤的开放生态，覆盖传感器 CAD、制造、读取、仿真、跨传感器数据迁移和自适应采样。
- **沉淀到 wiki：** 是 → [`wiki/entities/opentac.md`](../../wiki/entities/opentac.md)

## 官方技术摘录

- 每片皮肤由柔性 PCB 上的压阻 taxel 阵列、压阻薄膜和可打印外壳构成；剪切感知版本在每四个 taxel 上方布置一个柔性凸点。法向按压产生近似共模响应，切向载荷使凸点倾斜并产生差分响应，用于估计两个方向的剪切分量。
- OpenTac-CAD 可从 2D 轮廓、DXF 或可展开的 STL 表面生成传感器布局、走线、制造文件和传感器 manifest。
- OpenTac-Sim、OpenTac-Transfer、OpenTac-Xampling 分别支持传感器仿真、不同布局间的数据映射和自适应采样。
- 官方报告：1024-taxel 传感器全阵列 raster 扫描为 67 frames/s；自适应 Xampling 在每次更新读取 16 个 taxel 时达到 1286 partial frames/s。该值是部分采样更新率，不代表每帧读取完整阵列。

## 开放范围

- 官方代码仓库公开，项目页标注代码采用 MIT、硬件采用 CERN-OHL-P-2.0、文档与媒体采用 CC BY 4.0。
- OpenTac-CAD 在线工具可用，也可下载代码在本地运行；制造还需要按生成的 Gerber 文件制作柔性 PCB，并打印弹性凸点外壳。

## 对 wiki 的映射

- [OpenTac（可定制 3D 触觉皮肤生态）](../../wiki/entities/opentac.md)
- [触觉感知](../../wiki/concepts/tactile-sensing.md)

## 参考

- [OpenTac 官方项目页](https://opentac.github.io/)
- [OpenTac 论文页](https://opentac.github.io/paper/)
- [OpenTac-CAD](https://opentac.github.io/cad/)
- [官方代码仓库](https://github.com/opentac/opentac)
