# FEMM 官方站点（femm.info）

> 来源归档（documentation site）

- **标题：** Finite Element Method Magnetics（FEMM）
- **类型：** site / docs / examples
- **链接：** https://www.femm.info/doku/doku.php?id=start
- **文档：** https://www.femm.info/doku/doku.php?id=documentation
- **示例：** https://www.femm.info/doku/doku.php?id=examples
- **下载：** https://www.femm.info/doku/doku.php?id=download
- **作者：** David C. Meeker
- **许可：** Aladdin Free Public License（程序本体）；内嵌 Triangle 网格与 Lua 4.0 各有独立许可
- **代码：** 稳定版源码 zip（随 Download 页发布，非 GitHub 主仓）  
  https://www.femm.info/doku/lib/exe/fetch.php?media=upload:files:femm42src_21apr2019.zip
- **入库日期：** 2026-07-25
- **一句话说明：** 开源 2D/轴对称有限元求解器门户：磁、静电、热流、电流场；含用户手册、脚本接口文档与大量电机/磁轴承/涡流示例。
- **开源状态：** **已开源**（Windows 二进制 + 源码 zip；无原生 Linux，官方推荐 Wine；商业使用分析结果免费，再分发程序或嵌入源码需另议许可）
- **沉淀到 wiki：** [femm](../../wiki/entities/femm.md)

---

## 入库页核查（本次三 URL）

| 页面 | URL | 用途 |
|------|-----|------|
| start | https://www.femm.info/doku/doku.php?id=start | 门户导航：Download / Documentation / FAQ / Linux / Examples / Contrib / Related Links；副题 Magnetics, Electrostatics, Heat Flow, and Current Flow |
| documentation | https://www.femm.info/doku/doku.php?id=documentation | 手册与接口索引（见下表） |
| examples | https://www.femm.info/doku/doku.php?id=examples | 官方示例目录（电感、磁轴承、永磁、涡流、感应电机、扬声器、Lua/Matlab 客户端等） |

补充打开（步骤 2.5 / FAQ）：[download](https://www.femm.info/doku/doku.php?id=download)、[faq](https://www.femm.info/doku/doku.php?id=faq)。

## 文档索引（documentation）

| 资源 | 入口 |
|------|------|
| FEMM User's Manual | https://www.femm.info/doku/lib/exe/fetch.php?media=upload:files:manual.pdf |
| Magnetics Tutorial | https://www.femm.info/wiki/MagneticsTutorial |
| Electrostatics Tutorial | https://www.femm.info/wiki/ElectrostaticsTutorial |
| Heat Flow Tutorial | https://www.femm.info/wiki/HeatFlowTutorial |
| OctaveFEMM Reference | https://www.femm.info/doku/lib/exe/fetch.php?media=upload:files:octavefemm.pdf |
| MathFEMM Reference | https://www.femm.info/Archives/doc/mathfemm.pdf |
| pyFEMM Reference | https://www.femm.info/wiki/pyFEMM/manual.pdf |
| .FEM 文件格式说明 | https://www.femm.info/Archives/contrib/FEMM_file_format.docx |
| Lua 4.0 手册 | https://www.lua.org/manual/4.0/ （另有站点 Archives PDF） |
| 用户组 | https://groups.io/g/femm/ |

## 发行与接口（download 核查）

- **定位：** Windows 上的 2D 与轴对称有限元求解器，带图形前后处理；覆盖磁、静电、热流、电流场。
- **稳定发行：** 21Apr2019（32/64-bit 安装包 + 同源 `femm42src_21apr2019.zip`）。
- **开发构建：** NewBuild 页（入库时站点记最新新构建为 22Oct2023）。
- **脚本/客户端：** 发行包含 Octave/Matlab、Scilab、Mathematica、Python（pyFEMM，亦见 PyPI）工具箱；内嵌 **Lua 4.0**。
- **平台：** FAQ 明确 **无原生 Linux**；经 Wine 支持（见 LinuxSupport）。**无 3D 版**。
- **学术引用示例（FAQ）：** D. C. Meeker, Finite Element Method Magnetics, Version 4.2, https://www.femm.info

## 示例分类摘录（examples）

面向机器人/电机学习者优先跟做（完整列表以站点为准）：

| 类别 | 示例（站点标题） |
|------|------------------|
| 入门磁路 | Inductance of a Gapped EI-Core Inductor；Permanent Magnet Example；Open Boundary / IABC |
| 力与轴承 | Eight Pole Radial Magnetic Bearing；Taper Plunger Magnet；Eddy Currents in Steel Tube |
| 电机 | Induction Motor Modeling；LRK Motor；Optimization of Outrunner BLDC；Squirrel Cage IM magneto-static（标注 NEW）；Surface Mount PM rotating losses |
| 气隙/旋转 | (Anti)Periodic Air Gap BC；Torque Benchmark；Frozen Permeability Benchmark |
| 自动化 | Lua Coil Gun；Filelink / ActiveX Client；Multiple FEMM from Matlab/Octave |
| 其它物理 | Transient Heat Flow；Loudspeaker blocked impedance / transient models |

## 对 wiki 的映射

- [FEMM](../../wiki/entities/femm.md)
- [FEMM-FOC-Simulation](../../wiki/entities/femm-foc-simulation.md)
- [PYLEECAN](../../wiki/entities/pyleecan.md)
- [Ironless QDD Actuator](../../wiki/entities/ironless-qdd-actuator.md)
- [电机电磁仿真软件选型](../../wiki/comparisons/motor-em-simulation-software.md)
- [开源力矩电机电磁设计完整度对比](../../wiki/comparisons/open-source-torque-motor-em-design.md)
- [力矩电机设计纵深](../../roadmap/depth-torque-motor-design.md)

## 推荐继续阅读（外部）

- FAQ：https://www.femm.info/doku/doku.php?id=faq
- Linux / Wine：https://www.femm.info/doku/doku.php?id=linuxsupport
- pyFEMM 页：https://www.femm.info/doku/doku.php?id=pyfemm
