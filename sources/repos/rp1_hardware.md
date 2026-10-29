# rp1_hardware（RP1 机械 / PCB / 线束 / BOM 制造资料）

> 来源归档

- **标题：** RP1 Hardware — Hardware design and manufacturing files for the RoboParty RP1 humanoid robot
- **类型：** repo（硬件设计与制造文件）
- **机构：** 萝卜派对（RoboParty）
- **链接：** https://github.com/Roboparty/rp1_hardware
- **入库日期：** 2026-10-27（核查快照：`main` @ `c595061`，2026-10-26 "Add 46 missing 3D printed part PDF drawings"）
- **一句话说明：** RP1.3.4 整机的 SolidWorks/Parasolid 机械模型、CNC / 3D 打印 / 橡胶制造图、14 种 PCB 生产资料、42 种线束图纸与两种配置整机 BOM。
- **沉淀到 wiki：** 是（[`wiki/entities/rp1-hardware.md`](../../wiki/entities/rp1-hardware.md)）
- **上级目录：** [RP1 总览仓库归档](roboparty_rp1.md)

## 开源状态

- **已开源（设计 + 制造文件）**：仓库内直接提供 PDF 图纸、STEP、Gerber、CPL、BOM xlsx、线束图；SolidWorks 源 / Parasolid / 可编辑工程图以 Release `v0.1.0` 附件（`RP1_SolidWorks_Source.zip`、`RP1_Parasolid_Model.zip`、`RP1_Drawings.zip`）分发，本次未下载附件，未核对其内容。
- **许可：** RoboParty 自有设计 CERN-OHL-W-2.0；文档 CC BY-SA 4.0；第三方文件保留原条款（`LICENSES.md`）。
- 仓库无 `.gitattributes`，文件为实体而非 LFS 指针。

## 关键文件清单

| 路径 | 内容 |
|------|------|
| `00_Docs/RP1_Hardware_Architecture.pdf` | 硬件系统架构（48V + 30V 双电源、RK3588S 主控、5×CANFD） |
| `01_Mechanic/02_Manufacturing/{01_CNC,02_3D_Printing,03_Rubber}` | 三类工艺的加工 BOM、图纸、STEP、CMF 规格表（CNC 5DOF 8 个 / 7DOF 18 个 / Common 200 个条目） |
| `02_PCB/01…14_*` | 14 种板卡，料号 `YCL.E000xxx`，各含 01_BOM / 02_Gerber / 03_3D / 04_CPL / 05_PCBA加工要求 / 06_PCB_PDF |
| `03_Harness/` | `RP1_Harness_BOM.xlsx`（42 种线束）、`Drawings/` 43 份 PDF、`线束制作规范.docx` |
| `04_BOM/` | `RP1.3.4 （五自由度+运控头）BOM表.xlsx`、`RP1.3.4 （七自由度+运控头）BOM表.xlsx` |
| `LICENSE` / `LICENSES.md` | CERN-OHL-W-2.0 与许可范围映射 |

## 对 wiki 的映射

- [wiki/entities/rp1-hardware.md](../../wiki/entities/rp1-hardware.md)
- 关联：[roboparty-rp1](../../wiki/entities/roboparty-rp1.md)、[romomo](../../wiki/entities/romomo.md)、[rp1-test](../../wiki/entities/rp1-test.md)
