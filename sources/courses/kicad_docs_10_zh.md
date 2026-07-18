# KiCad 10.0 官方文档（简体中文）

> 来源归档

- **标题：** KiCad Documentation — 10.0 简体中文
- **类型：** course（官方用户手册 / 教程文档集）
- **来源：** KiCad 项目文档站点
- **链接：** https://docs.kicad.org/10.0/zh/
- **英文对照：** https://docs.kicad.org/10.0/en/
- **PDF：** 各章节页面提供 Alternate formats → PDF 下载
- **入库日期：** 2026-07-18
- **一句话说明：** KiCad **10.0** 各子程序的 **简体中文官方手册**：入门、原理图、PCB、Gerber 检视、图框编辑器、PCB 计算器与 **kicad-cli** 命令行；机器人硬件团队做驱动板/BMS/转接板时的 **标准操作参考**。
- **沉淀到 wiki：** [KiCad](../../wiki/entities/kicad.md)

---

## 文档目录（10.0 / zh 首页索引）

| 手册 | 对应程序 | 机器人相关用途 |
|------|----------|----------------|
| **KiCad 入门** | 全流程 | 项目结构、库管理、首次打样路径 |
| **KiCad** | 项目管理器 | 工程、版本、库表配置 |
| **原理图编辑器** | Eeschema | 电机驱动、BMS、传感前端原理图；ERC |
| **PCB 编辑器** | Pcbnew | 功率环路布局、电流采样走线、散热铜皮 |
| **Gerber 浏览器** | Gerbview | 发板前 CAM 检视 |
| **图框编辑器** | 标题栏/图框 | 团队图纸规范 |
| **PCB 计算器** | 工程计算 | 走线宽度、过孔载流、阻抗（打样前自检） |
| **KiCad 命令行界面** | kicad-cli | CI 导出 Gerber、批处理 DRC |

另见英文文档中的 **SPICE 仿真**、**设计规则**、**3D 模型** 等章节（中文版以站点实际目录为准）。

---

## 为什么值得保留

- 本知识库已覆盖 **FOC 电流环**（[SimpleFOC](../../wiki/entities/simplefoc.md)）、**底软总线**（[motor-drive-firmware-bus-protocols](../../wiki/overview/motor-drive-firmware-bus-protocols.md)）与 **PCB 在整机中的角色**（[Hardware 101 · 05](../../wiki/overview/humanoid-hardware-101-power-compute-electronics.md)），但缺少 **开源 EDA 工具链** 的系统入口。
- KiCad 10 与稳定版 10.0.x 对齐；中文手册降低国内团队与课程门槛。
- **kicad-cli** 章节支持把「原理图 → Gerber → DRC」纳入 Git 仓库自动化，适合开源硬件协作。

---

## 核心摘录（对 wiki 的映射）

### 1) 原理图 → PCB → 制造 闭环

- **要点：** 同一工程内符号-封装-网表-布局-3D-制造文件一致；ERC/DRC 在发板前捕获开路、短路与间距违规。
- **对 wiki 的映射：** [KiCad 实体页](../../wiki/entities/kicad.md)「核心结构/机制」

### 2) 电机驱动板布局约束

- **要点：** PCB 手册中的 **铺铜、过孔、差分/阻抗、热焊盘** 与 Stage 4「功率环路面积最小化、采样开尔文接法」直接对应；PCB 计算器辅助载流与温升粗算。
- **对 wiki 的映射：** [力矩电机设计纵深 Stage 4](../../roadmap/depth-torque-motor-design.md)

### 3) 命令行与协作

- **要点：** `kicad-cli` 可在无 GUI 环境导出制造数据，便于 PR 中附带 Gerber 变更证据。
- **对 wiki 的映射：** [KiCad](../../wiki/entities/kicad.md)「工程实践」

---

## 推荐继续阅读（外部）

- [KiCad 文档首页（版本选择）](https://docs.kicad.org/)
- [KiCad 开发者文档](https://dev-docs.kicad.org/)
- [KiCad 入门（10.0 zh PDF）](https://docs.kicad.org/10.0/zh/getting_started/) — 以站点当前路径为准
