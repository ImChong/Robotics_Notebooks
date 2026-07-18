# Altium Designer 一手资料索引

> 来源归档（ingest）

- **标题：** Altium Designer 官方技术文档与 QuickStart 体系
- **类型：** site / course（官方文档 + 入门教程）
- **来源：** Altium（Altium Pty Ltd）
- **主入口：** <https://www.altium.com/documentation/>
- **Altium Designer 文档根：** <https://www.altium.com/documentation/altium-designer>
- **入库日期：** 2026-07-18
- **一句话说明：** 商业级统一 EDA 环境（原理图 → PCB → 制造输出 → ECAD-MCAD 协同）的官方一手文档；2026 年起文档不再按版本分册，与 Altium Designer / Develop / Agile 平台共用在线文档集。
- **沉淀到 wiki：** 是 → [wiki/entities/altium-designer.md](../../wiki/entities/altium-designer.md)

## 为什么值得保留

- 机器人自研关节驱动板、传感器载板、电源分配板等 **板级硬件** 在 [力矩电机纵深路线 Stage 4](../../roadmap/depth-torque-motor-design.md) 中需要走完 **原理图 → layout → 制造数据** 闭环；Altium Designer 是工业/科研团队常用的 **商业 PCB EDA** 之一。
- 与 [SimpleFOC 官方板卡](../repos/simplefoc_arduino_foc.md)（EasyEDA 开源）等 **开源参考设计** 互补：后者适合低功率学习与 BOM 复刻；Altium 文档覆盖 **设计规则、层叠、制造发布、机械协同** 等量产向流程。
- 一手资料强调 **原理图为设计主记录（master record）**、**ECO 双向同步**、**规则驱动 layout**——与「只改 PCB 不回传原理图」的作坊式流程形成对照。

## 文档体系（官方结构）

| 层级 | 链接 | 用途 |
|------|------|------|
| 文档门户总览 | <https://www.altium.com/documentation/readme> | 搜索、F1 上下文帮助、QuickStart 索引、Legacy 离线包说明 |
| Altium Designer 主页 | <https://www.altium.com/documentation/altium-designer> | 统一设计环境、Workspace、ECAD-MCAD、QuickStart 分页入口 |
| 完整入门教程 | <https://www.altium.com/documentation/altium-designer/tutorial> | 从空白原理图到制造输出的九器件示例（astable multivibrator） |
| 发布说明 | <https://www.altium.com/documentation/altium-designer/public-release-notes> | AD 26.x / Develop / Agile 公开版本变更 |
| 新功能摘要 | <https://www.altium.com/documentation/altium-designer/new> | 各小版本功能与 BugCrunch 修复索引 |
| 旧版离线文档 | <https://www.altium.com/documentation/other_installers> | AD 25 及更早 HTML 快照 Zip（在线文档自 AD 26 起不再按版本分册） |

**上下文帮助：** 在软件内对菜单/对话框/面板按 `F1` 跳转对应文档；交互命令中按 `Shift+F1` 查看快捷键列表（见 [Using Altium Documentation](https://www.altium.com/documentation/readme)）。

## QuickStart 分页（官方推荐学习路径）

| 主题 | 官方页面 | 机器人硬件相关要点 |
|------|----------|-------------------|
| 设计环境 | [Getting Familiar with the Altium Design Environment](https://www.altium.com/documentation/altium-designer/getting-familiar-with-the-altium-design-environment) | Projects 面板、统一编辑器范式 |
| 原理图捕获 | [Capturing Your Design Idea as a Schematic](https://www.altium.com/documentation/altium-designer/schematic) | 元件放置、连线、电源端口、多页/层次化、标注与 Validate |
| 电路仿真 | [Analyzing Your Design Using Circuit Simulation](https://www.altium.com/documentation/altium-designer/simulation) | 驱动级拓扑与滤波预研（可选） |
| BOM | [BOM Management with ActiveBOM](https://www.altium.com/documentation/altium-designer/activebom) | 供应链与替代料 |
| 原理图↔PCB 变更 | [Managing Design Changes between the Schematic & PCB](https://www.altium.com/documentation/altium-designer/sch-pcb) | ECO 流程、Constraint Manager 与 Rules 双轨 |
| PCB layout | [Laying Out Your PCB](https://www.altium.com/documentation/altium-designer/pcb) | 层叠、设计规则、布线、铺铜 |
| 制造图 | [Streamlining Board Design Documentation with Draftsman](https://www.altium.com/documentation/altium-designer/draftsman) | 装配/制造图 |
| 制造输出 | [Preparing Your Design for Manufacture](https://www.altium.com/documentation/altium-designer/preparing-for-manufacture) | OutJob、Gerber/NC Drill、项目发布 |
| 多板 | [Designing with Multiple PCBs](https://www.altium.com/documentation/altium-designer/multi-board) | 多板装配与机壳空间 |
| 线束 | [Harness Design](https://www.altium.com/documentation/altium-designer/harness) | 机器人线束拓扑（高阶） |

## 核心摘录

### 1) 统一设计环境与平台形态

- **来源：** [Altium Designer Documentation](https://www.altium.com/documentation/altium-designer)（更新 2026-07-15）
- **要点：**
  - **Unified Design Environment**：原理图、PCB、仿真、BOM、3D、发布在同一环境完成，数据在 realm 间无缝流转。
  - **平台变体：** 通过 Altium Develop 提供 **Altium Designer Develop**，通过 Altium Agile 提供 **Altium Designer Agile**；文档统称 **Altium Designer**。
  - **许可：** 现有 term license 在期内可继续获更新；到期后新功能需迁移至平台方案（Develop / Agile）。
  - **Workspace：** 可连接 Altium 365 Workspace 或 On-Prem Enterprise Server，管理元件、版本化发布与协作。
- **对 wiki 的映射：** [altium-designer](../../wiki/entities/altium-designer.md)

### 2) 原理图捕获与电源网络

- **来源：** [Capturing Your Design Idea as a Schematic](https://www.altium.com/documentation/altium-designer/schematic)、[Creating Circuit Connectivity](https://www.altium.com/documentation/altium-designer/schematic/creating-circuit-connectivity)、[Working with Directives](https://www.altium.com/documentation/altium-designer/schematic/directives)
- **要点：**
  - **Power Port** 定义全局电源/地网；同名电源网默认全设计自动连接（层次化设计可用 `Strict Hierarchical` 本地化）。
  - **Parameter Set / PCB Layout 指令** 可在原理图侧预定义 **PCB 设计规则**（如线宽、 clearance）；`Power Net` 参数在 Update PCB 时建议生成 **Supply Nets** 规则。
  - **Validate PCB Project** 在 ECO 前检查电气/连接矩阵违规。
- **对 wiki 的映射：** [altium-designer](../../wiki/entities/altium-designer.md)、[depth-torque-motor-design Stage 4](../../roadmap/depth-torque-motor-design.md)

### 3) PCB layout：层叠、规则与布线

- **来源：** [Laying Out Your PCB](https://www.altium.com/documentation/altium-designer/pcb)、[Defining, Scoping & Managing PCB Design Rules](https://www.altium.com/documentation/altium-designer/pcb/defining-scoping-managing-design-rules)、[PCB Design Rule Types](https://www.altium.com/documentation/altium-designer/pcb/design-rule-types)
- **要点：**
  - **Layer Stack Manager**（Design » Layer Stack Manager）定义信号层、内电层、介质与盲埋孔类型；变更需 **Save to PCB**。
  - **规则驱动：** Design Rules 覆盖 clearance、线宽、via、拓扑等；在线 DRC + 批量 DRC 报告。
  - **约束双轨（AD25+）：** 新建工程可选 **Constraint Manager**（表格化、原理图/PCB 双向 ECO）或经典 **PCB Rules and Constraints Editor**（Design » Rules）；**启用 Constraint Manager 后不可回退**（可自 Rules 单向迁移至 Constraint Manager）。
  - **电机驱动板相关：** 为大电流网建立 **Net Class** + **Width** 规则；功率回路最小化、采样小信号与功率地分区——需在规则与 layout 中显式约束（官方文档为通用 EDA 机制，具体电流容量仍须按器件手册与 IPC 经验核算）。
- **对 wiki 的映射：** [altium-designer](../../wiki/entities/altium-designer.md)

### 4) 原理图 ↔ PCB 同步（ECO）

- **来源：** [Managing Design Changes between the Schematic & PCB](https://www.altium.com/documentation/altium-designer/sch-pcb)、[Keeping the Schematics & PCB Synchronized](https://www.altium.com/documentation/altium-designer/sch-pcb/keeping-synchronized)
- **要点：**
  - 无中间 netlist 文件：比较引擎生成 **Engineering Change Order (ECO)** 列表，**Execute Changes** 应用。
  - **Design » Update PCB Document**（在原理图编辑器）= 把变更推向 PCB；**Design » Update Schematics**（在 PCB 编辑器）= 反向推送。
  - 规则通过 **Unique ID** 在原理图 Parameter Set 与 PCB 规则间保持对应，同步时一并更新。
- **对 wiki 的映射：** [altium-designer](../../wiki/entities/altium-designer.md)

### 5) 制造输出与项目发布

- **来源：** [Preparing Your Design for Manufacture](https://www.altium.com/documentation/altium-designer/preparing-for-manufacture)、[Output Jobs](https://www.altium.com/documentation/altium-designer/preparing-for-manufacture/output-jobs)、[Design Project Release](https://www.altium.com/documentation/altium-designer/preparing-for-manufacture/design-release)
- **要点：**
  - **OutJob**（Output Job File）：预配置 Gerber、NC Drill、Pick&Place、BOM、装配图等；建议 **光板制造** 与 **贴装** 分 OutJob。
  - **Project Releaser**：一次性生成制造数据包，发布前可跑 DRC/验证；连接 Workspace 时与版本控制集成。
  - 教程制造章节示例输出：**Gerber + NC Drill**（见 [Tutorial — Output & Release](https://www.altium.com/documentation/altium-designer/tutorial/output-documentation-project-release)）。
- **对 wiki 的映射：** [altium-designer](../../wiki/entities/altium-designer.md)、[humanoid-hardware-101-power-compute-electronics](../../wiki/overview/humanoid-hardware-101-power-compute-electronics.md)

### 6) ECAD–MCAD 协同（机器人整机）

- **来源：** [Altium MCAD CoDesigner](https://www.altium.com/documentation/altium-designer/ecad-mcad-codesign)、[PCB CoDesign](https://www.altium.com/documentation/altium-designer/pcb/codesign)
- **要点：**
  - **CoDesigner** 插件支持 SOLIDWORKS、Fusion、Inventor 等，**双向推送** 板形/元件/约束变更（非仅 STEP 一次性导出）。
  - PCB 编辑器内置 **3D clearance**、**rigid-flex 折叠** 检查；不支持 CoDesigner 时可导出 **STEP / Parasolid**。
  - 多板装配、线束同步见 CoDesigner 专题页（Creo / SOLIDWORKS 等有限制）。
- **对 wiki 的映射：** [altium-designer](../../wiki/entities/altium-designer.md)、[cad-skills](../../wiki/entities/cad-skills.md)（机械 CAD 侧对照）

### 7) 电源完整性分析（可选扩展）

- **来源：** [Power Analyzer by Keysight QuickStart](https://www.altium.com/documentation/altium-designer/analyzing-pcb/pi-analysis/power-analyzer-keysight/quickstart-guide)
- **要点：** 可下载扩展，在 AD 内做 **PI-DC** 仿真；变更后须先 **Resync schematic ↔ PCB** 再分析。
- **对 wiki 的映射：** [altium-designer](../../wiki/entities/altium-designer.md)

## 开源 / 代码状态

- **Altium Designer 本体：** 商业闭源桌面 EDA；**无** 官方 GitHub 源码仓库。
- **协作云：** Altium 365 / Enterprise Server 为托管服务；MCAD **CoDesigner 插件** 从 [altium.com Downloads](https://www.altium.com/downloads) 获取。
- **对照：** 开源 PCB 路线见 KiCad、EasyEDA（如 [SimpleFOC 板卡](../repos/simplefoc_arduino_foc.md)）；机械侧见 [CAD Skills](../../wiki/entities/cad-skills.md)。

## 对 wiki 的映射

- [wiki/entities/altium-designer.md](../../wiki/entities/altium-designer.md) — 工具实体与机器人场景选型
- [roadmap/depth-torque-motor-design.md](../../roadmap/depth-torque-motor-design.md) — Stage 4 电机驱动 PCB
- [wiki/overview/humanoid-hardware-101-power-compute-electronics.md](../../wiki/overview/humanoid-hardware-101-power-compute-electronics.md) — PCB/BMS 模块视角
- [wiki/entities/simplefoc.md](../../wiki/entities/simplefoc.md) — 开源低功率参考板对照

## 推荐继续阅读（外部）

- [Altium Learning Hub](https://www.altium.com/resources) — 官方文章与培训
- [Altium Knowledge Base](https://www.altium.com/documentation/knowledge-base) — 支持知识库
- [KiCad Documentation](https://docs.kicad.org/) — 开源 EDA 对照

## 当前提炼状态

- [x] 官方文档入口与 QuickStart 索引
- [x] 原理图 / PCB / ECO / 制造 / ECAD-MCAD 核心摘录
- [x] wiki 实体页升格
