# tscircuit（官方源码仓库）

> 来源归档

- **标题：** tscircuit — React for Circuits
- **类型：** repo
- **来源：** tscircuit 组织
- **主仓库：** https://github.com/tscircuit/tscircuit
- **官网：** https://tscircuit.com/
- **文档：** https://docs.tscircuit.com/
- **在线 Playground：** https://tscircuit.com/playground
- **入库日期：** 2026-10-23
- **一句话说明：** MIT 开源、TypeScript/React 风格的电子设计工具链；把电路组件与连接描述为代码，预览原理图、PCB 与 3D，并导出 Gerber、BOM、Pick-and-Place 等制造文件。
- **沉淀到 wiki：** [tscircuit](../../wiki/entities/tscircuit.md)、[KiCad](../../wiki/entities/kicad.md)

---

## 开源状态（2026-10-23 核查）

tscircuit 官方 GitHub 仓库公开发布主 npm 包，仓库 README 与 package metadata 标注 MIT 许可。README 提供本地 CLI 入口（安装后运行 `tsci dev`），并链接官网、在线 Playground 与官方文档。

官网/文档展示电路设计、原理图/PCB/3D 预览与导出能力；官方文档同时将自动 PCB 布局标为 beta，布局算法仍在持续迭代。

---

## 项目结构与使用入口

| 入口 | 用途 |
|------|------|
| `tscircuit` npm 包 | 核心一体化包；支持 TypeScript/React 风格的电路描述 |
| `tsci` CLI | `tsci dev` 本地预览；CLI 还提供构建、检查、导出、仿真等入口 |
| 在线 Playground | 浏览器中编辑、运行并查看电路 |
| 组织下的配套仓库 | core、CLI、schematic/PCB viewer、routing 等独立子项目 |

典型流程：编写可复用的电路组件与网络连接 → 生成 Circuit JSON → 检查/布局 → 在原理图、PCB、3D 视图中审阅 → 导出制造文件或 KiCad、STEP、GLB、SPICE 格式。

---

## 与机器人硬件的关联

- **关节驱动与电源板：** 项目 README 展示 RP2040 电机控制板、步进电机控制器和 BLDC 电机控制器，可用于理解其机器人硬件适用场景。
- **传感与接口板：** 以代码复用电源、MCU、连接器和传感电路模块，便于参数化板卡变体与 Git 评审。
- **工具链分工：** tscircuit 适合代码化建模、协作与快速生成；与 [KiCad](../../wiki/entities/kicad.md) 可通过格式导出互通。制造前仍须检查电气规则、PCB 细节并完成实物验证。

