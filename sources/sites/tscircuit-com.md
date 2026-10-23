# tscircuit（官方项目页）

> 来源归档

- **标题：** tscircuit — Code Electronics with React
- **类型：** site（开源 EDA 工具链官网与文档）
- **来源：** tscircuit
- **链接：** https://tscircuit.com/
- **文档：** https://docs.tscircuit.com/
- **在线 Playground：** https://tscircuit.com/playground
- **入库日期：** 2026-10-23
- **代码：** https://github.com/tscircuit/tscircuit（已开源，MIT；官网同时链接 CLI 及多个配套模块仓库）
- **一句话说明：** 用 TypeScript/React 编写电路，生成原理图、布线 PCB、3D 预览及制造文件；可在浏览器、编辑器或 AI 辅助开发流程中使用。
- **沉淀到 wiki：** [tscircuit](../../wiki/entities/tscircuit.md)、[KiCad](../../wiki/entities/kicad.md)

---

## 官网与文档核查

官方文档将 tscircuit 定义为开源电子设计工具链，支持在浏览器或本地 CLI 项目中编写电路，预览原理图、PCB 与 3D，并导出 Gerber、BOM、Pick-and-Place；文档亦列出 KiCad、STEP、GLB 与 SPICE 等导出格式。

官方文档注明自动布局仍处于 beta 阶段。具体板卡仍需人工审阅布局、设计规则与制造文件，并通过实际电气验证。

## 与机器人工程的关联

官方展示案例包括 RP2040 电机控制器与步进/无刷电机控制板，说明它可以用于机器人控制板、接口板等原型设计；这些案例不代表工具本身具备机器人专用验证能力。

