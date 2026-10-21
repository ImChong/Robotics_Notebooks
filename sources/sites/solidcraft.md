# SolidCraft 官方项目页与使用文档

- **类型：** site / official-documentation
- **官网与手册：** <https://bherbruck.github.io/solidcraft/>
- **浏览器应用：** <https://bherbruck.github.io/solidcraft/app/>
- **源码仓库：** [bherbruck/solidcraft](https://github.com/bherbruck/solidcraft)
- **入库日期：** 2026-10-16
- **沉淀到 wiki：** [SolidCraft 工具实体页](../../wiki/entities/solidcraft.md)

## 官方资料要点

项目页将 SolidCraft 描述为纯 Rust 编写的开源参数化 3D CAD。建模主线是平面草图、几何约束与尺寸、实体特征时间线；修改参数后，特征时间线会重新构建。软件提供 Windows/Linux 桌面版、基于 WebAssembly 的浏览器版、无头命令行工具 `solidcraft-cli`，以及供 AI agent 建模的 MCP 服务。浏览器版页面标注使用 WebGPU 或 WebGL2。

官方项目页直接链接源码仓库，README 声明 MIT OR Apache-2.0 双许可。项目仍处于早期阶段：文档列出草图、实体建模、参数、组件关节、钣金与塑料外壳子集等已有功能；曲面、工程图、CAM 与仿真尚在规划。

## 机器人工程边界

浏览器入口降低了开始参数化建模的门槛，可用于试做机器人支架、外壳和实验夹具等几何件。建模结果本身不等于机器人描述：质量/惯量、坐标系、碰撞体、连杆与关节仍需通过下游 CAD/仿真流程核验和补充。

## 来源归档

- [官方源码仓库归档](../repos/solidcraft.md)
- [SolidCraft 工具实体页](../../wiki/entities/solidcraft.md)
