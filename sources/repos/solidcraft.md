# SolidCraft（bherbruck/solidcraft）

- **类型：** repo / 参数化 3D CAD 源码
- **链接：** <https://github.com/bherbruck/solidcraft>
- **官网与文档：** <https://bherbruck.github.io/solidcraft/>
- **在线应用：** <https://bherbruck.github.io/solidcraft/app/>
- **许可证：** MIT OR Apache-2.0（以仓库当前许可文件为准）
- **入库日期：** 2026-10-16
- **一句话说明：** 用 Rust 构建的开源参数化 CAD，支持草图约束、尺寸标注、特征时间线与参数修改后的模型重建，并提供浏览器、桌面、CLI 和 MCP 使用入口。
- **沉淀到 wiki：** [SolidCraft 工具实体页](../../wiki/entities/solidcraft.md)

## 仓库核查

官方仓库公开可访问，默认分支为 `main`，主要实现语言为 Rust。官方项目文档将其描述为早期但可用于简单零件建模的软件；官网直接提供浏览器应用，并链接到本仓库源码。许可为 MIT 或 Apache-2.0 双许可。

## 功能与机器人工程关联

SolidCraft 的建模流程包括在平面上绘制草图、添加约束与尺寸、将轮廓转为实体特征并保存在时间线中；修改参数后可重建时间线。桌面应用使用 egui + wgpu，浏览器端基于 WebAssembly，可使用 WebGPU 或 WebGL2；另提供无头 CLI 与 MCP 服务。

对机器人开发者而言，可将其作为支架、外壳、夹具等零件的参数化建模候选工具。CAD 几何仍需经过质量属性、坐标系、碰撞模型和关节语义处理，才能成为完整的 URDF/MJCF 仿真资产。

## 当前边界

根据官方文档，草图、实体建模、参数、带关节组件、钣金和塑料外壳子集已有支持；曲面、工程图、CAM 与仿真列为后续规划。用于实际机械设计前，应按项目需求检查功能清单与路线图。

## 参考

- [SolidCraft 官方仓库 README](https://github.com/bherbruck/solidcraft/blob/main/README.md)
- [SolidCraft 官方项目页与用户/开发者文档](https://bherbruck.github.io/solidcraft/)
- [SolidCraft 浏览器应用](https://bherbruck.github.io/solidcraft/app/)
- [官方路线图与功能清单](https://github.com/bherbruck/solidcraft)
