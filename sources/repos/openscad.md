# OpenSCAD（openscad/openscad）

- **类型：** repo / CAD 软件源码
- **链接：** <https://github.com/openscad/openscad>
- **官网：** <https://openscad.org/>
- **文档：** <https://openscad.org/documentation.html>
- **许可证：** 官方 About 页面标注 GNU GPL version 2；以仓库许可证及分发版本附带文本为准
- **入库日期：** 2026-10-10
- **一句话说明：** 以脚本描述参数化 2D/3D 实体，通过 CSG 布尔操作、几何变换和二维轮廓拉伸构建模型；支持 GUI 预览、完整渲染与命令行导出。
- **沉淀到 wiki：** [OpenSCAD 工具实体页](../../wiki/entities/openscad.md)

## 仓库核查

官方 GitHub 仓库为非归档公开仓库，主要语言为 C++。README 将 OpenSCAD 定义为面向机械零件等 CAD 对象的脚本式建模器：模型由源脚本驱动，可通过参数改变尺寸或重复生成部件；建模主线是 constructive solid geometry（CSG）与二维轮廓挤出。官方源码 README 记录使用 Git 子模块拉取依赖，并提供跨平台构建说明。

## 机器人建模相关性

可将重复孔阵列、传感器支架、外壳、简易连接件等描述为可版本控制的参数脚本，再导出 STL/3MF 等网格供打印或碰撞几何使用。官方文档描述的主要输出是网格及 2D 格式；不要把它误认为 STEP/B-rep 原生设计系统，也不要将导出网格直接视为已具备 URDF 关节、惯量和碰撞组配置。

## 参考

- [官方 README](https://github.com/openscad/openscad/blob/master/README.md)
- [官方文档入口](https://openscad.org/documentation.html)
- [命令行导出说明](https://files.openscad.org/documentation/manual/Using_OpenSCAD_in_a_command_line_environment.html)
- [官方 About 页面（许可证与技术简介）](https://openscad.org/about.html)
