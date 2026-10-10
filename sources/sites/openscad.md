# OpenSCAD 官方网站与文档

- **类型：** site / official-documentation
- **官网：** <https://openscad.org/>
- **文档入口：** <https://openscad.org/documentation.html>
- **下载页：** <https://openscad.org/downloads.html>
- **源码仓库：** [openscad/openscad](https://github.com/openscad/openscad)
- **文档语言参考：** <https://files.openscad.org/documentation/manual/The_OpenSCAD_Language.html>
- **命令行说明：** <https://files.openscad.org/documentation/manual/Using_OpenSCAD_in_a_command_line_environment.html>
- **入库日期：** 2026-10-10
- **沉淀到 wiki：** [OpenSCAD 工具实体页](../../wiki/entities/openscad.md)

## 官方资料要点

官网将 OpenSCAD定位为“程序员式”的实体 3D CAD 建模工具。语言参考说明模型由 .scad 脚本描述，基本手段包括 2D/3D primitive、变量与模块、变换、CSG 布尔运算，以及二维轮廓挤出。用户手册区分快速预览与完整渲染：预览可出现近似显示伪影，完整渲染生成网格，复杂模型可能耗时更长。

命令行手册说明通过 -o 可无 GUI 运行脚本并按扩展名导出；当前版本支持的确切格式与参数应以安装版本的 openscad --help 为准。官网文档和下载页还链接源码仓库与跨平台发行包。

## 机器人工程边界

OpenSCAD 适合用源码记录尺寸和重复几何，便于版本控制、参数扫描及无头批量生成；模型导出仍是几何资产，不会自动提供机器人描述所需的关节树、坐标惯量、材质和接触参数。机器人仿真前应另行生成/维护 URDF 或 MJCF，并检查几何单位、坐标和质量属性。

## 来源归档

- [官方源码仓库归档](../repos/openscad.md)
- [OpenSCAD 工具实体页](../../wiki/entities/openscad.md)
