# raysan5/raylib

- **类型：** C / C++ 图形与游戏编程库
- **官方仓库：** <https://github.com/raysan5/raylib>
- **官方主页：** <https://www.raylib.com/>
- **许可证：** zlib/libpng（可用于商业项目；发行时保留版权与许可声明，并标明修改版本）
- **核查日期：** 2026-10-08
- **最新正式版：** raylib 6.0，2026-04-23 发布
- **沉淀到 wiki：** [Raylib 实体页](../../wiki/entities/raylib.md)

## 仓库要点

raylib 是用 C99 编写的轻量、模块化编程库，提供窗口、输入、图形、纹理、文字、模型、音频和数学功能。官方 README 表示所需底层库随主仓库提供，默认无需额外安装外部依赖。官方 README 将其定位为适合原型、工具、图形应用、嵌入式系统和教学的库；它不提供可视化编辑器或完整游戏引擎工作流。

主要模块包括 rcore（平台、窗口、输入与文件系统）、rlgl（图形 API 抽象）、rshapes、rtextures、rtext、rmodels、raudio；raymath 是独立数学头文件。多个模块可单独使用。默认硬件渲染使用 OpenGL；6.0 增加 CPU 软件渲染后端 rlsw 和面向内存帧缓冲的 rcore_memory。

## 官方学习入口

- [README](https://github.com/raysan5/raylib#learning-and-docs) — 快速上手、安装与平台构建
- [Examples](https://github.com/raysan5/raylib/tree/master/examples) — 以可运行示例作为主要学习材料
- [Cheatsheet](https://www.raylib.com/cheatsheet/cheatsheet.html) — API 函数速查
- [raylib architecture](https://github.com/raysan5/raylib/wiki/raylib-architecture) — 模块设计图
- [Releases](https://github.com/raysan5/raylib/releases) — 版本及变更

## 与 wiki 的关系

GenoView-InverseKinematics 使用 C + raylib 实现动画查看器；Raylib 是底层绘图库，GenoView 才是具体应用。
