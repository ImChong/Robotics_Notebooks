# Raylib 官方网站与架构说明

- **官网：** <https://www.raylib.com/>
- **架构页：** <https://github.com/raysan5/raylib/wiki/raylib-architecture>
- **官方仓库：** <https://github.com/raysan5/raylib>
- **核查日期：** 2026-10-08
- **对应 wiki：** [Raylib 实体页](../../wiki/entities/raylib.md)

## 官网定位

raylib 自称为简洁、易用的游戏编程库；强调通过 C 代码直接调用 API，不附带可视化编辑器或复杂 GUI。官网列出 C99、多平台、OpenGL 硬件加速、2D/3D 绘制、模型/骨骼动画、着色器、数学和音频能力，并把示例集合作为主要学习入口。

架构页和 README 强调功能被拆为小型、边界明确的模块，其中部分模块可以独立使用。对于机器人开发，这让它适合作为自定义状态查看器或动画/轨迹调试窗口的渲染层；机器人模型解析、ROS/控制数据接入和物理仿真仍需其他组件负责。

## 6.0 更新

官方发布记录显示 raylib 6.0 于 2026-04-23 发布。主要变化包括 CPU 软件渲染后端 rlsw、可输出到内存帧缓冲的 rcore_memory、骨骼动画系统重设计、文件系统 API 整理和构建配置改进。内存后端支持 headless 绘制并将帧导出为图像；软件渲染不需要 GPU，但速度低于硬件加速路径。

## 资源

- [raylib examples](https://www.raylib.com/examples.html)
- [raylib cheatsheet](https://www.raylib.com/cheatsheet/cheatsheet.html)
- [raylib 6.0 release notes](https://github.com/raysan5/raylib/releases)
- [Platforms / build wiki](https://github.com/raysan5/raylib/wiki#development-platforms)
