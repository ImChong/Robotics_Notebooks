# Utilities (通用机器人工具链)

收录动力学计算、模型可视化、设计组装等通用工具链。

## 动力学与控制计算
- **Pinocchio**: 高性能刚体动力学库，支持分析导数。
- **RBDL**: 刚体动力学库。
- **Drake**: 专注于控制、仿真和分析。
- **[cuRobo](../../wiki/entities/curobo.md)**：NVIDIA 开源的 GPU 并行无碰撞运动生成库（初版与 cuRoboV2 论文见实体页）。

## Web 端可视化与分析
- **[robot-explorer](../../wiki/entities/robot-explorer.md)**: 交互式 3D 机器人探索工具，支持运动学分析与可操纵性椭球可视化。
- **[robot-viewer](../../wiki/entities/robot-viewer.md)**: 多格式（URDF/Xacro/MJCF/USD）查看器，集成 MuJoCo WASM 仿真与实时编辑。
- **[Open Duck Mini Viewer](../../wiki/entities/open-duck-mini-viewer.md)**: 浏览器内 Open Duck Mini V2 GUI（脚本步态 + URDF，无 MuJoCo）；在线 [Pages demo](https://mertcookimg.github.io/Open_Duck_Mini_Viewer/)。

## 机器人设计工作站
- **[urdf-studio](../../wiki/entities/urdf-studio.md)**: 专业级 Web 机器人设计与组装平台，支持 Skeleton/Detail/Hardware 全流程管理与 AI 辅助。
- **[step2urdf](../../wiki/entities/step2urdf.md)**: 浏览器端 STEP→URDF 转换；OpenCascade.js 本地解析、几何驱动关节识别与惯量估算，在线版 [step2urdf.top](https://step2urdf.top/)。
- **[Tnkr](../../wiki/entities/tnkr.md)**: 开源整机项目协作平台（CAD/线束/代码/部署一体），对接 GitHub、Onshape 与 AI 助手 Leonardo。

## Agent 驱动的图示与 CAD 桥
- **[Draw.io Scientific Illustrator](../../wiki/entities/drawio-scientific-illustrator.md)**: Codex 插件；本机 MCP 可见操控 draw.io 画布，逐步重绘可编辑科研插图。
- **[FreeCAD MCP](../../wiki/entities/freecad-mcp.md)**: FreeCAD Addon RPC + MCP server，自然语言驱动桌面 CAD。
- **[CAD Skills](../../wiki/entities/cad-skills.md)**: build123d STEP-first Agent Skills（CAD/URDF/制造交接）。
- **[Manim](../../wiki/entities/manim.md)**: Python 程序化数学/技术讲解动画（对外沟通层）。

