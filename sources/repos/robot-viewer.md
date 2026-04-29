# Robot Viewer

- **标题**: Robot Viewer
- **链接**: https://github.com/fan-ziqi/robot_viewer
- **类型**: repo / utility
- **作者**: fan-ziqi
- **核心关注点**: 多格式模型检查、实时代码编辑、浏览器端 MuJoCo 仿真

## 核心内容摘要

### 1. 核心功能
- **广泛格式支持**: 原生支持 URDF, **Xacro** (支持宏展开), **MJCF** (MuJoCo), 以及 USD 格式。
- **集成仿真**: 利用 MuJoCo WASM 直接在浏览器中运行 MJCF 模型的动力学仿真。
- **实时编辑**: 内置 CodeMirror 编辑器，支持实时修改 XML/Xacro 代码并预览变更。
- **工程工具**: 提供关节/链接间距、地面高度的精确测量工具。

### 2. 技术栈
- JavaScript, Vite, **Three.js**, `xacro-parser`, `mujoco_wasm`。

### 3. 独特价值
- **Xacro 原生支持**: 少数无需 ROS 后端即可在 Web 端处理 Xacro 逻辑 (宏、条件判断) 的查看器。
- **仿真无缝衔接**: 支持从静态可视化到动态 MuJoCo 仿真的平滑过渡。

## 对 Wiki 的映射
- **wiki/entities/robot-viewer.md** (新建)
- **references/repos/utilities.md** (更新)
