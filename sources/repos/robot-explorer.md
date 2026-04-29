# Robot Explorer

- **标题**: Robot Explorer
- **链接**: https://github.com/ferrolho/robot-explorer
- **类型**: repo / utility
- **作者**: ferrolho
- **核心关注点**: 动力学分析、教育可视化、Web 端交互式机械臂探索

## 核心内容摘要

### 1. 核心功能
- **交互式运动学**: 支持实时正向 (FK) 和逆向运动学 (IK)，支持多末端全身 IK 和零空间限制规避。
- **性能指标可视化**: 可视化可操纵性椭球 (Manipulability Ellipsoids，包括速度、加速度、力) 和基于 URDF 限制的力多胞体 (Force Polytopes)。
- **工作空间分析**: 通过对随机位姿进行采样生成可达性点云。
- **运动录制**: 记录和回放运动关键点，支持导出 STL 格式的凸包。

### 2. 技术栈
- TypeScript, Vite, **Three.js**, `urdf-loader`, KaTeX (数学公式渲染)。

### 3. 独特价值
- **教育深度**: 包含 IK 和椭球体背后的数学背景说明面板。
- **庞大库支持**: 内置来自 35+ 品牌的 81+ 机器人模型。
- **运动稳定性**: 具备关节限位锁定功能，确保在边界处的稳定表现。

## 对 Wiki 的映射
- **wiki/entities/robot-explorer.md** (新建)
- **references/repos/utilities.md** (更新)
