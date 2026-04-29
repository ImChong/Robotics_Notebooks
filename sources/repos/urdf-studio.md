# URDF-Studio

- **标题**: URDF-Studio
- **链接**: https://github.com/OpenLegged/URDF-Studio
- **类型**: repo / workstation / design-tool
- **作者**: OpenLegged Community
- **核心关注点**: 专业机器人设计与组装工作站、硬件 BOM 管理、AI 辅助建模

## 核心内容摘要

### 1. 核心功能
- **结构化工作流**: 提供 **Skeleton** (拓扑)、**Detail** (几何/网格) 和 **Hardware** (电机/执行器元数据) 专项模式。
- **多机器人组装**: 支持通过“桥接关节”将多个机器人组装到同一个工作空间中。
- **AI 助手**: 集成 AI 用于生成机器人结构、检查模型以及导出分析报告。
- **全面导出能力**: 支持 URDF, MJCF, USD, SDF, Xacro 导出，并提供 CSV/BOM (物料清单) 和 PDF 报告。

### 2. 技术栈
- **React 19**, TypeScript, Vite, **React Three Fiber**, Zustand, Tailwind CSS。

### 3. 独特价值
- **工作站定位**: 不仅仅是查看器，而是完整的设计创作环境，具备工作区和文件管理能力。
- **工程实用性**: 关注真实硬件工程需求，如电机库和 BOM 清单生成。
- **模块化设计**: 提供可复用的 `@urdf-studio/react-robot-canvas` 包。

## 对 Wiki 的映射
- **wiki/entities/urdf-studio.md** (新建)
- **references/repos/utilities.md** (更新)
