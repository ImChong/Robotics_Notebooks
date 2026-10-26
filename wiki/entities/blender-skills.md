---
type: entity
project_id: blender-skills
project: https://github.com/arjun988/blender-skills
code: https://github.com/arjun988/blender-skills
tags: [blender, coding-agents, mcp, 3d-assets, asset-pipeline, open-source]
status: complete
updated: 2026-10-26
related:
  - ./blender.md
  - ./scenario-blender-plugin.md
  - ./paper-apollo-blender.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/blender-skills.md
summary: "Blender Skills 是一套 MIT 开源的代理技能包，含 94 个 Blender 工作流技能和 175 个参考文件；它通过 Blender MCP 将自然语言任务路由到专业建模、动画、资产检查与导出流程。"
---

# Blender Skills

**Blender Skills** 是一个面向 Cursor、Claude Code、Kiro 与 Codex 的 Blender 技能仓库。它把建模、材质、动画、场景搭建、物理效果、资产优化和格式导出等操作整理成可组合的代理技能，通过 Blender MCP 与运行中的 Blender 交互。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MCP | Model Context Protocol | 技能通过 Blender MCP 插件调用 Blender 场景工具 |
| DCC | Digital Content Creation | Blender 所属的数字内容创作软件类别 |
| USD | Universal Scene Description | 技能列出的通用场景交换与导出格式之一 |
| GLTF | GL Transmission Format | 面向 3D 场景和模型交换的导出格式之一 |
| QA | Quality Assurance | 资产交付前的检查技能与清单 |

## 为什么对机器人栈有用

机器人仿真与展示项目常需制作机械外观、测试场景、碰撞代理和可视化素材。该仓库提供按任务编排的 Blender 工作流，可以减少从空场景开始搭建资产管线的摸索成本；其输出可作为 [Blender](./blender.md) DCC 流程的参考，再交给机器人仿真与资产工具验证。

## 核心结构

- **`blender-director`：** 根据任务类型选择后续技能，并串接建模、材质、优化、QA 与导出等步骤。
- **专业技能：** 包含 hard-surface、procedural-modeling、rigging、physics-sim、collision-proxy、asset-optimization、export-pipeline 等方向，也覆盖多种艺术风格与环境创作。
- **共享参考资料：** 仓库提供 175 个参考文件，包含通用资产流程、MCP 集成、命名规范和检查清单等内容。
- **执行接口：** 技能文件由 coding agent 读取；场景操作依赖单独安装并运行的 Blender MCP 插件。README 提供将技能目录复制或挂载到客户端技能目录的方式。

## 工程实践

面向机器人资产时，可将仓库用于“需求拆分 → 网格/场景制作 → 碰撞代理与优化 → USD/GLTF 等格式导出”的人工审核工作流。导出后仍应检查单位、坐标系、比例、拓扑与碰撞形状；机器人关节、惯量、URDF/MJCF 语义及仿真可用性需要通过专用工具补齐和验证，不能由通用 DCC 技能默认保证。进入训练或部署前应按 [Sim2Real](../concepts/sim2real.md) 流程核对仿真资产与目标环境。

## 局限与风险

- 该仓库主要是代理技能说明和参考文档，不是 Blender 本体、MCP 服务或机器人仿真框架；MCP 集成需要单独配置。
- 技能覆盖通用 3D 生产流程，没有提供完整的 URDF/MJCF 机器人导入与语义校验管线。
- MIT 是该技能仓库声明的许可；Blender 本体、MCP 插件及第三方素材各自有许可边界，使用时需分别核对。

## 关联页面

- [Blender（开源 3D 创作套件）](./blender.md) — 技能包所操作的 DCC 与资产处理环境。
- [Scenario for Blender](./scenario-blender-plugin.md) — 通过 Blender 插件和 MCP 将生成式资产能力接入 DCC。
- [APOLLO Blender](./paper-apollo-blender.md) — 面向机器人 URDD/URDF 与论文级可视化的专用 Blender 工具链。
- [Sim2Real](../concepts/sim2real.md) — 检查导出资产进入仿真与真机流程时的差异。

## 参考来源

- [Blender Skills 仓库归档](../../sources/repos/blender-skills.md)
- [Blender Skills README](https://github.com/arjun988/blender-skills/blob/main/README.md)
- [Blender Skills 技能索引](https://github.com/arjun988/blender-skills/blob/main/SKILLS_GUIDE.md)

## 推荐继续阅读

- [Blender Skills 安装与使用说明](https://github.com/arjun988/blender-skills)
- [Blender MCP](https://github.com/ahujasid/blender-mcp) — 技能包所依赖的 Blender 连接插件。
