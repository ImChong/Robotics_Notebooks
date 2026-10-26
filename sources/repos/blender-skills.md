# Blender Skills

- **标题：** Blender Skills
- **类型：** repo
- **来源：** arjun988
- **链接：** https://github.com/arjun988/blender-skills
- **版本：** 1.3.0（`version.json`）
- **许可：** MIT
- **入库日期：** 2026-10-26
- **一句话说明：** 面向 Cursor、Claude Code、Kiro 和 Codex 的 Blender 技能包；用 `blender-director` 将建模、动画、物理模拟、资产检查与导出任务路由到专业技能，并通过 Blender MCP 操作场景。
- **代码：** https://github.com/arjun988/blender-skills（**已开源**）
- **沉淀到 wiki：** 是 → [`wiki/entities/blender-skills.md`](../../wiki/entities/blender-skills.md)

## 仓库概况（README / `version.json`，2026-10-26）

| 字段 | 值 |
|------|-----|
| 技能数 | 94 个技能；仓库同时包含单独的共享 `references/` 目录 |
| 参考文件 | 175 个（`version.json`） |
| 版本 | 1.3.0 |
| 许可 | MIT |
| 支持客户端 | Cursor、Claude Code、Kiro、Codex |
| 执行接口 | Blender MCP；需另行安装并运行 Blender MCP 插件 |

## README 摘要

仓库以一组面向 Blender 艺术与技术生产的技能文件组织工作流。`blender-director` 可按任务选择建模、角色、场景、动画、材质、物理模拟、碰撞代理、优化、QA 与导出等技能；各技能目录还提供参考文档、流程组合和检查清单。上游给出的安装方式是把技能目录复制或挂载到对应 coding agent 的技能路径。

## 与机器人研究/工程的关联点

- **仿真资产准备：** `hard-surface`、`procedural-modeling`、`collision-proxy`、`asset-optimization` 与 `export-pipeline` 可辅助制作机械外观、场景道具、简化碰撞网格并导出 USD、GLTF、FBX 等格式，供后续仿真资产流程检查。
- **可视化与动画：** 建模、绑定、动画与渲染技能可用于机器人展示场景、动作预览和教学素材；机器人关节语义与控制数据仍需由专用工具或脚本接入。
- **边界：** 技能包提供给代理使用的文本工作流与参考资料；场景操作依赖独立的 Blender MCP 插件。它不包含机器人动力学求解器、URDF/MJCF 导入管线或策略训练环境。

## 参考链接

- 仓库 README：<https://github.com/arjun988/blender-skills/blob/main/README.md>
- 技能索引：<https://github.com/arjun988/blender-skills/blob/main/SKILLS_GUIDE.md>
- Blender MCP：<https://github.com/ahujasid/blender-mcp>
