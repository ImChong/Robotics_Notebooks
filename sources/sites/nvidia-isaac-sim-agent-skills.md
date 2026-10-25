# Agent Skills — Isaac Sim 官方文档

> 来源归档

- **标题：** Agent Skills — Isaac Sim Documentation
- **类型：** 官方开发者文档 / AI Agent 工作流指南
- **URL：** <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/development_tools/agent_skills.html>
- **文档更新时间：** 2026-10-14
- **官方源码：** <https://github.com/isaac-sim/IsaacSim/tree/develop/skills>
- **技能索引：** <https://github.com/isaac-sim/IsaacSim/blob/develop/skills/SKILLS.md>
- **官方仓库：** <https://github.com/isaac-sim/IsaacSim>
- **一句话说明：** NVIDIA 随 Isaac Sim 发布的 `SKILL.md` 工作流指南，让 Cursor、Claude Code、Codex CLI 等 AI 编码 Agent 按需执行仿真开发任务。
- **范围：** 仓库构建、调试和性能分析；下游仿真用户工作流，包括资产导入、物理、导航、操作、传感器、合成数据、渲染与 ROS 2。
- **发布方式：** 官方文档说明技能包含在源码树中，并随 pip 包与二进制包分发。
- **安全提示：** 远程控制类技能可能在 Kit 进程执行任意 Python；文档建议将 Python Server 绑定至 `127.0.0.1`。
- **沉淀到 wiki：** [`wiki/entities/isaac-sim.md`](../../wiki/entities/isaac-sim.md)

---

## 核心资料

1. **官方文档**：说明技能目录结构、Agent 发现机制、可用技能与安装方式。
2. **官方技能索引**：列出技能分类和组合关系，链接至各技能的 `SKILL.md`。
3. **官方源码目录**：可核对技能文件、脚本和参考资料的实际内容。

## 官方资料链接

- [Agent Skills 官方文档](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/development_tools/agent_skills.html)
- [Isaac Sim 技能目录](https://github.com/isaac-sim/IsaacSim/tree/develop/skills)
- [技能分类索引](https://github.com/isaac-sim/IsaacSim/blob/develop/skills/SKILLS.md)
- [Meta-Skills 示例](https://github.com/isaac-sim/IsaacSim/blob/develop/skills/meta-skills/SKILL.md)
- [构建配置：随安装包打包公开技能](https://github.com/isaac-sim/IsaacSim/blob/develop/premake5.lua)

## 对本仓库的映射

- Isaac Sim 实体页：[wiki/entities/isaac-sim.md](../../wiki/entities/isaac-sim.md)
- Isaac Sim 仓库归档：[sources/repos/isaac_sim.md](../repos/isaac_sim.md)
