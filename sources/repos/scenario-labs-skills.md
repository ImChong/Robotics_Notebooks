# Scenario Agent Skills（scenario-labs/skills）

> 来源归档（repo / Agent Skills 合集）

- **标题：** Scenario Agent Skills
- **类型：** repo
- **来源：** Scenario Inc.
- **原始链接：** https://github.com/scenario-labs/skills
- **许可：** MIT（LICENSE：Copyright (c) 2026 Scenario Inc.）
- **入库日期：** 2026-11-07
- **抓取方式：** 浅克隆（提交 `e7e19d5`，2026-11-07 提交）；只做静态阅读，**没有运行任何脚本**，没有连接 Scenario MCP。
- **核对范围：** README 全文（约 7 万字符）、`skills.sh.json` 头部、`AGENTS.md` 前 40 行、`SECURITY.md` 开头、Unity 与 Blender 家族 README 的状态段、`scenario` 与 `scenario-video-to-motion` 两个 SKILL.md 的开头；目录结构与技能数量由 `find` 统计。其余约 170 个 SKILL.md、约 1,900 个文件（含 Python 与 C# 脚本、references）**未读**。
- **通读：** 未完成
- **一句话说明：** Scenario 为 AI 编程智能体（Claude Code、Cursor、Codex、Copilot 等）发布的 Agent Skills：教智能体通过 Scenario MCP 服务器生成图像、视频、音频、纹理、天空盒、3D 与定制模型；另有一组「专家工具」技能，驱动本机的 DCC 软件和游戏引擎。

## 结构（实测）

- 仓库约 1,928 个文件；`find` 统计到 175 个 SKILL.md。
- `skills/scenario*`：81 个顶层目录（核心技能，通过 Scenario MCP 工作，如图像、视频、3D、音频、工作流、团队管理）。
- 专家工具：`skills/dcc/`（Blender 15、Maya 11、ZBrush 9 个 SKILL.md）与 `skills/game-engines/`（Unity 14、Godot 14、Roblox 14、Unreal 10）。
- 技能格式遵循 Agent Skills 规范（`SKILL.md` + 可选 references / scripts）；安装命令 `npx skills add scenario-labs/skills`，MCP 端点 `https://mcp.scenario.com/mcp`。
- 仓库自述：每个随技能发布的脚本有测试套件；`pnpm run validate` 与 CI 做内容检查；`SECURITY.md` 说明仓库只含技能文档与少量辅助脚本，不运行服务。

## 作者自述的验证情况（未独立复核）

- Unity 家族 README：14 个技能，据 203 个专家视频与 76 页文档写成；在 Unity 6000.3.21f1 / macOS 上跑过 447 项实测；12 个书面场景盲评，无技能 63%（113/180），有技能 97%（174/180），作者注明每场景只有一位评分者，只能当方向、不是统计，且这是书面方案而非成品游戏。
- Blender 家族 README：15 个技能，书面场景盲评有技能 7 战全胜；12 个测试套件在 Blender 5.2.1 无头通过。
- 作者列出的未执行项包括手机安装、商店上传、Apple 签名等。

## 与机器人的关系（本库判断）

- 与具身智能的交集在资产与动作：`scenario-video-to-motion`（视频转 3D 角色动作）、3D 资产生成与 Blender / Unity 自动化，可能用于仿真场景资产或动作数据来源；README 里没有机器人相关用例，该关联是本库的推测。

## 对 wiki 的映射

- [Scenario Agent Skills](../../wiki/entities/scenario-labs-skills.md)
