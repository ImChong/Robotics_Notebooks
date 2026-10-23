# Scenario for Blender（官方扩展仓库）

- **标题**：Scenario for Blender
- **类型**：repo
- **来源**：Scenario Labs
- **链接**：<https://github.com/scenario-labs/blender-plugin>
- **项目页**：<https://blender.scenario.com/>
- **许可**：GPL-3.0-or-later
- **支持版本**：Blender 5.0+
- **入库日期**：2026-10-23
- **一句话说明：** Scenario Labs 为 Blender 提供的实验性 Python 扩展，把 Scenario 云端图像、视频、3D、PBR 材质、音频与 360 环境生成接入视口，并提供本地 MCP 服务供代理读取场景和调用工具。
- **沉淀到 wiki：** 是 → [Scenario for Blender](../../wiki/entities/scenario-blender-plugin.md)

## 官方仓库信息

仓库 README 将插件标为 **Experimental**，要求 Blender 5.0 或更高版本、Scenario API 账户及联网权限。项目采用 GPL-3.0-or-later；生成服务需使用 Scenario 账户 API 凭据，生成前会显示 Creative Units（CU）费用估算。

## README 功能摘要

- 从文本提示、参考文件、视口截图、场景相机视图、动画 playblast 或选中网格提交生成任务。
- 支持图像、视频、3D 网格、PBR 材质、音频和 360 环境；另含 Render Image、Render Video 与 Blockout 工作流。
- 对选中网格执行 remesh、retexture、UV unwrap、rig、animate、split 等编辑任务。
- 将生成结果回填到 Blender：网格放在 3D 游标处或原网格旁，材质作为纹理集应用，视频和音频进入 Sequencer。
- 提供本地 MCP 服务，让 Claude Code、Cursor 等代理读取场景、运行 Blender 工具，并通过 Scenario 生成内容。

## 参考链接

- 官方源码：<https://github.com/scenario-labs/blender-plugin>
- 官方用户指南：<https://blender.scenario.com/>
- 官方安装仓库索引：<https://blender.scenario.com/repo/index.json>
- 许可证：<https://github.com/scenario-labs/blender-plugin/blob/main/LICENSE>
