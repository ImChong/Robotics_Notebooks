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
- **核对范围：** 2026-11-07 浅克隆 `main`（提交 `42af568`，2026-11-04，1,044 个文件；其中 `scenario/` 252 个、`tests/` 425 个、`docs/` 223 个）。逐字读完：`README.md`；`docs/MCP.md` 的概览、安全模型、请求边界、Agent 文件上传、默认启动与上传决定、本地服务与 mcp.scenario.com 各节，并按标题通览全篇；`docs/PRIVACY.md` 的「本地 MCP 服务」节与标题；`docs/KNOWN_LIMITATIONS.md` 开头；`scenario/blender_manifest.toml` 的权限与依赖轮子清单；同日读完的 `blender.scenario.com` 手册（见对应站点归档，它由 `docs/USER_GUIDE.md` 渲染而来，故用户指南内容已读）。**未读：** `scenario/` 下约 232 个 Python 源文件、`tests/` 下约 358 个测试、`docs/MCP.md` 其余约 130 KB 的逐工具契约、`SDK_ADOPTION.md`（102 KB）等开发文档、CI 工作流与 `.agents`、`.claude`。「通读」在此指仓库文档与源码都读完；因源码与大部分开发文档未读，标未完成。
- **通读：** 未完成
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

## 2026-11-07 重读补充（README 与 docs/MCP.md，提交 42af568）

- **版本与支持：** 清单版本 0.13.0；Blender 5.0 及以上，原生 CI 覆盖 Blender 5.0.1、5.1.2、5.2.3（Linux 与 Windows，pull request 上跑），macOS 与 Windows 另有每周的信息性工作流；扩展带固定版本的 SDK 依赖轮子（httpx、pydantic 等）。
- **为何不上 extensions.blender.org：** README 说该站条款限制依赖外部注册、密钥或付费的扩展，所以走自己的仓库 `https://blender.scenario.com/repo/index.json`，每个发布的 ZIP 也附在 GitHub release。
- **许可：** 第一方代码 GPL-3.0-or-later（因为导入 `bpy` 的 Blender 加载项须与 GPL 兼容）；Scenario for Unity 为 MIT；同组织的 skills 仓库为 MIT；打包的依赖与采用的源码保留原许可；GPL 不授予商标权。
- **本地 MCP 的安全模型（`docs/MCP.md`、`docs/PRIVACY.md`）：** 只绑定 `127.0.0.1`，**不要求凭据，本机任何账号下的任何程序都能调用全部工具**；校验 Host 头（只认 `127.0.0.1`、`localhost`、`[::1]`）与 Origin（须为回环 HTTP(S) 来源）以防 DNS 重绑定；请求体上限 10 MiB，连接空闲 60 秒断开；`execute_python` 默认关闭且「黑名单只是护栏、不是安全边界」；生成必须带本会话的报价 ID 与用户批准的精确费用，但文档同时写明「工具描述只是给智能体的提示，不是服务端强制的花费批准」；上传仅限扩展自己保存的文件或用户本会话在 Blender 中选过的文件；智能体写文件仅限 Output Folder，且不会覆盖已有文件。
- **与 skills 仓库的差异（本库发现）：** skills 仓库的 `scenario-blender` 技能写本地服务用 bearer token 且每次重启变化，README 与 `docs/MCP.md` 则写 Codex 若仍报 `SCENARIO_BLENDER_TOKEN` 未设置是旧配置、应重新连接；两处在同一周的快照里不一致，哪个是当前行为未核实。
- **已知限制文档的口径：** 明说合成探针与无头测试不等于真机 / 付费验收；Film 仍实验性，视频导出与案例包的桌面验证待办（#494）；曾有一次 macOS 桌面探针崩溃（#263）已关闭但没有公开根因。
