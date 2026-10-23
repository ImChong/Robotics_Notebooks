# Scenario for Blender 官方用户指南

- **类型**：产品 / 项目文档
- **入口**：<https://blender.scenario.com/>
- **主体**：Scenario Labs
- **对应源码**：<https://github.com/scenario-labs/blender-plugin>
- **核查日期**：2026-10-23
- **文档版本提示**：页面当前列出 Blender 0.12.0；手册会跟随开发更新，可能包含尚未进入发布版的功能。

## 一句话

Scenario for Blender 是 Blender 扩展，将 Scenario 的云端生成模型与场景输入接入 Blender，支持图像、视频、3D、PBR 材质、音频、360 世界环境，以及网格编辑、场景渲染和 Blockout；本地 MCP 服务可让代理读取场景、调用 Blender 工具并请求生成。

## 工作方式

- **输入**：文本提示、参考文件、视口截图、场景相机视图、动画 playblast 或当前选中的网格。
- **生成与编辑**：按工作流调用 Scenario 模型，可生成或编辑图片、视频、3D 网格、PBR 材质、音频与 360 环境；网格操作包括 remesh、retexture、UV 展开、绑定、动画和拆分。
- **回到 Blender**：图片进入 datablocks/纹理，网格插入场景，PBR 纹理集由用户应用到材质槽，音视频可供 Sequencer 使用，360 全景可设为 World，Blockout 输出到独立 collection。
- **Agent 接口**：本地 MCP 服务可读取场景、运行 Blender 工具（包括相机移动规划）并发起 Scenario 生成。

## 安装和使用条件

官方指南要求 Blender 5.0 或更新版本、Scenario 账户及 API 访问权限，并开启 Blender 的 **Allow Online Access**。可在 Blender 扩展仓库中添加 https://blender.scenario.com/repo/index.json，或从 GitHub 下载 ZIP 安装。每次生成前会展示 Creative Units 费用；API 凭据可保存在 Blender 偏好设置或从启动环境变量读取。

## 开源状态与边界

插件源码在对应 GitHub 仓库公开，仓库标注 GPL-3.0-or-later。插件本体开源不代表云端生成服务、模型权重或模型推理栈开源；实际生成依赖 Scenario 账户、API 凭据、网络和额度。官方指南将扩展标为 experimental，并提示先查阅已知限制；因此不要仅凭生成网格推断其具备仿真所需的尺度、碰撞几何、质量属性或关节结构。

## 参考链接

- 官方用户指南：<https://blender.scenario.com/>
- 官方源码仓库：<https://github.com/scenario-labs/blender-plugin>
- 官方发布说明：<https://github.com/scenario-labs/blender-plugin/releases>
- 场景代理（MCP）指南：<https://blender.scenario.com/#agents-mcp>
