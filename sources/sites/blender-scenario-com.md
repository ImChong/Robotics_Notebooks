# Scenario for Blender 官方用户指南

- **类型**：产品 / 项目文档
- **入口**：<https://blender.scenario.com/>
- **主体**：Scenario Labs
- **对应源码**：<https://github.com/scenario-labs/blender-plugin>
- **核查日期**：2026-10-23 首次归档；2026-11-07 用 curl 重新抓取并通读（页面标 0.13.0）
- **文档版本提示**：2026-10-23 归档时页面列 0.12.0，2026-11-07 重读时为 0.13.0（`repo/index.json` 同为 0.13.0）；手册会跟随开发更新，可能包含尚未进入发布版的功能。
- **核对范围：** 2026-11-07 抓取 `https://blender.scenario.com/` 单页 HTML（150,000 字节，以 `</html>` 结尾）并转成文本（约 1,290 行），从「What it does」到「Licence」每节逐段读完；另读 `https://blender.scenario.com/repo/index.json`（1,264 字节）。「通读」在此指这一个手册页面的正文。**未读：** 页面链接出去的 25 页插图 PDF、更新日志、Privacy 页、Film 参考、MCP 参考页、downloads 页和 17 个 Explorations 子页（其中 MCP 与 Privacy 的仓库 Markdown 在对应 repo 归档里读了一部分）。
- **通读：** 已完成

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

## 2026-11-07 通读补充（0.13.0 手册）

- **版本与分发：** 扩展 id `scenario`，版本 0.13.0，包 19,331,649 字节，平台 linux-x64、windows-x64、macos-arm64、macos-x64，Python 3.11 与 3.13，权限声明网络（api.cloud、clerk、blender.scenario.com 可选、127.0.0.1）、文件、剪贴板；手册说目标 Blender 5.0、5.1、5.2，自动发布附 SHA256SUMS 与构建来源证明（`index.json`、手册）。
- **车道（lanes）：** Image、Video、3D、Materials（PATINA 优先）、Audio、Render Image、Render Video、Blockout、Environment；另有 Studio 视图（工作流、资产库、结果）与实验性的 Film（JSON 配方到镜头、时间线、成片、案例包）。
- **3D 编辑与回填：** Edit 模式的任务为 Remesh、Retexture、UV Unwrap、Rigging、Animate、Parts；导入格式 GLB、glTF、FBX、OBJ、mesh PLY，单文件最大 256 MiB；Apply animation 只在已有人形骨架上重定向（骨骼名为 Meshy、Mixamo、SMPL 或 Unreal 风格），「其他骨架会被拒绝」；Gaussian splat 世界只能以彩色点云导入。
- **本地 MCP：** 手册方块计 88 个工具（Scene 11、Generate & place 18、Library & uploads 9、Workflows 7、Jobs & recovery 5、Blockout 5、Film 33）；服务「不需要 token，本机任何程序都能连」；`execute_python` 默认关。
- **隐私与凭据：** 价格检查在编辑时把提示词和参数发往 `api.cloud.scenario.com`；API key 以明文存于 Blender 的 `userpref.blend`（非系统钥匙串）；浏览器登录会话存在扩展私有目录，经 `clerk.scenario.com` 刷新；匿名使用数据默认关。
- **费用控制：** Generate 按钮显示精确 CU，高于偏好里的阈值（默认 500 CU）会再确认；提交结果不确定时拒绝重复提交，必须先检查已保存作业。
- **已知限制：** 手册标注实验性，Trained models、语音转文字、视频转动作为 Experimental，HY World 尚无真机验证（#494）。
