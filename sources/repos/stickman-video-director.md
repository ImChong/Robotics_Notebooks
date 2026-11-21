# Stickman Video Director｜火柴人视频导演（kaomei/stickman-video-director）

> 官方开源项目一手资料归档；本站只归纳核对结果，不复制上游完整文档或样片。

- **标题：** Stickman Video Director
- **类型：** repo / Agent Skill / 视频制作
- **作者/来源：** kaomei / Stickman Video Director contributors
- **链接：** https://github.com/kaomei/stickman-video-director
- **代码：** https://github.com/kaomei/stickman-video-director/tree/main/skills/directing-stickman-videos
- **主要原文：** [README](https://github.com/kaomei/stickman-video-director/blob/main/README.md)、[SKILL.md](https://github.com/kaomei/stickman-video-director/blob/main/skills/directing-stickman-videos/SKILL.md)、[模式边界](https://github.com/kaomei/stickman-video-director/blob/main/skills/directing-stickman-videos/references/mode-execution-boundaries.md)、[B 模式](https://github.com/kaomei/stickman-video-director/blob/main/skills/directing-stickman-videos/references/autopilot.md)、[B 运行时](https://github.com/kaomei/stickman-video-director/blob/main/skills/directing-stickman-videos/references/autopilot-runtime.md)
- **入库/复核日期：** 2026-11-21
- **许可：** MIT（根目录 [LICENSE](https://github.com/kaomei/stickman-video-director/blob/main/LICENSE)）；第三方图片、音乐、音效、模型/API 各按单独许可和费用处理。
- **开源结论：** **已开源**：Skill 规约、A/B 风格参考、B 模式的 Node.js/Python 脚本、测试与部分样例媒体可在官方仓库获取。**不等于**云视频生成平台、外部模型配额、全部素材版权或本地效果验收免费且完备。
- **一句话说明：** 把选题或脚本交给 AI Agent，通过「模型生成镜头」或「素材分层 + 代码动画」两种制作路径，编排火柴人视频、配音、字幕、声音与成片复核。
- **沉淀到 wiki：** [Stickman Video Director 独立实体节点](../../wiki/entities/stickman-video-director.md)
- **核对范围：** 通读上游 README 的两种引擎、A1–A7 / B1–B7 风格说明、使用示例、FAQ、许可证和规划；核对 SKILL.md 主路由、mode-execution-boundaries、autopilot、autopilot-runtime、b-style-installed-workflow 关键制作/验收章节；逐段阅读 `scripts/autopilot.mjs` 的命令入口及 `scripts/autopilot-core.mjs` 的项目/依赖加载实现，并检查 `package.json`、脚本/测试目录与 LICENSE。**不声称**读遍 73 份风格参考、运行所有渲染测试或实际观看全部演示视频。
- **通读：** 已完成

## 核心机制（2026-11-21 仓库快照）

| 项目 | 官方机制 | 实现与边界 |
|---|---|---|
| A · Model Engine（模型引擎） | 选故事、风格、比例、时长、声音 → 确认分镜 → 提交可独立使用的逐镜头提示词 → 获授权后可做平台预览/完整制作 | A1–A7；依赖 Google Flow 或可用的视频生成平台与额度；不代表所有环境都能自动操控外部网站 |
| B · Code Engine（代码引擎） | 生成/复用图层素材，按声音节拍写动画/镜头、字幕与音效，使用本地代码渲染、检查结果 | B1–B7 风格，具体执行入口因风格而异；需 Agent 文件与命令执行能力、图像/TTS 工具和本地编解码环境 |
| B1–B4 执行入口 | `node scripts/autopilot.mjs doctor`、`init`、`validate`、`preview`、`approve`、`render` | `autopilot.mjs` 路由到 `autopilot-core.mjs` 与 `autopilot-render.mjs`；`render` 校验之前的预览与真实用户批准 |
| B5/B6/B7 特化 | B5 独立 `character-explainer.mjs` 代码路径；B6 专用 painted-world 运行时；B7 参考 Pixel Theater 专属路线 | **不能**把全部 B 风格当作同一个 manifest CLI；部分样例/验收针对特定宽高比 |
| 运行依赖 | `package.json` 要求 Node.js ≥20，并声明 `@napi-rs/canvas`、`sharp` | B 运行时另需 FFmpeg、FFprobe、字体及所选素材/配音工具；doctor 是环境诊断，不是平台额度和最终画质证明 |
| 内容发布 | 14 个当前编号风格（A1–A7 + B1–B7）；A8 / B8 仍是 Coming soon | README 中后续 Channel Autopilot、桌面 Studio 是路线图，不应当成已发布能力 |

## 真实代码入口与数据流

读取 [`scripts/autopilot.mjs`](https://github.com/kaomei/stickman-video-director/blob/main/skills/directing-stickman-videos/scripts/autopilot.mjs) 得到 B1–B4 代码路径：

1. `init DIR --style B1 --ratio 16:9`：创建独立目录的 `project.json` 空骨架；必须补入可使用的声音、时间戳字幕、场景及素材。
2. `validate PROJECT`：`loadProject` 载入配置和媒体路径，`painter` 做画面/字幕等技术校验；不表示审美验收。
3. `preview PROJECT --seconds 20`：`autopilot-render.mjs` 产出试听/观测样片与证据；B1–B3 还需要独立的样片效果评审，B4 另有画面/字幕独立检查。
4. `approve PROJECT --note "..."`：必须基于用户真实批准与相应样片评审记录，不可模拟通过。
5. `render PROJECT`：复核预览文件哈希、批准与工程指纹，再执行整片渲染；仍需成片声音、画面与版权复核。
6. B5/B6 使用各自代码入口，B7 使用专用像素剧场路径；上述顺序不能直接照搬全部风格。

## 工程价值、边界与复核状态

- **适用：** 机器人论文/知识库的概念讲解视频、产品故事化介绍、匿名频道短片。该项目用于**内容生产**，不提供机器人运动策略、动力学或 Sim2Real 工具。
- **适用性对照：** [video-shotcraft](../../wiki/entities/video-shotcraft.md) 以产品宣发镜头与 Remotion 为主；Stickman Video Director 以火柴人叙事/多风格 Agent 工作流为主；[HyperFrames](../../wiki/entities/hyperframes.md) 是 HTML/CSS 编程视频框架。
- **明确限制：** 「可免费使用 Skill」不等于图像/TTS/API 免费；A 需要生成服务与授权，B 需要实际本地素材、音画复核；音效/音乐对外分发须核实原始许可。
- **未验证事项：** 本次未安装依赖、执行 `node --test`、真实渲染 MP4、观察样片，也未验证所有浏览器服务登录和付费功能。资料核对与软件实际验收应分开表述。

## 参考来源

- [README.md](https://github.com/kaomei/stickman-video-director/blob/main/README.md)
- [SKILL.md](https://github.com/kaomei/stickman-video-director/blob/main/skills/directing-stickman-videos/SKILL.md)
- [A/B 模式边界](https://github.com/kaomei/stickman-video-director/blob/main/skills/directing-stickman-videos/references/mode-execution-boundaries.md)
- [B 模式运行时文档](https://github.com/kaomei/stickman-video-director/blob/main/skills/directing-stickman-videos/references/autopilot-runtime.md)
- [运行时 CLI](https://github.com/kaomei/stickman-video-director/blob/main/skills/directing-stickman-videos/scripts/autopilot.mjs)
- [项目 LICENSE](https://github.com/kaomei/stickman-video-director/blob/main/LICENSE)
