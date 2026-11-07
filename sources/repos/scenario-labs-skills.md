# Scenario Agent Skills（scenario-labs/skills）

> 来源归档（repo / Agent Skills 合集）

- **标题：** Scenario Agent Skills
- **类型：** repo
- **来源：** Scenario Inc.
- **原始链接：** https://github.com/scenario-labs/skills
- **许可：** MIT（LICENSE：Copyright (c) 2026 Scenario Inc.）
- **入库日期：** 2026-11-07
- **抓取方式：** 浅克隆（提交 `e7e19d5`，2026-11-07 提交）；只做静态阅读，**没有运行任何脚本**，没有连接 Scenario MCP。
- **核对范围：** 本次逐字读完：`README.md`（519 行）、`AGENTS.md`、`docs/decisions.md`、`INSTALL.md`、`SECURITY.md`；4 个 SKILL.md 全文（`scenario`、`scenario-video-to-motion`、`scenario-blender`、`scenario-3d`）与 `scenario/references/setup.md`；`scenario-meshy`、`scenario-walkable-room` 的开头与要点段；7 份家族 README（Blender、Maya、ZBrush、Unity、Unreal、Godot、Roblox）的 Status 段，Blender 与 Unity 的 README 读了全文。另用脚本（只静态解析，未运行）对全部 175 个 SKILL.md 做了 frontmatter、正文字数与目录统计，对 16 个带脚本的核心技能列出 78 个脚本文件及头部说明行，并对核心技能脚本 grep 了网络与子进程调用；对全仓库 grep 了 robot、URDF、MuJoCo、Isaac、ROS 等关键词。**未读：** 其余约 171 个 SKILL.md 的正文（合计约 2.6 MB，核心技能平均约 1,300 词、专家技能平均约 2,200 词）、各技能 `references/` 与 `scripts/` 的正文、`tests/`、`.github/workflows`、`.claude` 与 `.agents` 下 7 个开发用技能的正文。「通读」在此指：5 份顶层文档 + 全部 175 个 SKILL.md 正文逐字读完；因 SKILL.md 正文只读了约 4 份，故标未完成。
- **通读：** 未完成
- **一句话说明：** Scenario 为 AI 编程智能体（Claude Code、Cursor、Codex、Copilot 等）发布的 Agent Skills：教智能体通过 Scenario MCP 服务器生成图像、视频、音频、纹理、天空盒、3D 与定制模型；另有一组「专家工具」技能，驱动本机的 DCC 软件和游戏引擎。

## 结构（实测）

- 仓库约 1,928 个文件，`version.txt` 为 0.60.1，提交 `e7e19d5`；`find` 统计到 **175 个 SKILL.md**，分三组：
  - **核心技能 81 个：** `skills/scenario*` 顶层目录，经 Scenario MCP 工作（图像、视频、音频、3D、工作流、团队管理，另有按模型家族分的技能）；
  - **专家工具 87 个：** `skills/dcc/`（Blender 15、Maya 11、ZBrush 9）与 `skills/game-engines/`（Unity 14、Godot 14、Roblox 14、Unreal 10），不需要 Scenario MCP；
  - **仓库自用的开发技能 7 个：** `.agents/skills/`（skill-creator、skill-development、skills-validate、skills-pr-handle、pr-squash-message、pr-summary、pr-handle-comments），不属于对外发布的 168 个。
- README 技能表列出 168 个（81 核心 + 87 专家），与 `skills.sh.json` 分组、`INSTALL.md` 按角色 / 按目标的安装清单对应；`INSTALL.md` 的专家工具节未列 Roblox，README 列了。
- 技能格式遵循 Agent Skills 规范：frontmatter 只有 `name`、`description`、`license: MIT` 三项（脚本核对 168 个对外技能全部如此，且 `name` 等于目录名）；正文 1,000 词为目标、2,500 词为硬上限（粗数：核心技能 598 至 2,496 词，专家技能 1,776 至 2,497 词）。
- 20 个核心技能带 `references/`，16 个带 `scripts/`（共 78 个文件，Python 为主，另有 `.mjs`、`.swift`、`.sh`、`.html`）；`tests/` 下 25 项（含各专家家族的套件）。
- 安装命令 `npx skills add scenario-labs/skills`；MCP 端点 `https://mcp.scenario.com/mcp`；README 另给固定版本形式 `npx skills@1.7.4 add scenario-labs/skills#<commit>`。
- `AGENTS.md` 规定：核心技能只教 MCP 一条路径（不绕道 REST、SDK、CLI），不点名具体生成模型，只有平台工具（合成、剪切、缩放、切片等）和专用技能才固定写模型 id，其余一律先 `recommend` / `search`；每个询问、审批或计费步骤都要写明无人值守时的默认动作。

## 安全与发布设计（读到的原文）

- `SECURITY.md`：漏洞走私下渠道（support@scenario.com 或应用内支持），不开公开 issue；范围写明仓库含技能文档与「少量辅助脚本」、不运行服务；签名资产 URL 视同凭据，不得提交或贴进 issue / PR。
- `docs/decisions.md` 记录的取舍：正文字数上限来自本仓库自己而非外部规范（引用 Claude Code 压缩后只回灌每个技能前 5000 token、共享 25,000 token 池）；`recommend` 报价 48.2 CU 而实扣 55，所以要用 `dry_run`；技能名不带版本号；不提供「装全部」命令（因为每个已装技能的描述都会在启动时加载）；技能里不写 `npx skills add`（市场扫描曾把含该句的技能标为可能加载可变来源的代码，作者自述这只是嫌疑而非证实原因）；`AGENTS.md` 控制在 32 KiB 内；视频检查改用扫完整段的联系表（单帧抽检曾漏掉 11.0 秒处的瑕疵）。
- 静态观察（本库）：核心技能脚本里有网络访问——`scenario-text-overlay/scripts/pillow_render.py` 从 `fonts.gstatic.com` 取字体，`scenario-kinetic-music-video/scripts/audio_analysis.py` 从 `cloud.cp.jku.at` 的一个公开链接下载模型权重（`final0.ckpt`）并调用 demucs；不少脚本经 `subprocess` 调 ffmpeg / ffprobe。这些与「少量辅助脚本、不运行服务」的表述不矛盾，但比字面印象重；未运行。
- 发布面：`AGENTS.md` 规定 `.github/workflows/` 里每个 `uses:` 写完整 commit SHA（读到的 `ci.yml` 照此执行，其余 workflow 未逐个读）；release-please 出版本；README 带 OpenSSF Scorecard 徽章。

## 作者自述的验证情况（未独立复核）

各家族 README 的 Status 段（均为作者自述，评分是「书面场景盲评」、每场景一位评分者，作者自己注明只能当方向）：

| 家族 | 技能数 | 书面场景盲评（无技能 → 有技能） | 实测说明 |
|------|--------|-------------------------------|----------|
| Unity | 14 | 63%（113/180）→ 97%（174/180），12 场景 | Unity 6000.3.21f1 / macOS，447 项实测与检查 |
| Blender | 15 | 有技能 7 战全胜 | 12 个测试套件在 Blender 5.2.1 无头通过；blockout 脚本无无头套件，靠真机验证 |
| Maya | 11 | 61% → 91%，8 场景 | 脚本模块「离线测试」 |
| ZBrush | 9 | 43% → 91%，7 场景 | 代理桥接实测；各技能的完整真机测试被启动卡死挡住 |
| Unreal | 10 | 51% → 95%（69/135 → 128/135） | **引擎未装，没在 Unreal 里跑过**；1,534 项离线检查用假的 `unreal` 模块；README 说在首次真机运行前应视为未验证 |
| Godot | 14 | 79% → 92%（143/180 → 165/180），12 场景中 10 改善、1 持平、1 变差 | Godot 4.7.2 实测，分技能计数 |
| Roblox | 14 | 60% → 97%（108/180 → 174/180），12 场景 | Roblox Studio 0.743 实测；未发布、未测真实 DataStore 与多客户端 |

- 作者列出的未执行项包括手机安装、商店上传、Apple 签名、真实购买与广告等，各家族 README 各有清单。
- 顶层 README 对 Unreal 家族写的是「drive Unreal Engine 5.8 on macOS」，没有提示未在引擎里验证；该提示只在 `skills/game-engines/unreal/README.md`。

## 与机器人的关系（本库判断）

- **仓库没有机器人用例：** 对全仓库 `.md`、`.py`、`.json` 搜 URDF、MJCF、MuJoCo、Isaac、ROS、强化学习、sim-to-real，没有命中；`robot` 只作为玩具机器人道具出现（Godot 导入示例的 toy robot、Blender 导出示例 `robot.glb`、previs 例子里的角色）。
- **输出格式是内容创作格式：** `asset_download` 转 `glb` / `fbx` / `obj`，`scenario-3d` 列 GLB、FBX、OBJ、STL、VOX；`scenario-blender-export` 写 FBX、GLB、USD / USDZ、STL（并支持 OBJ、PLY 导入检查）。没有 URDF / MJCF / SDF 导出。
- **技能自己承认「看着像 ≠ 能用」：** `scenario-3d` 写明生成网格不等于游戏可用，部件分离、关节、材质、动画各是独立一步；真实尺寸需要 `autoSize`（默认关），`dimensions` 读到的是归一化网格（最长边 1.0）；Godot 管线技能实测 12 个 AI 生成的 GLB 比例任意（玩具机器人 1.96 m）、8 个用了 4.7.2 会静默丢弃的材质扩展。
- **碰撞只是引擎碰撞：** `scenario-blender-export` 的 `collision()` 生成凸包碰撞网格（上限 64 顶点）、`ground()` 设真实高度并把原点放在底面中心；没有质量、惯量、关节限位。
- **动作来自视频：** `scenario-video-to-motion` 做人体无标记动捕，输出 Mixamo / MetaHuman 风格骨架或自动绑定的角色，导出 Blender、Maya、Unreal、Unity、Roblox 的 FBX；原文写明动作提取针对人体，非双足生物需逐帧检查。绑定方面 `scenario-3d` 说人形有专用 rigger，非双足用 `rigType`（quadruped、hexapod、octopod、avian、serpentine、aquatic）。
- **本库推测（原文没有）：** 这些可当仿真场景外观资产或人形参考动作的上游素材，但进仿真前要自己补尺度、碰撞、质量惯量与关节，机器人关节骨架不在其覆盖范围内。

## 对 wiki 的映射

- [Scenario Agent Skills](../../wiki/entities/scenario-labs-skills.md)
