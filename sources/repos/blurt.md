# blurt（屏幕口述反馈整理工具）

> 来源归档

- **标题：** blurt — screen recording and voice feedback for coding agents
- **类型：** repo
- **组织：** AGI Hunt
- **代码：** <https://github.com/AGIHunt/blurt>
- **项目页：** <https://blurt.agihunt.info/en/>
- **许可证：** MIT
- **入库日期：** 2026-10-26
- **一句话说明：** 录制屏幕并口述反馈，由 coding agent 将长录制拆成带画面定位的 bug、想法或待办，再供人工审阅、导出或直接修复。

---

## 仓库要点（README ingest 快照）

- **工作流：** 录制屏幕与语音 → 本地转写、抽取关键帧并生成条目 → 在 review 页面逐项分拣 → 导出到 Feishu/Lark、GitHub、Linear 或 Markdown，或让 agent 继续修复。
- **交互：** 支持框选录制区域；macOS 可用快捷键和屏幕圈画标注所指位置。README 称 agent 会根据指针位置选取对应画面，并为条目补充复现步骤、实际/预期结果及可能涉及的代码。
- **实现：** 脚本负责录制、语音转写、帧提取、审阅页和导出；agent 通过仓库内的 skill 处理内容。README 将语音识别描述为 local-first，默认不离开本机，也支持自带云端模型密钥。
- **平台：** macOS 提供原生录屏/菜单栏应用；Windows 与 Linux（X11）支持录制、转写、审阅和导出。托盘、画笔及系统音频等功能目前限 macOS。
- **适用场景：** 适合把软件界面走查、产品想法和研究观察转成可追溯的问题单或需求草案；它是通用工程工具，不是机器人专用算法或框架。

## 来源与边界

以上为官方 README 快照。平台支持、隐私默认项和工作流能力均按 README 描述记录；实际使用前应按目标操作系统及项目配置核对。项目页链接由 README 提供。
