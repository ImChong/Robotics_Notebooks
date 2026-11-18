# ffmpeg-skill：面向 Coding Agent 的本地视频音频工具箱

> 原始资料归档 · GitHub 项目 / Agent Skill

- **标题：** ffmpeg-skill — Give your coding agent a video editor
- **作者：** kajisho5（S. Kajiyama）
- **类型：** repo
- **仓库：** https://github.com/kajisho5/ffmpeg-skill
- **官方站点：** https://kajisho5.github.io/ffmpeg-skill/
- **主页归档：** [ffmpeg-skill 官方主页](../sites/kajisho5-ffmpeg-skill.md)
- **许可证：** MIT（以仓库 `LICENSE` 为准；输入媒体/字幕/音频的版权另行处理）
- **版本快照：** `package.json` 3.11.1（2026-11-18 读取 `main`，非承诺锁版本）
- **入库日期：** 2026-11-18
- **代码：** **已开源**；含 `bin/install.js`、`scripts/`、`mcp/server.py`、`templates/` 与测试。
- **知识页：** [ffmpeg-skill（Agent 视频编辑）](../../wiki/entities/ffmpeg-skill.md)
- **核对范围：** 通读官方 `README.md`、`SKILL.md`、`docs/contract.md`、`package.json`；阅读 `bin/install.js` 的安装/入口、`mcp/server.py` 的 JSON-RPC 分发及运行队列、`scripts/_contract.py` 的 ToolSpec、`scripts/probe.py`、`scripts/cut.py`、`scripts/check.py`、`scripts/render.py` 和 `scripts/_common/runner.py` 的主要运行接口；未逐行审计全部 42 个工具，也未执行媒体处理测试。
- **通读：** 已完成

## 官方能力与边界

- `SKILL.md` 是面向 Claude Code、Cursor、Codex 等代理的步骤规范；工具实现为本地 Python 脚本，底层调用 `ffmpeg` / `ffprobe`。
- 3.11.1 声明 **42 个工具**：剪辑/拼接、变速、静音段/口头赘词处理、重构图、字幕、叠层、HDR 转 SDR、音频归一化/同步、平台导出、批处理和交付检查。
- MCP：`npx ffmpeg-skill mcp` 通过 stdio 启动 `mcp/server.py`；`tools/list` 默认只列核心 12，`FFMPEG_SKILL_MCP_FULL=1` 才展示全部 42；未列出工具仍可用 `tools/call` 按名称调用。
- **SPEC 自生成执行契约：** `scripts/_contract.py` 从工具的 `argparse` 生成 `input_schema`，并提供能力依赖、输入输出、验证策略与 `doctor` 能力发现；契约版本为 2.0，软件版本独立演进。
- 主要处理流程：`probe` 实测 → 尽量免重编码 → `--dry-run --json` 规划 → 剪辑/渲染 → `check` 格式/平台约束 → `probe` 核实 → `look` 联系表供人工/代理查看。
- **非自主导演：** 项目不自行决定最佳剪辑点、题材、审美，也不会把源文件内容当作代理指令；没有可靠字幕文本时应使用可选本地 ASR 或明确提供字幕。
- **验证范围：** `check.py` 可校验时长、分辨率、FPS、编码、音量等可测属性；不能据此断言主观画面质量、语义正确或内容真实。

## 可运行入口与代码核对

| 入口 | 官方用途 | 核读事实 |
|---|---|---|
| `npx ffmpeg-skill` | 安装 Skill（默认 Claude Code） | `bin/install.js` 拷贝 SKILL、scripts、templates、mcp、references 等 |
| `npx ffmpeg-skill --cursor / --codex` | 其他代理环境安装 | 安装器解析显式目标和 `--project`、`--dir` |
| `npx ffmpeg-skill doctor` | FFmpeg 组件自检 | 调 `scripts/_contract.py doctor`，分别报告能力 |
| `npx ffmpeg-skill contract --json` | 获取工具机器契约 | `scripts/_contract.py` 从 argparse 与 ToolSpec 汇编 |
| `npx -y ffmpeg-skill mcp` | MCP stdio 服务器 | `mcp/server.py` 处理 `initialize`、`tools/list`、`tools/call` |
| `python3 scripts/probe.py input.mp4 --json` | 输入媒体探测 | `_common` 读取 `ffprobe` JSON |
| `python3 scripts/cut.py input.mp4 --start 0:10 --end 0:25 --json` | 裁片 | 默认优先 stream copy，必要时精确重编码，并报告时间误差/原因 |
| `python3 scripts/check.py output.mp4 --platform reels --json` | 交付校验 | 平台表来自 `_platforms.py`；输出 PASS/WARN/FAIL |
| `python3 scripts/render.py project.json --dry-run` | 多阶段工程规划 | 官方管线有 `cut`、`join`、`fit`、`caption`、`overlay`、`loudness`、`export`、`check` 等 |

源码调用链：`mcp/server.py::handle` 处理 `tools/call` → `call_tool` 通过 `build_argv` 转换结构化参数 → `_common.run_tool` 启动对应 `scripts/<name>.py` 子进程 → 脚本调用 `ffmpeg` 或 `ffprobe` → `--json` 结果回到 MCP。此处是**静态代码阅读**，不是实际运行验收。

## 部署与已知风险

- 需要 **FFmpeg 6.1+、Python 3.10+**；使用 `npx` 安装时还需 Node 16+。Python 工具仅依赖标准库，不代表系统无需安装 FFmpeg。
- Ubuntu 22.04 官方仓库通常提供 FFmpeg 4.4；只执行 `sudo apt install ffmpeg` 可能不满足 6.1+，需要核查版本并选择更新发行版或独立构建。
- `doctor --json` 的总体 `ok: false` 可能仅代表个别过滤器缺失；以每个工具的 `usable` 判断可用性。
- `--dry-run` 在测量类工具上可能仍运行只读分析；`--overwrite` 应只用于用户明确允许覆盖的输出。
- 默认单次执行超时 1800s；长时间编码可能触发超时，应根据官方说明调整参数。
- 本次未运行自带测试或视频实测；所有速度/效果为官方示例与源码声明，**不可当成独立测量**。

## 一手资料

- [README — 功能、安装、限制](https://github.com/kajisho5/ffmpeg-skill/blob/main/README.md)
- [SKILL.md — 代理操作流程与 42 工具映射](https://github.com/kajisho5/ffmpeg-skill/blob/main/SKILL.md)
- [MCP 服务器](https://github.com/kajisho5/ffmpeg-skill/blob/main/mcp/server.py)
- [执行契约实现](https://github.com/kajisho5/ffmpeg-skill/blob/main/scripts/_contract.py)
- [契约文档](https://github.com/kajisho5/ffmpeg-skill/blob/main/docs/contract.md)
- [安装器](https://github.com/kajisho5/ffmpeg-skill/blob/main/bin/install.js)
