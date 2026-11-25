# live-panel-skill — JSON / Excalidraw 到动态架构图的 Agent Skill 与渲染器

- **类型：** repo / agent skill / animated architecture diagram / visualization
- **上游维护者：** ythx-101
- **项目代码：** https://github.com/ythx-101/live-panel-skill
- **固定核对版本：** [main@d5aa86d8](https://github.com/ythx-101/live-panel-skill/tree/d5aa86d8a5729cea2c78f083df672977f168a147)（上游提交 2026-11-12；本次核对 2026-11-22）
- **许可证：** MIT（根目录 LICENSE；图形复刻示例须另遵原作者署名要求）
- **开源状态：** **已开源**。`SKILL.md`、Python 渲染与检验脚本、HTML 模板、可运行示例配置和测试均公开；无需远程 API Key。
- **核对范围：** 通读官方 README（英文使用指南及中文说明）、`SKILL.md`；逐文件核对 `scripts/render.py`、`scripts/check_frames.py`、`scripts/livepanel.py`（包括 `build_page`、`Chrome` 驱动）、`examples/minimal/config.json`；核对原始命令、平台依赖与主要输出链路。未在本次环境实际执行 Chromium / ffmpeg 的端到端视频编码或运行上游测试。
- **通读：** 已完成
- **对应 Wiki：** [live-panel-skill 独立实体](../../wiki/entities/ythx-101-live-panel-skill.md)

## 原文核心要点

live-panel 以一份 JSON 配置表达盒子、连线、状态机、计数器、滚动日志与运动节奏，生成一个**布局固定、状态持续变化**的架构图，而不是逐页出现的 PPT 动画。可通过 `SKILL.md` 让 Claude Code、Codex 等 agent 先整理系统、产出配置，再调用渲染器生成视频或循环网页。

| 维度 | 上游已公开的能力 / 边界 |
|---|---|
| 输入 | JSON 配置；已有 `.excalidraw` 可先用 `scripts/excalidraw_import.py` 转换为 JSON |
| 主题 | `terminal-dark` 终端监控风格、`light-pastel` 浅色信息图 |
| 画幅 | `4:5`（1200×1500）、`3:4`（1080×1440）、`1:1`（1080×1080）；位置以画布绝对像素编排 |
| 内置视图 | `ring`、`funnel`、`loop`、`excalidraw`；`nebula`、`timeline`、`tree`、`kpi` 为规划功能，不可当作已实现 |
| 输出 | `scripts/render.py` 输出 H.264 MP4；传入 `--html-out` 可另存自包含、循环运行的 HTML 页面 |
| 渲染驱动 | `scripts/livepanel.py::build_page` 内联配置与视图脚本；`Chrome` 对 `window.seek(t)` 逐帧求值截图，PNG 流通过 stdin 写给 ffmpeg |
| 质量校验 | `scripts/check_frames.py` 采样约 120 个时刻调用页面 `window.__check()` 检测排版溢出/重叠；`--repeat` 检验回放一致性 |
| 运行要求 | Python 3.8+、Chrome/Chromium、ffmpeg，Linux / macOS；Windows 需通过 WSL；基本 Python 渲染脚本无 pip 包依赖 |

## 官方执行入口（对照源码）

`scripts/render.py::main` 解析 `--config`/`--out`，定位 Chrome 和 ffmpeg，调用 `lp.build_page`，从配置读取宽高/时长/帧率，启动 `ffmpeg` 的 PNG 输入管道；随后 `lp.Chrome` 通过 DevTools pipe 打开自包含 HTML（`?manual`），每帧调用 `br.seek(i/fps)` 与 `br.shot()`，最后写出 H.264 MP4。

`scripts/check_frames.py::main` 则另建 HTML 页面，用 `br.seek(t)` + `window.__check()` 检测布局；开启 `--repeat` 后跳离时间点再返回，与原截图做像素比对。此检查不能代替人工查看 PNG 的视觉质量。

最小示例：
```bash
python3 scripts/render.py --config examples/minimal/config.json --out minimal.mp4 --html-out minimal.html
python3 scripts/check_frames.py --config examples/minimal/config.json --out-dir frames --repeat
```

## 局限与工程注意

1. **并非真实遥测。** 状态机、计数、日志展示按配置合成；不自动连接机器人硬件、ROS 2 Topic 或真实 Agent 运行指标。示意数字须标明 simulated / illustrative。
2. **非交互式监控产品。** 输出用于解释系统关系和制作传播素材；不要将动画当作实时监控证据。
3. **画幅非自动重排。** 配置使用绝对像素，不同宽高比应调整布局并重新运行 `check_frames.py`。
4. **系统依赖。** 缺少 Chrome/Chromium 或 ffmpeg 无法完成 MP4 渲染；Windows 原生缺少 POSIX 文件描述符转交，须用 WSL。
5. **署名。** README 和 SKILL 说明灵感来自 @thedelost，部分 examples 是对已有作品的重新绘制；复刻时保留原图作者和来源链接。
6. **验证范围。** 已核对源码入口，但没有声称本次在上游版本上实跑编码、渲染一致性或单元测试。

## 对知识库的映射

- [live-panel-skill 唯一实体](../../wiki/entities/ythx-101-live-panel-skill.md)：动态架构图的配置结构、运行路径、工程取舍。
- [Diagrams](../../wiki/entities/mingrammer-diagrams.md)：Python → Graphviz 的**静态**架构图，可与 live-panel 的**状态动画**对比。
- [Mermaid.js](../../wiki/entities/mermaid-js.md)：轻量文本式流程图/时序图，适合知识库文档中的静态可视化。

## 一手资料

- [官方 README（固定版本）](https://github.com/ythx-101/live-panel-skill/blob/d5aa86d8a5729cea2c78f083df672977f168a147/README.md)
- [官方 Agent Skill](https://github.com/ythx-101/live-panel-skill/blob/d5aa86d8a5729cea2c78f083df672977f168a147/SKILL.md)
- [scripts/render.py](https://github.com/ythx-101/live-panel-skill/blob/d5aa86d8a5729cea2c78f083df672977f168a147/scripts/render.py)
- [scripts/livepanel.py](https://github.com/ythx-101/live-panel-skill/blob/d5aa86d8a5729cea2c78f083df672977f168a147/scripts/livepanel.py)
- [scripts/check_frames.py](https://github.com/ythx-101/live-panel-skill/blob/d5aa86d8a5729cea2c78f083df672977f168a147/scripts/check_frames.py)
- [examples/minimal/config.json](https://github.com/ythx-101/live-panel-skill/blob/d5aa86d8a5729cea2c78f083df672977f168a147/examples/minimal/config.json)
