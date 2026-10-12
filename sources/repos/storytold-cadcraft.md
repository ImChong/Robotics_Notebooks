# CADCraft（storytold/cadcraft）

> 来源归档（官方开源仓）

- **标题：** CADCraft: Computer-aided design and drafting, reimagined in pure Rust
- **类型：** repo（桌面 / Web CAD 应用 + Agent 工具接口）
- **代码：** <https://github.com/storytold/cadcraft>
- **项目页：** <https://getartcraft.com/apps/cadcraft>
- **许可：** MIT OR Apache-2.0（仓库双许可）
- **实现：** Rust workspace、egui；桌面原生应用、命令行工具与 WebAssembly 应用
- **入库 / 核查日期：** 2026-10-12
- **版本核查：** 项目页下载区域显示 0.5.0；当前仓库 `Cargo.toml` workspace 版本为 0.6.2。分别记录页面提供的发行下载版本与主分支元数据，不假定两者是同一 release。
- **一句话说明：** 开源原生 CAD 程序，提供 AutoCAD 风格命令行、DXF/DWG 读写、参数约束与图层布局，并允许代理通过 cadcraft-cli / MCP 执行命令、查询和渲染图纸。

## 项目定位

CADCraft 是可独立使用的 CAD 编辑器，不是只生成 CAD 模型的 LLM 服务，也不是把 MCP 接到现有 FreeCAD 会话上的桥接插件。它提供 GUI、CLI、WebAssembly 三类运行入口，共用 Rust 几何、图纸数据库、渲染、IO 与命令引擎。

## 仓库可运行入口

- **桌面：** `cargo run --release -p cadcraft -- --sample`
- **CLI：** 构建后运行 `cadcraft-cli`；README 列有 `info`、`convert`、`run`、`commands` 与 `mcp` 子命令。
- **MCP：** `cadcraft-cli mcp` 启动 stdio MCP 服务；也可用 `mcp --connect 127.0.0.1:7979` 连接到由桌面应用 `--control` 选项启动的控制通道。
- **Web：** `apps/cadcraft-web` 的 WASM 前端可用 Trunk 本地运行，也可下载静态 web build 自行托管。
- **工程校验：** 仓库的 `cargo xtask ci` 覆盖格式、clippy、测试、资源署名、crate 分层与 wasm build。

## 主要能力（按 README / 项目页）

- **交互绘图：** 命令提示与补全、对象捕捉、极轴 / 正交跟踪、直接距离输入、框选、grips、撤销重做。
- **编辑与几何：** 常见线、圆弧、样条等 2D 绘制；移动、复制、旋转、缩放、镜像、偏移、修剪、倒角、阵列等修改命令。
- **工程图：** 图层与属性面板、尺寸标注、多重引线、表格、文字、多行文字、剖面填充、块与属性。
- **约束与布局：** 几何 / 尺寸约束、参数表达式、冲突检测；paper space、viewports、页面设置和 PDF plot。
- **格式与导出：** DXF ASCII / binary 读取范围标为 R12–2018，README 记载写出 R2000；DWG 读取依赖开源 acadrust，格式表列到 R13–2018，写出标为 R2000；另支持 PDF / SVG / PNG 导出。
- **Agent 接口：** `command_line`、JSON `execute`、`inspect_drawing`、`query_entities`、`render`；连接 GUI 时另提供 `screenshot` 与 `ui_click`。

## 已知边界

README 将 CADCraft 定位为 early, fast development。它的文件格式支持不等于 AutoCAD 功能完全兼容：尚未建模的实体保存时不会保留；实际工程需按目标 DWG/DXF 文件测试往返兼容。读写版本应按 README 各自说明区分，不应把格式读入范围当作格式写出承诺。仓库未提供机器人专用的 URDF / 惯量校验流程；将图纸几何用于机器人仿真仍需额外导出、简化和验证。

## 与知识库页面的关系

- [CADCraft 实体页](../../wiki/entities/cadcraft.md) — 能力、Agent 接口、运行路径与边界。
- [CAD Skills](../../wiki/entities/cad-skills.md) — 对比可安装技能 / Python CLI 工作流与原生 CAD 应用内核。
- [文字生成 CAD](../../wiki/concepts/text-to-cad.md) — 概念层 CAD / LLM 方式谱系。
