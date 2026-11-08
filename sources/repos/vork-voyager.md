# Vork（voyager-so/vork）官方源码、Benchmark 与授权条款

> 来源归档（ingest）｜官方仓库 README、源码、实测报告与 LICENSE

- **类型：** repo / agent-native GPU renderer / 3D / VFX
- **项目：** Vork — A headless 3D and VFX engine built for agents（Vork：为 AI Agent 构建的无界面三维与视效引擎）
- **官方仓库：** https://github.com/voyager-so/vork
- **项目页：** https://voyager.so/vork
- **官方原始说明：** https://github.com/voyager-so/vork/blob/main/README.md
- **基准结果：** https://github.com/voyager-so/vork/blob/main/benchmarks/RESULTS.md
- **许可证：** https://github.com/voyager-so/vork/blob/main/LICENSE
- **研究预览：** v0.1.0；以 2026-11-08 可访问版本为准
- **核对范围：** 已通读官方 README.md（包括机制、例子、安装、速度、局限与复现说明）、benchmarks/RESULTS.md 全部结果表和 LICENSE 全文；逐段核对源码 crates/vork-cli/src/main.rs 的命令分派、crates/vork-render/src/shot.rs 的 Shot::render 与逐帧渲染链路，阅读 crates/vork-render/src/lib.rs 模块说明、crates/vork-scene/src/lib.rs、crates/vork-render/src/frame.rs 的入口说明，并检查 examples/first-scene.json。没有宣称通读仓库所有实现文件，也未实际运行 GPU。
- **通读：** 已完成
- **入库日期：** 2026-11-08

## 官方源码模块与入口

| 路径 / 命令 | 从原始源码核对的实际职责 |
|---|---|
| `vork init my-shot` | `crates/vork-cli/src/init.rs` 创建离线示例工程；可选择 JSON、Python 或 TypeScript 场景编写 |
| `vork help` / `vork help programs --all` | `crates/vork-cli/src/help.rs` 输出由工具提供的场景字段与程序钩子文档 |
| `vork render scene.json --profile std/draft --out render.png` | `crates/vork-cli/src/main.rs` 的 clap `Command::Render` 解析、构造 `vork_render::shot::Shot` 并调用 `render()` |
| `crates/vork-scene/src/` | Scene v2：解析和校验场景；渲染设置、节点与场景数据 |
| `crates/vork-render/src/shot.rs` | `Shot::render_scoped`：场景与 Profile 设置、GPU 初始化、`install_with_options`、帧调度和输出 |
| `crates/vork-render/src/install.rs` / `graph.rs` | `InstalledScene` 资源安装、渲染 pass 图的编排 |
| `crates/vork-render/src/frame.rs` / `passes/` | `FrameSlot` 的 per-frame GPU 工作、提交与收集 |
| `crates/vork-slang/src/` | Slang 运行程序与自定义钩子：摄像机、材质、传播、合成、计算等 |
| `crates/vork-render/src/png_output.rs` / `crates/vork-encode/` | 静帧 PNG / EXR 等输出与视频编码 |

上述链接和函数位置均来自仓库公开代码，而非对通用渲染器架构的推测。

## 实际运行链路

1. Agent 生成 `scene.json` 或通过 Python/TypeScript SDK 生成声明式场景；若需要特殊效果则编写自定义 Slang 程序。
2. CLI 解析 `Command::Render`，创建 `Shot`。`render_scoped` 解析 Profile、调用 `load_with_settings` 并 `scene.validate()`；同时准备适配设备。
3. `Renderer::with_requirements` 选择满足要求的 GPU，`renderer.install_with_options` 编译/安装场景、材质、GPU 管线及资源，并创建 `FrameSlot`。
4. 帧循环调用 `slot.submit_outputs` 或 `slot.submit_gpu`，随后 `slot.collect_readback`；视频调用 `push_encoders`，图像路径调用 `FrameOutputWriter.push`。
5. 完成编码器与写入、保存报告 sidecar，`RenderedShot.report` 经 CLI 显示结果与耗时。Agent 是否看图迭代由上层 Agent 驱动，并非 Vork 内置自治决策。

源码证据：
- [CLI `Command::Render`](https://github.com/voyager-so/vork/blob/main/crates/vork-cli/src/main.rs)
- [`Shot::render_scoped`](https://github.com/voyager-so/vork/blob/main/crates/vork-render/src/shot.rs)
- [渲染模块与 FrameSlot](https://github.com/voyager-so/vork/blob/main/crates/vork-render/src/lib.rs)
- [Scene v2](https://github.com/voyager-so/vork/blob/main/crates/vork-scene/src/lib.rs)

## 性能：原始报告的可复核数字和限制

| 场景 | Vork（官方 L4，秒） | Blender 比较 | 注释 |
|---|---:|---|---|
| 百万鸟 skim，1080p24 / 192 帧，draft 4 samples + OIDN | **96.942 s**（n=3） | EEVEE 4 samples **773.7–1082.5 s**，按有限热帧外推 | 约 8–11× 为**投影比值**，不是 EEVEE 完整电影的直接实测，也不保证画质等价 |
| 百万鸟 skim，16 samples EEVEE | 同上 | **1511.108 / 1817.625 s** 外推 | 两端采样数不同，不应宣传为严格加速 |
| 1080p 静图，final8 + OIDN | 不同场景 | EEVEE 64 samples、无降噪 | 已保留可匹配的 10 组为 **1.15–1.90×**，非同画质 |
| 1080p、240 帧完整片段 | 不同场景 | EEVEE 留存测量基线 | 7 组为 **1.14–2.32×**，编码及预热计时边界不同 |
| 老版本 path tracing | 不同场景 | Cycles | Vork 对部分 case **慢 1.99–5.81×** |

百鸟 skim 的额外 **setup 243.809 s** 和 **320×180 warmup 166.030 s** 不在 96.942 s 内；合计约 506.781 s（仍不含素材准备和下载）。`benchmarks/RESULTS.md` 说明原始日志与 contact sheets 未在仓库中提供；Blender 5.2.1 对比 helper 在相机准备时报错；重跑命令的 profile / seeds / denoise 配置与报告原始版本存在差异，独立复现尚不完整。**不能据此得出“24× 同画质通用渲染提速”。**

## 能力、硬件与授权边界

- **实现：** Rust 宿主 + Slang GPU 程序，底层经 Vulkan（NVIDIA）或 Metal（Apple Silicon）；CPU 仍承担解析、动画 tracks、I/O，GPU 执行程序与 ray queries。
- **渲染：** 默认混合光栅可见性和硬件光追照明，可选路径追踪；允许场景程序定制 camera rays、材质响应、传播、film、合成与 compute。
- **示例：** `examples/murmuration/`（鸟群）；`examples/black-hole/`（近似 Schwarzschild 弯曲光线）；`examples/vfx-artifact/`（实拍相机跟踪、遮挡与代理几何）；`examples/water/`（有限粒子流体实验）。
- **尚有限制：** fog / participating volumes 忽略并告警、复杂透明/大型场景内存风险、编译预热昂贵、Mac 可靠性尚未在该轮 Linux 测试复验；黑洞不是严格的广义相对论数值求解器。
- **设备：** NVIDIA GPU 需支持 Vulkan hardware ray queries；Apple Silicon 走 Metal。Ubuntu 22.04+ 是列明的 Linux 基础；Mac 包内 Slang 工具链的最低目标是 macOS 26。A100/H100 等无 RT 核心，不是合适替代。
- **许可证：** 主引擎是 **source-available（不是 open-source）**，研究预览许可允许内部修改及商业作品，限制修改版对外分发、再许可和向外部用户提供托管渲染；`integrations/blender/` 是独立 GPL-3.0-or-later 组件。不可把“公开源码”直接写成“开源可自由二次分发”。

## 知识库映射

- [Vork 官方站点与技术文章归档](../sites/vork-voyager.md)
- [Vork 主实体节点及源码时序图](../../wiki/entities/vork.md)
- [3D-as-Code 与机器人仿真选型](../../wiki/comparisons/3d-as-code-to-robot-simulation.md)

## 未做的验证

没有安装或运行 Vork，也没有 L4/NVIDIA/Metal 渲染复测；本页数字、限速与行为均为**官方公开可检查证据**，软件集成到 MuJoCo / Isaac Sim 等机器人动力学平台的可靠性未经官方基准验证。