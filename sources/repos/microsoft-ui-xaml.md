# WinUI（microsoft/microsoft-ui-xaml）

- **标题：** WinUI / Microsoft UI XAML
- **类型：** repo
- **来源：** Microsoft
- **链接：** <https://github.com/microsoft/microsoft-ui-xaml>
- **文档：** <https://learn.microsoft.com/windows/apps/winui/winui3/>
- **发布说明：** <https://aka.ms/winui-releasenotes>
- **路线图：** <https://aka.ms/winappsdk/plans>
- **入库日期：** 2026-08-31
- **一句话说明：** MIT 开源的 **WinUI 3** UI 层：Fluent Design 控件与样式，随 **Windows App SDK** 交付，供 C# / C++ 构建高性能 Windows 桌面应用；仓库亦含 WinUI 2（UWP）历史分支。
- **沉淀到 wiki：** 是 → [`wiki/entities/winui.md`](../../wiki/entities/winui.md)

## 开源状态核查（2026-08-31）

| 项 | 值 |
|----|-----|
| **开放程度** | **已开源** — 完整 C++/C# 源码、构建脚本、`Samples/` 与 `GettingStarted.md` 可公开获取；**外部 PR 代码贡献尚未开放**（README 警告，OSS 仍在推进） |
| Stars / Forks（API） | ~8,192 / — |
| 推荐开发分支 | `winui3/main`（WinUI 3 主线） |
| 历史分支 | `winui2/main`（WinUI 2 for UWP） |
| 主要语言 | C++ |
| 最新发行标签 | **winui3/release/2.4.0**（WinUI 3 in WinAppSDK 2.4.0，2026-08-13） |
| 许可 | **MIT**（仓库 `LICENSE`） |
| 运行时要求 | Windows 10 1809（Build 17763）及以上（含 Insider） |

步骤 2.5：无独立 `*.github.io` 项目页；以 GitHub README 与 [Microsoft Learn WinUI 文档](https://learn.microsoft.com/windows/apps/winui/winui3/) 为准。**源码已开源**；日常应用开发通过 **Windows App SDK NuGet** 消费预编译包，不必自建本仓。

## 仓库概况（2026-08-31 API / README）

| 字段 | 值 |
|------|-----|
| 描述 | WinUI: a modern UI framework with a rich set of controls and styles to build dynamic and high-performing Windows applications. |
| 创建 | 2018-07-26 |
| 定位 | Windows 现代 UI 层；WinUI 3 为当前代，属 **Windows App SDK** 组件 |
| 示例应用 | [WinUI 3 Gallery](https://aka.ms/winui-gallery)（Microsoft Store） |

## README 摘要

- **Modern UI：** Fluent Design，无障碍与现代交互模式。
- **开发者可控：** C#（.NET）或 C++，目标 x86 / x64 / ARM。
- **Windows App SDK：** 与 Win32 原生应用并存，统一现代 Windows 平台 API。
- **系统级采用：** Windows 自带体验由 WinUI 驱动。

**入门文档（Learn）：**

- [Get started with WinUI](https://learn.microsoft.com/windows/apps/get-started/start-here)
- [Build your first WinUI app](https://learn.microsoft.com/windows/apps/tutorials/winui-notes/)
- [Migrate from UWP to Windows App SDK](https://learn.microsoft.com/windows/apps/windows-app-sdk/migrate-to-windows-app-sdk/migrate-to-windows-app-sdk-ovw)

**贡献状态（README 警告）：**

> 尚未接受外部代码 PR；可本地 `init.cmd` + `build.cmd` 构建产品二进制，测试与贡献流程仍在完善。详见 [WinUI OSS Update](https://github.com/microsoft/microsoft-ui-xaml/discussions/10700)。

## 仓库结构要点（`winui3/main` tree）

| 路径 | 角色 |
|------|------|
| `init.cmd` / `init.ps1` | 初始化 VS 2022 MSBuild 环境 |
| `build.cmd` / `Build.cmd` | 主编译入口；`/c` 干净构建 |
| `OneTimeSetup.cmd` | 首次安装 VS 2022 构建组件 |
| `GettingStarted.md` | 克隆、`winui3/main` 检出、构建与产物路径 |
| `Microsoft.UI.Xaml-Product.sln` | 产品级解决方案 |
| `controls/` | WinUI 控件实现与 `controls/dev/` API 测试 |
| `dxaml/` | XAML 框架核心 |
| `build/` | 打包、NuGet spec（`Microsoft.WindowsAppSDK.WinUI.nuspec` 等） |
| `Samples/` | ChartApp、WinUISnoop、WinUIDesktop 等样例 |
| `docs/` | 外部文档镜像与贡献说明 |

构建产物示例（`GettingStarted.md`）：`%BuildArtifactsDir%\packaging\%Configuration%\runtimes\win10-%Platform%\native` 下的 `Microsoft.UI.Xaml.dll`、`Microsoft.UI.Xaml.Controls.dll` 等。

## 与机器人研究/工程的关联点

- **Windows 工控机操作员控制台：** 实验室在 **Windows x64 工控机** 上为遥操作、数据采集或策略调试做 **原生桌面 HMI**（多相机预览、状态面板、急停）时，WinUI 3 是微软官方现代 UI 栈，与 [ONNX Runtime](../../wiki/entities/onnxruntime.md) C# 推理、Win32 设备 SDK 同栈集成。
- **与 Linux 输入层分工：** Linux 侧 USB 手柄常走 [xpad](../../wiki/entities/xpad.md) + evdev；Windows 侧游戏手柄走 XInput / Windows.Gaming.Input，**上层 WinUI 面板**负责任务模式、参数与可视化——见 [Teleoperation](../../wiki/tasks/teleoperation.md)。
- **非机器人通用 UI：** 本仓是 **Windows 应用 UI 框架**，不是 ROS / 实时控制中间件；选型时应与 [RIO](../../wiki/entities/robot-io-rio.md)、[XRoboToolkit](../../wiki/entities/paper-xrobotoolkit.md) 等 **遥操作/IO 专用栈** 区分。

## 对 wiki 的映射

- 升格页面：[wiki/entities/winui.md](../../wiki/entities/winui.md)
- 交叉引用：[wiki/tasks/teleoperation.md](../../wiki/tasks/teleoperation.md)、[wiki/entities/onnxruntime.md](../../wiki/entities/onnxruntime.md)、[wiki/entities/paper-notebook-intuitive-gui-for-non-expert-teleoperation-of-hu.md](../../wiki/entities/paper-notebook-intuitive-gui-for-non-expert-teleoperation-of-hu.md)

## 参考链接

- 源码仓库：<https://github.com/microsoft/microsoft-ui-xaml>
- WinUI 3 文档：<https://learn.microsoft.com/windows/apps/winui/winui3/>
- Windows App SDK：<https://learn.microsoft.com/windows/apps/windows-app-sdk/>
- WinUI 3 Gallery：<https://aka.ms/winui-gallery>
- Release notes：<https://aka.ms/winui-releasenotes>
- Roadmap：<https://aka.ms/winappsdk/plans>
- OSS 状态讨论：<https://github.com/microsoft/microsoft-ui-xaml/discussions/10700>
