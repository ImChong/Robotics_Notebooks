# phone-harness/phone-harness — AI Agent 手机交互工具

- **类型：** repo（可运行 CLI + Python helpers + Agent Skill）
- **官方仓库：** https://github.com/phone-harness/phone-harness
- **官网及文档：** https://phone-harness.com/ 、 https://phone-harness.com/docs/introduction
- **代码：** https://github.com/phone-harness/phone-harness
- **许可证：** MIT（仓库 LICENSE 与 GitHub 仓库页面）
- **核验日期：** 2026-11-18
- **关联官网归档：** [Phone Harness 官方站点](../sites/phone-harness.md)
- **知识实体：** [Phone Harness](../../wiki/entities/phone-harness.md)
- **核对范围：** 官方 README.md 全文、install.md 安装及模式说明、pyproject.toml、src/phone_harness/run.py CLI/脚本执行入口、helpers.py 交互助手、transport.py 抽象与 connect 选择、android.py 的 ADB 后端、cloud.py 云会话说明；以上按 2026-11-18 main 分支所见核对，未逐文件审计全部源码。
- **通读：** 已完成

## 官方定位

项目让 Claude Code、Codex 等外部 Agent 借 CLI 或 Skill 使用手机，提供读取屏幕文字、截图、点击、滑动、输入、打开应用等能力；不是独立 LLM、机器人运动控制器，也不是手机端必须安装的代理 App。

| 连接方式 | 实现与系统要求 |
| --- | --- |
| 本地 iPhone | macOS 15+ 的 iPhone Mirroring + iOS 18+ 配对 iPhone；macOS Accessibility 与 Screen Recording 权限；图像文字以 Apple Vision OCR 读取 |
| 本地 Android | macOS/Linux/Windows + adb；USB 或无线调试；uiautomator accessibility tree 提供文本及控件位置 |
| Cloud Android | 官方托管 Android，经 adb 接入；需账号、API 会话及 adb 网络连接 |
| Cloud iPhone | 官方云端真实 iPhone，HTTPS 控制，无需本机 Mac |

## 技术实现（源码核对）

1. `pyproject.toml` 定义包入口 `phone-harness = phone_harness.run:main`；安装包当前版本声明为 `0.3.3`（入库时源码所示，后续可能变化）。
2. `run.py` 解析 CLI 子命令；从 stdin 接收 Python 脚本并调用 `_exec_script()`，使用 `_script_globals()` 注入 `helpers.py` 函数，再执行脚本。CLI 有 `doctor`、`skill`、`android`、`cloud`、`config`、`update`。
3. `helpers.py` 提供 `screen()`、`find_text()`、`tap_text()`、`screenshot()`、`open_app()`、`type_text()` 等封装；核心 `send(op, **kw)` 交给 `transport.connect()` 产生的设备后端。
4. `transport.py` 定义语义 op 如 `screen.capture`、`screen.text`、`input.tap`、`input.text`、`apps.launch`，支持通过 `supports(op)` 判断能力；不把各平台差异伪装成完全等价。
5. `android.py` 使用 adb 截图、`uiautomator dump` 辅助 UI 文本和点击；`cloud.py` 负责账号、会话、租用以及连接路由，云 iPhone 的实际远端操作由 `remote.py` 完成（后者未详细审计）。

## 官方安装与最小使用

```bash
uv tool install --python 3.12 --upgrade --force phone-harness
phone-harness doctor android
phone-harness <<'PY'
open_app("Notes")
tap_text("New Note")
type_text("hello from the harness")
print([o["text"] for o in screen()][:10])
PY
```

- 安装说明：`uv` 推荐 Python 3.12；包的 `requires-python` 为 `>=3.10`。不同操作系统与手机需先按 `install.md`、`setup/` 完成配对与权限步骤，示例不代表已连接真机。
- 可通过 `phone-harness skill` 取得 Agent Skill 内容；MCP 入口 `https://mcp.phone-harness.com/mcp` 官方注明目前仅适用 Cloud Android / Cloud iPhone。
- 避免让手机上的敏感信息和验证码无审批地进入 Agent 日志或下游系统；此为部署安全建议，非上游提供的正式安全认证。

## 已知边界

- iPhone 解锁可能暂停镜像；Android PIN 锁屏需要用户交互。
- OCR 不能可靠识别无文字图标，需截图配合视觉模型；Android accessibility tree 对某些 WebView / 特殊 UI 覆盖不足。
- 官方 README 列明不支持多指、相机和 Face ID 流程，DRM 视频可能显示黑屏。
- 云手机是单独收费的托管服务，不应与 MIT 开源本地 CLI 混为一谈；本次未建立云服务实测会话。
- 项目侧未给出机器人硬实时闭环、工业安全认证或 ROS 2 控制接口的证据。

## 映射与来源

- [Phone Harness 独立知识节点](../../wiki/entities/phone-harness.md)
- [官方 README](https://github.com/phone-harness/phone-harness/blob/main/README.md)
- [安装文档](https://github.com/phone-harness/phone-harness/blob/main/install.md)
- [CLI 源码](https://github.com/phone-harness/phone-harness/blob/main/src/phone_harness/run.py)
- [helpers 源码](https://github.com/phone-harness/phone-harness/blob/main/src/phone_harness/helpers.py)
- [transport 源码](https://github.com/phone-harness/phone-harness/blob/main/src/phone_harness/transport.py)
- [Android 源码](https://github.com/phone-harness/phone-harness/blob/main/src/phone_harness/android.py)
- [Cloud 源码](https://github.com/phone-harness/phone-harness/blob/main/src/phone_harness/cloud.py)
