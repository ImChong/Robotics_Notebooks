# ComfyUI（Comfy-Org/ComfyUI）

- **标题：** ComfyUI
- **类型：** repo
- **来源：** Comfy-Org（历史命名空间 [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) 仍出现在 README 发行徽章与部分下载链）
- **链接：** <https://github.com/Comfy-Org/ComfyUI>
- **项目页：** <https://comfy.org/>（官网亦写 `www.comfy.org`）
- **文档：** <https://docs.comfy.org/>
- **入库日期：** 2026-08-13
- **一句话说明：** GPL-3.0 开源的 **节点图扩散/生成引擎**：GUI + HTTP/WebSocket API + 图执行后端，覆盖图像、视频、音频、3D 与文本工作流；不捆绑大模型权重。
- **沉淀到 wiki：** 是 → [`wiki/entities/comfyui.md`](../../wiki/entities/comfyui.md)

## 开源状态核查（2026-08-13）

| 项 | 值 |
|----|-----|
| **开放程度** | **已开源** — 完整 Python 核心、图执行、HTTP/WebSocket 服务、`script_examples/`、节点系统与测试均可公开获取 |
| Stars / Forks（API） | ~127,144 / ~14,976 |
| 默认分支 | `master` |
| 主要语言 | Python（PyTorch）；前端已拆至 [Comfy-Org/ComfyUI_frontend](https://github.com/Comfy-Org/ComfyUI_frontend)，经 PyPI `comfyui-frontend-package` 安装 |
| 最新稳定版 | **v0.32.0**（2026-08-11；`comfyui_version.py` / GitHub Releases） |
| 许可 | **GPL-3.0**（仓库 `LICENSE`） |
| 权重 / 数据 | **不自带** 扩散/视频大模型权重；需用户放入 `models/` 或走可选 **Partner / API nodes**（付费闭源模型入口，可用 `--disable-api-nodes` 强制离线） |
| 官网 | <https://comfy.org/> |
| 文档 | <https://docs.comfy.org/> |

步骤 2.5：项目页 [comfy.org](https://comfy.org/) 明确指向 Desktop / Cloud / API / Enterprise，并链到本仓与文档；源码以 GitHub 本仓为准，**已开源**。

## 仓库概况（2026-08-13 API / README）

| 字段 | 值 |
|------|-----|
| 描述 | The most powerful and modular diffusion model GUI, api and backend with a graph/nodes interface. |
| Topics | `ai`, `comfy`, `comfyui`, `python`, `pytorch`, `stable-diffusion` |
| 创建 | 2023-01-17 |
| Issue / PR | 公开（入库时 open issues ~4559） |

## README 摘要

> ComfyUI is the AI creation engine for visual professionals who demand control over every model, every parameter, and every output. Its powerful and modular node graph interface empowers creatives to generate images, videos, 3D models, audio, and more...

**安装入口（README，由浅到深）：**

1. [Desktop](https://www.comfy.org/download)（Windows / macOS，官方推荐新手）
2. Windows Portable（Nvidia / AMD / Intel 独立 7z）
3. `pip install comfy-cli && comfy install`
4. 手动：`git clone` → 按 GPU 装 PyTorch → `pip install -r requirements.txt` → `python main.py`

**产品三仓（README「Release Process」）：**

| 仓 | 角色 |
|----|------|
| **ComfyUI Core**（本仓） | 约两周一版稳定；Desktop 的底座 |
| [Comfy Desktop](https://github.com/Comfy-Org/Comfy-Desktop) | 桌面包，钉最新稳定核心（旧仓 `Comfy-Org/desktop` 已归档） |
| [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend) | Vue/TS 前端；约两周合入核心 |

## 仓库结构要点（2026-08-13 tree）

| 路径 | 角色 |
|------|------|
| `main.py` | 进程入口：启动服务、解析 CLI（含 `--enable-manager`、`--disable-api-nodes`） |
| `server.py` | `PromptServer`：HTTP REST + WebSocket |
| `execution.py` | 工作流 JSON → 依赖图执行、部分重跑与缓存 |
| `nodes.py` | 核心节点注册（`NODE_CLASS_MAPPINGS`） |
| `comfy/` | 模型加载、注意力、VRAM/设备管理 |
| `comfy_execution/` | 图执行子系统 |
| `comfy_api/` / `comfy_api_nodes/` | 新版节点 schema / 可选云端 Partner 节点 |
| `custom_nodes/` | 社区扩展落点 |
| `script_examples/` | 无 GUI 调 HTTP API 的脚本样本 |
| `openapi.yaml` | 本地/云 API 契约 |
| `folder_paths.py` / `models/` | 权重目录约定；可用 `extra_model_paths.yaml` 共享其它 UI 的模型盘 |
| `tests/` / `tests-unit/` | 回归测试 |

## 与机器人研究/工程的关联点

- **视觉合成数据：** 节点图可编排 ControlNet / inpaint / 背景替换 / 深度与分割，服务 [生成式数据增强](../../wiki/methods/generative-data-augmentation.md)；**不是** 接触动力学真值。
- **视频先验落地：** README 原生列出 Wan 2.1/2.2 等模板，是 [Wan](../../wiki/entities/paper-wan-video.md) 生态的常用推理宿主之一。
- **网格资产后端：** [3D Gen Studio](../../wiki/entities/3dgenstudio.md) 把本仓当 **ComfyUI Native** 生成运行时（Hunyuan3D 等），再导出 GLB/OBJ。
- **Agent 自动化：** 官方 [Comfy MCP](https://docs.comfy.org/agent-tools/mcp.md)（public beta）可连本地实例或 Cloud GPU，适合批跑工作流，仍须人工核物理一致性。
- **感知预处理：** 模板含 SAM 3/3.1、Depth Anything、RT-DETR 等，可当离线视觉工具链，而非实时控制环。

## 对 wiki 的映射

- 升格页面：[wiki/entities/comfyui.md](../../wiki/entities/comfyui.md)
- 交叉引用：[wiki/entities/3dgenstudio.md](../../wiki/entities/3dgenstudio.md)、[wiki/concepts/diffusion-model.md](../../wiki/concepts/diffusion-model.md)、[wiki/methods/generative-data-augmentation.md](../../wiki/methods/generative-data-augmentation.md)、[wiki/entities/paper-wan-video.md](../../wiki/entities/paper-wan-video.md)、[wiki/entities/blender.md](../../wiki/entities/blender.md)
- 项目页归档：[sources/sites/comfy-org.md](../sites/comfy-org.md)

## 参考链接

- 源码仓库：<https://github.com/Comfy-Org/ComfyUI>
- 官网：<https://comfy.org/>
- 文档：<https://docs.comfy.org/>
- 工作流库：<https://comfy.org/workflows>
- 节点注册表：<https://registry.comfy.org>
- Discord：<https://discord.com/invite/comfyorg>
