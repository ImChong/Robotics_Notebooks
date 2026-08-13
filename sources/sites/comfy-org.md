# Comfy 官网（comfy.org）

- **类型**：网站 / 产品主页
- **入口**：<https://comfy.org/>（GitHub `homepage` 写 <https://www.comfy.org/>，二者指向同一产品站）
- **主体**：Comfy-Org（GitHub 组织）；产品品牌 **Comfy / ComfyUI**
- **代码：** <https://github.com/Comfy-Org/ComfyUI>（**已开源**，GPL-3.0，见 [仓库归档](../repos/comfyui.md)）
- **收录日期**：2026-08-13
- **抓取说明**：以 **2026-08-13** 对首页公开文案、产品分区与文档索引（[docs.comfy.org/llms.txt](https://docs.comfy.org/llms.txt)）的抓取为准；型号卡片与云套餐会随发布周期更新。

## 一句话

**Comfy** 自称面向视觉专业人士的 **Visual AI 创作引擎**：在无限画布上把模型、处理步骤与输出连成可检查的节点图；产品线覆盖 **Desktop（本地）**、**Cloud**、**API** 与 **Enterprise**，核心推理引擎即开源 **ComfyUI**。

## 为什么值得保留

- 本库已有 [3D Gen Studio](../../wiki/entities/3dgenstudio.md) 把 **ComfyUI** 当网格生产后端，但此前缺少官方仓与官网的独立溯源页，读者无法区分「编排层」与「节点执行引擎」。
- 官网把同一引擎拆成 **本地 / 云 / API / 企业** 交付形态，对实验室选型（离线 GPU vs 托管端点 vs Agent MCP）有直接对照价值。
- 首页强调 **60,000+ nodes**、社区工作流库与 Partner Nodes（闭源模型），与机器人侧「开源权重 + 可选付费 API」并存的现实一致，必须在 wiki 里写清边界。

## 开源与项目页核查（2026-08-13）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — 首页与文档指向 <https://github.com/Comfy-Org/ComfyUI>；许可 GPL-3.0 |
| **数据 / 权重** | 核心 **不捆绑** 大模型权重；工作流库给模板，权重由用户或 Cloud 侧提供 |
| **闭源/付费面** | **Comfy Cloud / API / Enterprise / Partner Nodes** 为商业层；文档允许 `--disable-api-nodes` 强制核心离线 |
| **Agent** | [Comfy MCP](https://docs.comfy.org/agent-tools/mcp.md) 为 **public beta**：Cloud MCP `https://cloud.comfy.org/mcp` + 本地开源连接 |
| **文档** | <https://docs.comfy.org/>（安装、节点概念、Manager、Cloud API、MCP/CLI） |

## 公开功能要点（来自官网首页，2026-08-13）

| 模块 | 代表能力 |
|------|----------|
| **节点画布** | 连接模型、处理步骤与输出；每步可见、可调；可从社区模板起步 |
| **App Mode** | 把复杂图折叠成简化应用视图，随时切回节点图 |
| **Comfy Workflows** | 浏览/remix 社区工作流（<https://comfy.org/workflows>） |
| **Comfy Desktop** | 在自有硬件上跑 ComfyUI |
| **Comfy Cloud** | 浏览器使用完整引擎（文档写明 RTX 6000 Pro 等托管 GPU） |
| **Comfy API** | 把工作流变成生产端点 |
| **Comfy Enterprise** | 组织内创作引擎的基础设施叙事 |
| **新模型卡片（当时）** | MiniMax H3：多模态 I/O、原生立体、最高 2K、约 5–15s/次；强调音频条件不被覆盖 |

首页口号还包括「Now turn your agent into a creative technologist」——与文档里的 MCP / CLI / in-app agent 一致。

## 对 wiki 的映射

- 升格页面：[wiki/entities/comfyui.md](../../wiki/entities/comfyui.md)
- 交叉：[wiki/entities/3dgenstudio.md](../../wiki/entities/3dgenstudio.md)、[wiki/concepts/diffusion-model.md](../../wiki/concepts/diffusion-model.md)、[wiki/methods/generative-data-augmentation.md](../../wiki/methods/generative-data-augmentation.md)、[wiki/entities/blender.md](../../wiki/entities/blender.md)

## 参考链接

- 官网：<https://comfy.org/>
- GitHub：<https://github.com/Comfy-Org/ComfyUI>
- 文档：<https://docs.comfy.org/>
- Cloud：<https://cloud.comfy.org>
- 工作流库：<https://comfy.org/workflows>
- 下载 Desktop：<https://www.comfy.org/download>
- Discord：<https://comfy.org/discord>
