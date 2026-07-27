# visualbruno / 3DGenStudio

- **标题：** 3D Gen Studio
- **类型：** repo
- **来源：** Bruno Fargnoli（visualbruno）
- **链接：** <https://github.com/visualbruno/3DGenStudio>
- **项目页：** <https://www.3dgenstudio.com/>
- **入库日期：** 2026-07-27
- **一句话说明：** 开源 **AI 驱动 3D 网格生产工作台**（v2.1.0）：Kanban / Node Graph 编排 **ComfyUI** 与外部 API，覆盖文生图、图像编辑、mesh 生成、Mesh Editor（UV/Retopo/Rig/投影）、本地资产库与 **MCP Server**；导出 GLB / OBJ / FBX 等。
- **沉淀到 wiki：** 是 → [`wiki/entities/3dgenstudio.md`](../../wiki/entities/3dgenstudio.md)

## 开源状态核查（2026-07-27）

| 项 | 值 |
|----|-----|
| **开放程度** | **已开源** — 完整前后端、Electron 桌面包、`comfyui_workflows/` 示例、`python-server` mesh tools、`mcp/` 工具面均可公开获取 |
| Stars / Forks（API） | ~346 / ~60 |
| 默认分支 | `main` |
| 主要语言 | JavaScript（React + Vite + Express）；辅助 Python（`python-server`） |
| 版本（`package.json`） | **2.1.0** |
| 许可 | **3D Gen Studio Community License**（`LICENSE`，SPDX `NOASSERTION`）：允许个人/教育/研究/商用**使用与修改**；禁止转售软件本体、付费 SaaS 托管、付费捆绑分发；**生成输出归用户**；外部 API 费用与 ToS 由用户自担 |
| 官网 | <https://www.3dgenstudio.com/> |

## README 摘要

> Orchestrate complete 3D generation pipelines — from text-to-image, image editing, mesh generation, UV unwrapping, to texturing — all in a single visual workspace powered by ComfyUI and external APIs.

**安装入口（README）：**

```bash
git clone https://github.com/visualbruno/3DGenStudio.git
cd 3DGenStudio
npm install
npm run dev
# 另开终端：python-server 虚拟环境 + python main.py
```

前置：**Node.js / npm** + 已运行的 **ComfyUI**。开发态后端默认 `http://localhost:3001`，前端为 Vite。

**技术栈（README 表）：** React / Vite / Three.js / R3F；Express / Multer；SQLite / LowDB / 本地资产；ComfyUI + 外部 AI API。

## 仓库结构要点（2026-07-27 tree）

| 路径 | 角色 |
|------|------|
| `server.js` / Vite 前端 | 主应用：Kanban、Graph、Assets、Settings |
| `comfyui_workflows/` | 示例 API 工作流（Trellis2、Hy2.1、Qwen Edit、SAM 3.1、背景去除、上采样等） |
| `python-server/` | mesh-tools：Auto UV、Auto Retopo、缩略图等 |
| `mcp/` | MCP Server（HTTP `/mcp` + stdio 桥）；tools：projects / cards / graph / workflows / meshTools / assets / settings |
| `electron/` | 桌面安装包构建 |
| `docs/` | ComfyUI、Mesh Generation、MCP、Desktop Build 说明 |

## Mesh Editor 能力（README）

Texturing / Modeling / Sculpting / Painting / Displace / Projection（ComfyUI）/ Auto UV / Auto Retopo / Auto Rig（致谢 [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens)、[mesh2motion-app](https://github.com/Mesh2Motion/mesh2motion-app)）。

## MCP（docs/mcp.md）

- 应用启动后默认：`http://localhost:3001/mcp`（Streamable HTTP）
- 本机默认免 token；远程需 Settings 中 `mcp.token` Bearer
- 可驱动：建项目、构图、跑 ComfyUI、生成图/网格、mesh tools、导入导出项目

## 与机器人研究/工程的关联点

- **场景道具与外观资产**：为 Isaac / MuJoCo / 自研渲染管线快速产出 **GLB/OBJ 静态网格**；**不是** URDF/MJCF 关节真值，也不是 STEP/B-rep 制造模型。
- **与 Text-to-CAD / Articraft 选型边界**：制造向见 [text-to-cad](../../wiki/concepts/text-to-cad.md)；仿真就绪可关节见 [Articraft](../../wiki/entities/articraft.md)；本仓是 **ComfyUI 网格生产编排层**。
- **Agent 自动化：** 内置 MCP，可与 Claude / Cursor 等宿主串联批处理资产生成（对照 [FreeCAD MCP](../../wiki/entities/freecad-mcp.md) 的「遥控桌面 CAD」路线）。
- **DCC 下游：** 可再导入 [Blender](../../wiki/entities/blender.md) 做精细编辑，或导出 FBX 进 Unity/Unreal（Changelog：Mesh Preview FBX）。

## 对 wiki 的映射

- 升格页面：[wiki/entities/3dgenstudio.md](../../wiki/entities/3dgenstudio.md)
- 交叉引用：[wiki/concepts/text-to-cad.md](../../wiki/concepts/text-to-cad.md)、[wiki/entities/blender.md](../../wiki/entities/blender.md)、[wiki/entities/articraft.md](../../wiki/entities/articraft.md)、[wiki/entities/img2threejs.md](../../wiki/entities/img2threejs.md)、[wiki/entities/freecad-mcp.md](../../wiki/entities/freecad-mcp.md)
- 项目页归档：[sources/sites/3dgenstudio-com.md](../sites/3dgenstudio-com.md)

## 参考链接

- 源码仓库：<https://github.com/visualbruno/3DGenStudio>
- 官网：<https://www.3dgenstudio.com/>
- Discord：<https://discord.gg/kRFWNzFUSx>
- SkinTokens：<https://github.com/VAST-AI-Research/SkinTokens>
- mesh2motion-app：<https://github.com/Mesh2Motion/mesh2motion-app>
