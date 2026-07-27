# 3D Gen Studio 官网（3dgenstudio.com）

- **类型**：网站 / 产品主页
- **入口**：<https://www.3dgenstudio.com/>
- **主体**：Bruno Fargnoli（visualbruno）；产品品牌 **3D Gen Studio**
- **代码：** <https://github.com/visualbruno/3DGenStudio>（已开源，见 [仓库归档](../repos/3dgenstudio.md)）
- **收录日期**：2026-07-27
- **抓取说明**：以 **2026-07-27** 对首页公开文案与导航结构的抓取为准；版本号与功能列表会随发布周期更新（首页标注 **v2.1.0**）。

## 一句话

**3D Gen Studio** 是面向 **AI 驱动 3D 网格生产** 的本地优先工作台：在单一可视化工作区中编排 **文生图 → 图像编辑 → 网格生成 → UV / 纹理 → GLB/OBJ 导出**，原生对接 **ComfyUI** 与外部 REST/GraphQL API。

## 为什么值得保留

- 机器人与具身仿真管线常缺 **「好看 / 可用的静态网格道具」** 批生产工具；本站把 **ComfyUI 工作流编排** 与 **Kanban / Node Graph 资产管线** 产品化，是「网格资产层」而非制造级 CAD 的公开样本。
- 与 [文字生成 CAD 工具索引](./text-to-cad-tools.md) 中的 **Tripo / Hunyuan3D / Meshy / Wonder3D** 同属 **text/image → mesh** 赛道，但定位是 **本地编排层 + 资产库**，而非单一生成模型 API。
- 与 [Blender](./blender-org.md)（全流程 DCC）、[Articraft](./articraft3d-github-io.md)（仿真就绪可关节资产）形成对照：本产品强调 **生成管线编排与 Mesh Editor**，不宣称 URDF/MJCF/STEP 工业真值。

## 开源与项目页核查（2026-07-27）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — 首页 CTA「Star on GitHub」与 clone 命令指向 <https://github.com/visualbruno/3DGenStudio> |
| **数据 / 权重** | 不自带大模型权重；依赖本地 **ComfyUI** 工作流与可选第三方 API（Tripo、Tencent Cloud、Hitem3D 等，以仓库 Changelog / Settings 为准） |
| **许可** | 仓库 `LICENSE` 为自定义 **3D Gen Studio Community License**（非 OSI MIT）：可本地/研究/商用生成；禁止转售软件本体与付费 SaaS 托管；**生成物（mesh/纹理）归用户** |
| **文档** | 站内 Wiki（随应用分发）；公开 docs 目录含 ComfyUI / Mesh Generation / MCP 说明 |

## 公开功能要点（来自官网首页，2026-07-27）

| 模块 | 代表能力 |
|------|----------|
| **Kanban Board** | 卡片沿 Images → Image Edit → Mesh Gen → Mesh Edit → Texturing 流转；拖拽、轮播变体、每卡触发 ComfyUI/API |
| **Node Graph** | 资产依赖可视化；节点检查器内联工作流参数；一键启动 |
| **Assets Library** | 统一管理 Images / Meshes / Workflows；版本追踪；PNG / GLB / OBJ / EXR 等格式徽章 |
| **ComfyUI Native** | 从卡片或节点直接跑任意 ComfyUI workflow，参数动态注入并链式传递输出 |
| **External API** | REST / GraphQL；卡片打 API 标签后调用第三方 3D 服务 |
| **Local-first** | 项目落盘本地，可 Git 同步；强调无云锁定 |
| **管线阶段** | 文/图输入 → 图像编辑 → 3D Mesh Gen → UV → Texturing → GLB/OBJ 导出 |

## 对 wiki 的映射

- 升格页面：[wiki/entities/3dgenstudio.md](../../wiki/entities/3dgenstudio.md)
- 交叉：[wiki/concepts/text-to-cad.md](../../wiki/concepts/text-to-cad.md)、[wiki/entities/blender.md](../../wiki/entities/blender.md)、[wiki/entities/articraft.md](../../wiki/entities/articraft.md)、[wiki/entities/img2threejs.md](../../wiki/entities/img2threejs.md)、[wiki/entities/freecad-mcp.md](../../wiki/entities/freecad-mcp.md)

## 参考链接

- 官网：<https://www.3dgenstudio.com/>
- GitHub：<https://github.com/visualbruno/3DGenStudio>
- Discord（README 徽章）：<https://discord.gg/kRFWNzFUSx>
- Privacy / Terms：<https://www.3dgenstudio.com/privacy-policy.html> · <https://www.3dgenstudio.com/terms-and-conditions.html>
