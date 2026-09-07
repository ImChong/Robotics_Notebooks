# image-blaster（neilsonnn/image-blaster）

> 来源归档

- **标题：** image-blaster
- **类型：** repo（Claude Code Agent Skills + Node 资产管线 + React 预览器）
- **作者：** neilsonnn（Neilson）
- **链接：** https://github.com/neilsonnn/image-blaster
- **许可：** MIT
- **入库日期：** 2026-09-07
- **一句话说明：** 面向 **Claude Code** 的 **image-to-world** 技能集：从单张参考图经代理分析 → 动态物体 **Hunyuan3D**（FAL）→ 静态环境 **World Labs Marble**（`.spz` splat）→ **ElevenLabs SFX**（FAL），约 5 分钟内产出可嵌入 Unity/Unreal/Godot/Blender/Three.js 的网格 + splat + 音效资产包。
- **为什么值得保留：** 把 [Marble](../../wiki/entities/marble-world-model.md) / Hunyuan3D / FAL 等 **闭源 API** 编排成 **可版本化的 Agent Skills + 确定性 Node 脚本**；与 [img2threejs](../../wiki/entities/img2threejs.md)（程序化 Three.js 代码）和 [CAD Skills](../../wiki/entities/cad-skills.md)（STEP/URDF 制造链）形成 **「照片→可漫游 3D 世界」** 对照样本；README 明确 **机器人环境概念图 / 建筑渲染 / 关卡原型** 等用例。
- **沉淀到 wiki：** 是 → [`wiki/entities/image-blaster.md`](../../wiki/entities/image-blaster.md)

## 开源状态（步骤 2.5，2026-09-07）

| 组件 | 状态 |
|------|------|
| 本仓库（skills、Node 脚本、React viewer） | **已开源**（MIT） |
| World Labs Marble（`marble-1.1`） | **闭源 SaaS/API**；需 `WORLD_LABS_API_KEY` |
| FAL 后端（Hunyuan3D、图像编辑、ElevenLabs SFX） | **闭源 API**；需 `FAL_KEY` |
| 生成权重 / 训练代码 | **未开源**（经第三方 API 调用） |

## README 要点（归纳）

1. **Quickstart：** `git clone` → `cd image-blaster` → 在目录内启动 `claude` → 配置 World Labs + FAL API key → 将图放入 `input/` → 对 Claude 说 `blast it and confirm each step with me`。
2. **默认产物：** 动态物体 **`.glb` / `.obj`**（可单独拾取/推动的物体）、静态环境 **Gaussian splat `.spz`**、环境循环音与物体物理音效 **`.mp3`**。
3. **技能矩阵（`.claude/skills/`）：** `image-blast-uncover`（主分析 + 物体候选）、`image-blast-plate`（clean plate 决策）、`image-blast-3d`（Hunyuan3D 物体）、`image-blast-world`（Marble 静态环境）、`image-blast-sfx`（音效）、`image-blast-image-edit`（nano-banana / gpt-image-2 图像编辑）、`image-blast-project`（项目状态）、`image-blast-wildcard`（扩展）。
4. **子代理（`.claude/agents/`）：** 与上述技能一一对应的 fork 代理（如 `image-blast-world`），长任务与上下文隔离。
5. **确定性脚本：** `.claude/scripts/project/project-state.mjs` 管理 `worlds/<slug>/` 信封；`generate-world.mjs` 调 World Labs 并 **下载全部资产到本地**（前端禁止直连 provider URL）；`hunyuan-3d.mjs` / `fal-3d-provider.mjs` 走 FAL 队列。
6. **数据契约：** `worlds/<world>/image.json` 合并场景分析；`output/<object>/object.json` 仅存身份与溯源；生成状态与请求 JSON 与产物并列，不污染 object 真值文件。
7. **模型栈：** `marble-1.1`（环境）、`nano-banana`（默认图像编辑）、`gpt-image-2`（备选编辑）、`hunyuan-3d`（物体网格，可调 face-count / PBR / LowPoly）、`elevenlabs-sfx`（音效）。
8. **嵌入：** README 称可嵌入任意游戏引擎、DCC 或 Web 应用资产目录；仓内 `app/` 为 React + Vite 预览器（默认 `.claudeignore` 屏蔽，开发时可移除以让 Claude 改 viewer）。
9. **环境变量：** `.env.example` 仅 `WORLD_LABS_API_KEY`、`FAL_KEY`。

## 对 wiki 的映射

- **实体页：** [`wiki/entities/image-blaster.md`](../../wiki/entities/image-blaster.md) — Agent Skills 形态的 **单图→3D 世界资产包** 编排参考。
- **概念交叉：** [`wiki/methods/generative-world-models.md`](../../wiki/methods/generative-world-models.md) — 生成式环境 / splat 导出与机器人 Real2Sim 消费边界。
- **相邻实体：** [`wiki/entities/marble-world-model.md`](../../wiki/entities/marble-world-model.md)（环境生成 API）、[`wiki/entities/spark-3dgs-renderer.md`](../../wiki/entities/spark-3dgs-renderer.md)（`.spz` Web 运行时）、[`wiki/entities/img2threejs.md`](../../wiki/entities/img2threejs.md)（图像→程序化 Three.js，无外部 3D API）、[`wiki/entities/cad-skills.md`](../../wiki/entities/cad-skills.md)（制造向 CAD skills）、[`wiki/entities/video-shotcraft.md`](../../wiki/entities/video-shotcraft.md)（另一垂直 Agent Skills 样本）。

## 备注（维护者）

- 仓库 stars 高（2026-09-07 约 4.8k）但 **topics 为空**；以 README 与 `.claude/skills/*/SKILL.md` 为运行时规约真值。
- **费用与配额** 完全取决于 World Labs credits 与 FAL 计费；技能正文不含成本估算。
- 物体/环境 **物理可信度、碰撞体质量、尺度标定** 未做机器人 sim-ready 承诺；与 SimFoundry / NuRec 类管线对照时须写明「外观资产源」。
