# img2threejs（hoainho/img2threejs）

> 来源归档

- **标题：** img2threejs — Image → procedural Three.js（code-only reconstruction）
- **类型：** repo（Agent Skill + Python stdlib forge 管线）
- **作者：** hoainho
- **链接：** https://github.com/hoainho/img2threejs
- **演示画廊 / 项目页：** https://hoainho.github.io/img2threejs-showcase/（源仓 [hoainho/img2threejs-showcase](https://github.com/hoainho/img2threejs-showcase)）
- **入库日期：** 2026-07-23
- **一句话说明：** 把单张参考图重建为 **纯代码、程序化、质量门控、可动画** 的 Three.js `Group` 工厂（TypeScript）；**reconstruction-by-code**，不是摄影测量、网格提取或艺术资源包下载。
- **开源状态：** **已开源** — MIT；主仓含 `SKILL.md`、`forge/`（Python 3.10+ 纯标准库脚本）、`grimoire/` 评审规约；画廊另仓展示生成物。
- **沉淀到 wiki：** 是 → [`wiki/entities/img2threejs.md`](../../wiki/entities/img2threejs.md)

## 为何值得保留

- **Agent Skills 垂直样本：** 与 [CAD Skills](earthtojake-text-to-cad.md)（制造向 STEP/URDF）、[GSAP Skills](greensock-gsap-skills.md)（Web 动效）并列，是 **图像→浏览器 3D 程序** 的 skill 规约 + 确定性脚本门控范例。
- **Token 效率设计可读：** 机械校验/门控/对比图打包下沉到 stdlib 脚本，模型只做视觉判断与当前 pass 的 codegen——对本站维护者理解「脚本 enforce、模型 judge」的 agent 管线有对照价值（见仓库 `docs/TOKEN_COST.md`）。
- **与仿真资产生成的选型边界：** 输出是 **Three.js 程序化模型 + sculptRuntime（pivots/sockets/colliders）**，**不是** URDF/MJCF/USD 仿真就绪关节资产；可与 [Articraft](mattzh72-articraft.md)、PhysForge 等路线对照，避免把「好看的 WebGL prop」误当成操作仿真资产。

## README / SKILL 要点（归纳）

- **定位：** *Rebuild the object in a reference image as a code-only, procedural, quality-gated, animation-ready Three.js model.*
- **运行时：** Agent-agnostic（Claude Code / Codex / OpenCode）；依赖宿主提供的 vision / browser MCP / 预览截图。
- **主体分类：** `object` | `character` | `hybrid`；物体走硬表面管线，角色走解剖感知轨（`grimoire/character/`）。
- **阶段门控（mermaid 主干）：** 参考图 → 探针/适合性 → Pre-Spec（类/复杂度/quality contract）→ ObjectSculptSpec → strict-quality → 锁定 build passes → 生成工厂 → 浏览器渲染截图 → 对比 sheet → agent vision 评审 → 自校正（refine-spec / refine-code）或继续。
- **Build passes（固定顺序）：** `blockout → structural-pass → form-refinement → material-pass → surface-pass → lighting-pass → interaction-pass → optimization-pass`。
- **产物：** `ObjectSculptSpec` JSON + `createXxxModel(spec, options)` TypeScript 工厂（`THREE.Group`，`root.userData.sculptRuntime`）。
- **安装：** `git clone … ~/.claude/skills/img2threejs`；调用 `/img2threejs …`。脚本零 pip 依赖。
- **版本叙事（README）：** v1.0–v1.2 已发（物体管线、detail inventory、人形角色生成）；v1.3 likeness maximization / v1.4 SkinnedMesh·glTF 为 roadmap。`SKILL.md` frontmatter 标 `version: 1.3.0`（以仓库当前文件为准）。
- **诚实边界：** 单图无法揭示隐藏面；角色为 stylized 重建而非 photoreal likeness；允许输出「达不到请求保真度」。
- **协议：** MIT。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/img2threejs.md`](../../wiki/entities/img2threejs.md) |
| 演示站点归档 | [`sources/sites/img2threejs-showcase.md`](../sites/img2threejs-showcase.md) |
| Agent Skills 对照（CAD） | [`wiki/entities/cad-skills.md`](../../wiki/entities/cad-skills.md) |
| 仿真就绪可关节资产对照 | [`wiki/entities/articraft.md`](../../wiki/entities/articraft.md) |
| Text-to-CAD / mesh 工具谱系 | [`wiki/concepts/text-to-cad.md`](../../wiki/concepts/text-to-cad.md) |
| Web 动效 Agent Skills | [`wiki/entities/gsap-skills.md`](../../wiki/entities/gsap-skills.md) |

## 与本站 sources 的其它锚点

- 演示站点：[img2threejs-showcase.md](../sites/img2threejs-showcase.md)
- CAD 技能对照：[earthtojake-text-to-cad.md](earthtojake-text-to-cad.md)
- 可关节仿真资产对照：[mattzh72-articraft.md](mattzh72-articraft.md)
- Web 动效技能对照：[greensock-gsap-skills.md](greensock-gsap-skills.md)
