# earthtojake / text-to-cad（CAD Skills）

> 来源归档

- **标题：** CAD Skills（仓库名 `text-to-cad`）
- **类型：** repo
- **维护者（GitHub 显示）：** earthtojake（Jake）
- **链接：** https://github.com/earthtojake/text-to-cad
- **文档站：** https://www.cadskills.xyz
- **在线 Demo：** https://demo.cadskills.xyz
- **许可：** MIT
- **入库日期：** 2026-06-15
- **一句话说明：** 面向 **CAD / 机器人 / 硬件设计代理** 的 **Agent Skills** 技能库：以 **STEP 优先** 的 **build123d** 参数化建模为核心，串联 **URDF / SRDF / SDF**、本地 **CAD Viewer**、**DXF**、**G-code 切片**、**SendCutSend / Bambu Labs 打印** 与 **step.parts** 标准件检索；提供 **10 项机械基准** 与 **Codex / Claude Code / Skills CLI** 安装路径。
- **沉淀到 wiki：** 是 → [`wiki/entities/cad-skills.md`](../../wiki/entities/cad-skills.md)

## 核心摘录（据 README 与 `skills/cad/SKILL.md`）

1. **技能矩阵（11 项）** — `CAD`（自然语言/图像→STEP，build123d 源码为真值）、`CAD Viewer`（本地浏览器预览 CAD/G-code/机器人文件）、`step.parts`（螺钉/轴承/电机等标准 STEP 检索）、`DXF`、`URDF`、`SRDF`（MoveIt2 规划组/碰撞规则）、`SDF`（仿真世界）、`SendCutSend`（上传前 DXF/STEP 检查）、`G-code`（FDM 切片 CLI）、`Bambu Labs`（本地打印任务）、`Implicit CAD`（实验性 SDF/GLSL）。
2. **STEP-first 闭环** — CAD skill 默认 **毫米、XY 基面、Z 上向**；以 `gen_step()` Python 源生成 STEP，经 `scripts/inspect`（测量/对齐/diff）与 **强制 snapshot 视觉复核** 后再 handoff 至 `$cad-viewer`；STL/3MF/GLB 为次级导出。
3. **机器人描述链** — URDF skill 以 `gen_urdf()` Python 为真值、生成时默认校验 XML/关节/网格引用；SRDF 补 MoveIt2 语义；SDF 面向 Gazebo/Ignition 类仿真；与 CAD skill 分工（几何归 CAD，描述归 URDF/SRDF/SDF）。
4. **安装** — 首选 `npx skills install earthtojake/text-to-cad`；亦支持 Codex / Claude Code 插件 marketplace；生产用 `main` 分支（含生成物），开发 PR 对 `develop`。
5. **基准** — `benchmarks/01`…`10` 覆盖标定块、法兰、L 支架、阶梯轴键槽、开放顶壳体、航空 clevis、径向发动机缸体、离心叶轮、螺旋楼梯、行星齿轮级等机械 prompt→STEP 样例（重资产经 Git LFS）。

## 对 wiki 的映射

- **实体页**：[`wiki/entities/cad-skills.md`](../../wiki/entities/cad-skills.md) — Agent Skills 形态的 **文字→CAD→机器人描述→制造/打印** 全链路参考实现。
- **概念交叉**：[`wiki/concepts/text-to-cad.md`](../../wiki/concepts/text-to-cad.md) — 与 LLM+CadQuery/build123d 脚本路线、STEP 真值与 Sim2Real 几何一致性讨论互链。
- **相邻实体**：[`wiki/entities/urdf-studio.md`](../../wiki/entities/urdf-studio.md)、[`wiki/entities/articraft.md`](../../wiki/entities/articraft.md)、[`wiki/entities/mattpocock-skills.md`](../../wiki/entities/mattpocock-skills.md)（通用编码 Agent Skills 对照）。

## 备注（维护者）

- 仓库 **About** 与 README 强调 **skills library** 而非单一 text-to-CAD 模型；抓取时以各 `skills/*/SKILL.md` 为运行时规约真值。
- **依赖版本、切片器 CLI、Bambu 网络与打印安全策略** 须以仓库 `requirements.txt` 与各 skill 文档当前版本为准；G-code / Bambu 技能含 **dry-run /  cautious start** 叙事，不可当作无人值守产线指令。
