# 具身产业库（Embodied Industry DB）

- **URL**：<https://github.com/MasashiToda1/embodied-industry-db>
- **类型**：repo / 结构化产业数据库 + 入库工具链
- **维护方**：MasashiToda1（独立维护；上游 README 致谢 ImChong/Robotics_Notebooks 与 RealXiaoze/humanoid-motion-intelligence）
- **收录日期**：2026-09-23
- **Stars / Forks（入库核查）**：3 ★ / 0 forks
- **许可**：数据与文档 **CC BY 4.0**；代码 **MIT**
- **在线站点**：<https://embodied.menily.ai>（GitHub Pages，纯静态）
- **当前版本**：V0.1（数据集刚起步；schema / 词表 / lint / 入库工具已有测试）

## 一句话

记录**哪些公司在用什么技术栈、什么时间做了什么事**的可核查公开数据库：事件 YAML + 受控词表 + 来源快照 + lint 门禁；技术定义单向引用 [Robotics_Notebooks](https://github.com/ImChong/Robotics_Notebooks)，产业线索可对照 [humanoid-motion-intelligence](https://github.com/RealXiaoze/humanoid-motion-intelligence) 但**不得导入其表格**。

## 为什么值得保留

- **形态互补本库**：Robotics_Notebooks 编译技术知识；本库做**产业技术栈收敛、商业信号与结构化时间线**——上游 DESIGN 明确单向引用、不复制 wiki 正文。
- **可机器查**：11 条受控轴、历史取值不删、事件带快照；比 markdown 主表更适合观察「哪条轴在收敛」。
- **商业信号层**：招投标、部署交付、公开定价、招聘、交付形态、目标场景——同类技术知识库普遍不收。
- **入库后台可复用**：微信文章 / 招投标 / arXiv 解析 + 手工一手事件路径，与本库 ingest / lint 思路同源。

## 开源核查（步骤 2.5，2026-09-23）

| 项 | 结论 |
|---|---|
| GitHub 可见性 | **已开源**（`main` 分支；MIT 代码 + CC BY 4.0 数据） |
| 项目页 / 在线查 | **有** → <https://embodied.menily.ai>（五视图：时间线、主体卡片、轴/收敛、图谱、商业信号） |
| 可运行入口 | **有** — `make setup` → `make lint` / `make build` / `make serve`（入库后台，默认 `:8790`）；`make site-serve` 本地预览静态站（`:8800`） |
| 训练/推理 | **不适用**（产业事实库，非算法仓） |
| 与 HMI 许可边界 | humanoid-motion-intelligence 为 **CC BY-NC-SA 4.0**；**不得导入其表格或编排**，只作线索索引回到一手来源 |

## 仓库结构（维护者视角）

| 目录 | 作用 |
|------|------|
| `registry/orgs/` | 主体身份（名字、别名、成立年、总部、产业层）— 自建骨架，不外包 |
| `registry/datasets/` | 数据集身份（规模、模态、许可、开放程度） |
| `events/` | 事实层 YAML（日期、精度、来源链接、快照、佐证状态） |
| `vocab/` | 受控词表（11 条轴合法取值 + keywords 轴候选） |
| `build/` | 编译产物（主体页、数据集页、合并 JSON） |
| `docs/` | 静态前端（Pages 部署时 `scripts/build_site.py` 生成 JSON） |
| `snapshots/` | 来源页面快照（进 git，证据随数据走） |

## 对 wiki 的映射

- 升格实体页：[embodied-industry-db](../../wiki/entities/embodied-industry-db.md)
- 站点归档：[embodied-menily-ai](../sites/embodied-menily-ai.md)
- 交叉：[humanoid-motion-intelligence](../../wiki/entities/humanoid-motion-intelligence.md)（线索索引，非依赖）
- 交叉：本库自身作为技术轴定义上游（DESIGN § 相关工作）
