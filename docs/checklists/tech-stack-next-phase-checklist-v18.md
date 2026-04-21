# 技术栈项目执行清单 v18

最后更新：2026-04-21（V18 启动，基于 V17 完整交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v17.md`](tech-stack-next-phase-checklist-v17.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V17 交付基线 (V18 起点)

| 维度 | V17 状态 | V18 目标 |
|------|-----------|---------|
| 知识图谱节点 | 137 | **≥ 160** |
| 知识图谱边数 | 801 | **≥ 1000** |
| 事实库 (CANONICAL_FACTS) | 101 条 | **≥ 120 条** |
| 交互体验 | 社区焦点 + 阅读足迹 | **实现详情页反向引用预览** |
| 技术专题 | 具身数据流水线 | **建立“世界模型与生成式控制”专题** |

---

## P0: 自动化与工具链深度强化 (Engineering)

- [ ] **搜索回归用例库大扩容**：
    - [ ] 为 `schema/search-regression-cases.json` 新增 20+ 用例，确保 V15-V17 新增的灵巧操作、软件栈、中间件内容搜索准确。
- [ ] **反向引用预览逻辑 (Backlink Previews)**：
    - [ ] 修改 `docs/main.js`，在详情页底部的“关联项”区块，不仅显示标题，还显示指向该页面的**简短上下文片段**（类似 Obsidian 的反向引用预览）。
- [ ] **自动化事实库质量校验**：
    - [ ] 增强 `scripts/discover_facts.py`，支持检测 Wiki 页面中的 `pos_claims` 与 `neg_claims` 是否存在正则表达式重叠。

## P1: 世界模型 (World Models) 与生成式控制专题 (Quality)

- [ ] **建立世界模型知识链 (+3)**：
    - [ ] `wiki/concepts/latent-imagination.md` (潜空间想象：Dreamer 架构的核心原理)。
    - [ ] `wiki/methods/generative-world-models.md` (生成式世界模型：GAIA-1, UniSim 等具身视频生成器)。
    - [ ] `wiki/formalizations/variational-objective.md` (变分目标函数：世界模型训练中的 ELBO 与信息论基础)。
- [ ] **深化生成式动作表示**：
    - [ ] `wiki/methods/flow-matching-for-robotics.md` (流匹配在机器人动作生成中的应用，对比 Diffusion)。

## P2: 人形机器人真机部署实战 (Quantity)

- [ ] **发布真机部署 Playbook (+3)**:
    - [ ] `wiki/queries/humanoid-battery-thermal-management.md` (人形机器人高功率输出下的电池与热管理工程经验)。
    - [ ] `wiki/queries/field-robotics-troubleshooting.md` (野外/非结构化环境下足式机器人的现场排障指南)。
    - [ ] `wiki/entities/open-source-humanoid-hardware.md` (主流开源人形硬件方案对比：Roboto Origin vs. Berkeley Humanoid)。
- [ ] **新增对比页 (+1)**:
    - [ ] `wiki/comparisons/point-cloud-vs-heightmap.md` (足式机器人感知：点云直接处理 vs 高度图投影的精度与实时性对比)。

## P3: 交互层“沉浸感”再提升 (UX/UI)

- [ ] **图谱节点关系浮窗 (In-Graph Relation List)**：
    - [ ] 在图谱侧边栏增加“直接邻居”列表，点击节点时，不仅显示摘要，还列出所有入边和出边的标题。
- [ ] **全局搜索快捷键与聚焦**：
    - [ ] 增加 `/` 键自动聚焦搜索框，并在搜索结果中增加“分类过滤”气泡。

---

## 验收标准 (Definition of Done)

- [ ] `make lint`: 0 errors。
- [ ] 知识图谱节点数 **≥ 160**。
- [ ] 事实库扩展至 **120 条** 以上。
- [ ] 详情页“反向引用预览”功能上线。
- [ ] `log.md` 记录 V18 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
