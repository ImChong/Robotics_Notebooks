# 技术栈项目执行清单 v16

最后更新：2026-04-21（V16 启动，基于 V15 完整交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v15.md`](tech-stack-next-phase-checklist-v15.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V15 交付基线 (V16 起点)

| 维度 | V15 状态 | V16 目标 |
|------|-----------|---------|
| 知识图谱节点 | 123 | **≥ 140** |
| 知识图谱边数 | 739 | **≥ 900** |
| 社区分布 | 8 个 | **建立“大模型/VLA”专属社区** |
| 事实库 (CANONICAL_FACTS) | 71 条 | **≥ 85 条** |
| 搜索质量回归 | 36/37 (97%) | **通过率 100% (覆盖所有已交付用例)** |
| 跨模态知识链 | 仅触觉 | **建立“视觉-语言-动作”闭环链路** |

---

## P0: 自动化与工具链进化 (Engineering)

- [x] **语义搜索排序算法优化**：
    - [x] `scripts/search_wiki.py` 支持按 **Tag 权重** 提权（例如查询中含“对比”时，`comparison` 类型页面排名显著提升）。
- [ ] **自动化事实推荐 (Auto-Lint Expansion)**：
    - [ ] 编写一个实验性脚本 `scripts/discover_facts.py`，通过扫描 wiki 页面中的“一句话定义”或“核心假设”区块，自动提取潜在的新 `CANONICAL_FACTS` 条目供审核。
- [x] **Lint 规则补全**：
    - [x] `scripts/lint_wiki.py` 新增检测：`comparison` 页面必须包含 `<table>` 渲染块（V15 已支持表格渲染，需强制约束）。

## P1: 具身大模型 (VLA) 深度化 (Quality)

- [x] **新增 VLA 形式化数学定义 (+2)**：
    - [x] `wiki/formalizations/vla-tokenization.md` (动作分词与量化：从连续空间到离散 Token 的映射)。
    - [x] `wiki/formalizations/cross-modal-attention.md` (具身模型中的视-语-控交叉注意力机制)。
- [x] **扩充 Foundation Policy 核心页**：
    - [x] `wiki/methods/vla.md` (扩充至 600 字以上，涵盖 Open X-Embodiment 数据集的影响)。
    - [x] `wiki/methods/π0-policy.md` (DeepMind 最新 π₀ 策略的底层原理剖析)。

## P2: 灵巧操作 (Dexterity) 社区补完 (Quantity)

- [x] **建立灵巧手知识链 (Dexterous Manipulation, +3)**:
    - [x] `wiki/entities/allegro-hand.md` (科研级灵巧手标杆)。
    - [x] `wiki/concepts/dexterous-kinematics.md` (灵巧手运动学：多指闭链约束与工作空间)。
    - [x] `wiki/methods/in-hand-reorientation.md` (手内重定向：物体在掌心的灵巧翻转算法)。
- [x] **新增实践指南 (Query, +2)**:
    - [x] `wiki/queries/dexterous-data-collection-guide.md` (如何使用 Shadow Hand 或灵巧手遥控采集高质量数据)。
    - [x] `wiki/queries/multimodal-fusion-tricks.md` (视、触、本体感受多模态融合的权重分配技巧)。

## P3: 前端交互与沉浸式阅读 (UX)

- [ ] **图谱社区高亮 (Community Highlight)**：在图谱视图中增加一键高亮特定社区（如“Locomotion 社区”）的功能，淡化无关节点。
- [ ] **知识足迹 (Breadcrumbs)**：在 `detail.html` 中通过 SessionStorage 记录最近访问的 5 个页面，实现底部的“最近阅读”快速跳转。
- [ ] **移动端侧边栏优化**：优化图谱侧边栏在竖屏下的交互体验，支持上下滑动切换面板。

---

## 验收标准 (Definition of Done)

- [ ] `make lint`: 0 errors。
- [ ] 知识图谱节点数 **≥ 140**，边数 **≥ 900**。
- [ ] `python3 scripts/eval_search_quality.py` 通过率 **100%**。
- [ ] 所有 `methods/` 页面均有指向其数学原理的 `formalization/` 链接。
- [ ] `log.md` 记录 V16 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
