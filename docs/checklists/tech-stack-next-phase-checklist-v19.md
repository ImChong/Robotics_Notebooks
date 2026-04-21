# 技术栈项目执行清单 v19

最后更新：2026-04-21（V19 启动，基于 V18 核心知识交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v18.md`](tech-stack-next-phase-checklist-v18.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V18 交付基线 (V19 起点)

| 维度 | V18 状态 | V19 目标 |
|------|-----------|---------|
| 知识图谱节点 | 144 | **≥ 160** |
| 知识图谱边数 | 830 | **≥ 1000** |
| 事实库 (CANONICAL_FACTS) | 105 条 | **≥ 120 条** |
| 世界模型专题 | 潜空间想象已建立 | **建立 生成式世界模型 (Generative World Models) 专题** |
| 交互体验 | 反向预览 + 阅读足迹 | **实现图谱多级邻居探索与搜索权重可视化** |

---

## P0: 自动化与工具链深度强化 (Engineering)

- [ ] **搜索权重可视化工具**：
    - [ ] 编写 `scripts/debug_search.py`，输入查询词后不仅打印结果，还显式输出每个页面的 BM25 分数、Tag 提权系数、PageType 系数，方便调优。
- [ ] **自动化事实库质量校验**：
    - [ ] 增强 `scripts/discover_facts.py`，支持检测 Wiki 页面中的 `pos_claims` 与 `neg_claims` 是否存在正则表达式重叠，减少 Lint 误报。
- [ ] **Lint 规则严谨化**：
    - [ ] `scripts/lint_wiki.py` 新增检测：所有 `formalizations/` 页面必须包含 `## 数学定义` 区块。

## P1: 世界模型 (World Models) 专题深化 (Quality)

- [ ] **建立生成式仿真知识链 (+3)**：
    - [ ] `wiki/methods/generative-data-augmentation.md` (生成式数据增强：利用扩散模型生成长尾失败场景数据)。
    - [ ] `wiki/concepts/video-as-simulation.md` (视频即仿真：UniSim 等交互式视频预测器的底层逻辑)。
    - [ ] `wiki/formalizations/probability-flow.md` (概率流形式化：从扩散模型到流匹配的统一视角)。

## P2: 真机部署实战 Playbook 补完 (Quantity)

- [ ] **发布高阶部署指南 (+3)**:
    - [ ] `wiki/queries/multi-view-sync-guide.md` (多视角相机同步与标定在具身训练中的最佳实践)。
    - [ ] `wiki/queries/policy-quantization-distillation.md` (机器人策略模型的量化与蒸馏：如何在低算力边缘端跑通 VLA)。
    - [ ] `wiki/entities/open-source-humanoid-software.md` (主流开源人形软件栈横向对比：Humanoid-Gym vs. Berkeley Humanoid)。
- [ ] **新增对比页 (+1)**:
    - [ ] `wiki/comparisons/ethercat-vs-canfd.md` (工业级通信总线对比：人形机器人底层实时性与拓扑结构选型)。

## P3: 交互层“深度连接”优化 (UX/UI)

- [ ] **图谱节点多级关系预览**：
    - [ ] 修改 `docs/graph.html`，支持点击节点时，不仅高亮直接邻居，还通过半透明显示二级邻居（2-hop neighbors）。
- [ ] **详情页反向引用“快速预览”**：
    - [ ] 完善 `main.js` 逻辑，在“关联项”区块悬停时，通过浮窗展示目标页面的摘要，无需跳转。

---

## 验收标准 (Definition of Done)

- [ ] `make lint`: 0 errors。
- [ ] 知识图谱节点数 **≥ 160**。
- [ ] 事实库扩展至 **120 条** 以上。
- [ ] `scripts/debug_search.py` 工具上线。
- [ ] `log.md` 记录 V19 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
