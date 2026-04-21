# 技术栈项目执行清单 v20

最后更新：2026-04-21（V20 启动，基于 V19 核心知识交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v19.md`](tech-stack-next-phase-checklist-v19.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V19 交付基线 (V20 起点)

| 维度 | V19 状态 | V20 目标 |
|------|-----------|---------|
| 知识图谱节点 | 147 | **≥ 170** |
| 知识图谱边数 | 842 | **≥ 1100** |
| 事实库 (CANONICAL_FACTS) | 105 条 | **≥ 130 条** |
| 搜索质量回归 | 100% | **通过率 100% (含新增世界模型用例)** |
| 技术专题 | 生成式世界模型 | **建立“具身 Scaling Law 与数据引擎”专题** |

---

## P0: 自动化与工具链深度强化 (Engineering)

- [x] **多模态搜索增强 (Vector + Semantic)**：
    - [x] `scripts/search_wiki.py` 实现多模态 Embedding 的无感切换，支持基于 `bge-small-zh` 的本地极速语义检索。
- [x] **自动化内链推荐补全**：
    - [x] 优化 `scripts/discover_facts.py`，支持根据页面间的 Tag 重合度自动生成 `related` 区块补全建议。
- [x] **图谱元数据一致性 Lint**：
    - [x] `scripts/lint_wiki.py` 新增检测：`entities/` 页面必须包含至少 2 个指向 `methods/` 或 `tasks/` 的出边。

## P1: 具身 Scaling Law 与数据引擎专题 (Quality)

- [x] **建立数据规模化知识链 (+3)**：
    - [x] `wiki/concepts/embodied-scaling-laws.md` (具身规模法则：数据量、模型参数与泛化能力的幂律关系)。
    - [x] `wiki/methods/auto-labeling-pipelines.md` (自动化标注流水线：利用 VLM 进行轨迹描述与成功率判定)。
    - [x] `wiki/formalizations/foundation-policy-alignment.md` (基础策略对齐：从人类演示到大规模混合数据的特征空间映射)。
- [x] **深化多模态 Transformer 结构**：
    - [x] `wiki/methods/unified-multimodal-tokens.md` (统一多模态 Token：视、语、控在同一个感知器中的量化与位置编码技巧)。

## P2: 硬件控制与工业中间件深化 (Quantity)

- [x] **发布工业级部署指南 (+3)**:
    - [x] `wiki/queries/ethercat-master-optimization.md` (如何基于 SOEM/IGH 实现 2kHz 以上的硬实时 EtherCAT 主站优化)。
    - [x] `wiki/queries/hardware-abstraction-layer.md` (硬件抽象层 HAL 设计：实现控制代码与不同执行器（QDD/SEA/液压）的彻底解耦)。
    - [x] `wiki/entities/open-source-humanoid-brains.md` (开源人形大脑选型：基于 Jetson Orin 与高性能 X86 工控机的算力平衡)。
- [x] **新增对比页 (+1)**:
    - [x] `wiki/comparisons/kalman-filter-vs-optimization-based-estimation.md` (状态估计：经典的 EKF/UKF 与基于优化的滑窗估计（VIO/WIO）选型对比)。

## P3: 交互层“智能搜索”提升 (UX/UI)

- [x] **语义搜索结果“解释性预览”**：
    - [x] 修改 `docs/main.js`，在搜索结果下方显示“命中原因”，例如：“核心标签命中: rl”或“标题命中: ppo”。
- [x] **图谱节点“相似性”聚合**：
    - [x] 在图谱视图中增加“磁吸模式”，自动将具有相同 `community` 或 `type` 的节点在视觉上靠拢。

---

## 验收标准 (Definition of Done)

- [ ] `make lint`: 0 errors。
- [ ] 知识图谱节点数 **≥ 170**。
- [ ] 事实库扩展至 **130 条** 以上。
- [ ] `scripts/search_wiki.py` 本地 Embedding 模式上线。
- [ ] `log.md` 记录 V20 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
