# 技术栈项目执行清单 v22

最后更新：2026-05-13（V22 启动，基于 V21 完整交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v21.md`](tech-stack-next-phase-checklist-v21.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V21 交付基线 (V22 起点)

| 维度 | V21 状态 | V22 目标 |
|------|-----------|---------|
| 知识图谱节点 | 297 | **≥ 312** |
| 知识图谱边数 | 1933 | **≥ 2050** |
| 事实库 (CANONICAL_FACTS) | 140 条 | **≥ 155 条** |
| 社区结构 | 8 社区，最大社区占 46.1%（`community_quality_warning: true`） | **最大社区占比 ≤ 40%，warning 消除** |
| 技术专题 | 触觉与力觉闭环（Haptics） | **建立"动作重定向与角色化人形"专题** |

---

## P0: 自动化与工具链深度强化 (Engineering)

- [ ] **缩写/别名归一化检索**：
    - [ ] `scripts/search_wiki.py` 引入轻量缩写表（WBC/VLA/IL/RL/MPC/PPO/SAC/HQP/CBF/CLF 等），查询时与全称双向展开，并在 `print_results` 中提示"已展开为 …"。
- [x] **社区粒度二级拆分**：
    - [x] 优化 `scripts/generate_link_graph.py` 的社区检测：在 Locomotion 单一巨型社区（46.1%）内进一步用 Louvain `resolution > 1.0` 二级拆分，使 `largest_community_ratio ≤ 0.40` 且 `community_quality_warning` 转 `false`。
      - 实现：保留 Girvan-Newman 一级检测（`PRIMARY_COMMUNITY_CAP=8`），新增 `refine_oversized_communities` + 纯 Python `louvain_communities`（带 `resolution=1.15` 的 Reichardt-Bornholdt modularity），对占比 > 40% 且节点数 ≥ 30 的巨型社区做二级拆分；`MAX_COMMUNITIES` 提升至 16 容纳子社区命名。
      - 结果：`exports/graph-stats.json` 中 `community_count=17`、`largest_community_ratio=0.138`（Manipulation 42 / 304）、`community_quality_warning=false`；Locomotion 巨型社区拆出 WBC / RL / MPC / IL / Sim2Real / Isaac Gym / Humanoid / Unitree G1 等子社区。
- [ ] **方法-Query 闭环 Lint**：
    - [ ] `scripts/lint_wiki.py` 新增 `methods_without_practitioner_query` 检查：被超过 3 个其他页面引用的 `methods/` 必须存在至少一篇 `queries/` 操作指南或 `comparisons/` 对比页对应，否则给出"待落地"预警。

## P1: 动作重定向与角色化人形专题 (Quality)

- [ ] **动作重定向知识链 (+3)**：
    - [ ] `wiki/concepts/motion-retargeting-pipeline.md`（重定向流水线：MoCap → 骨架对齐 → IK/约束 → 物理可行性筛选 → 训练数据的端到端概念）。
    - [ ] `wiki/formalizations/motion-retargeting-objective.md`（重定向目标函数形式化：姿态相似项、接触/约束项、平衡项、关节限位项的数学组合）。
    - [ ] `wiki/comparisons/gmr-vs-nmr-vs-reactor.md`（GMR / NMR / ReActor 重定向方法谱系对比：监督 vs 优化 vs 物理感知 RL，输入形态、依赖、产物差异）。
- [ ] **角色化人形（Character Humanoid）边界澄清**：
    - [ ] `wiki/concepts/character-animation-vs-robotics.md`（角色动画 vs 机器人控制：动作风格化、表演意图与物理可控性之间的张力；面向 Disney Olaf / Roboto Origin / MotionCanvas 等案例）。

## P2: 抓取与操作感知深化 (Quantity)

- [ ] **抓取知识链 (+3)**：
    - [ ] `wiki/methods/grasp-pose-estimation.md`（抓取位姿估计：6-DoF 抓取检测、点云/RGBD 输入、AnyGrasp / GraspNet / Contact-GraspNet 谱系）。
    - [ ] `wiki/queries/grasp-policy-selection.md`（抓取策略选型 Query：开放场景 vs 已知物体、稀疏 vs 稠密抓取、几何 vs 学习方法）。
    - [ ] `wiki/comparisons/anygrasp-vs-graspnet.md`（AnyGrasp 与 GraspNet 家族对比：输入模态、训练数据、部署延迟与开放词汇支持）。
- [ ] **接触/操作交叉补强**：
    - [ ] 在 `wiki/concepts/contact-rich-manipulation.md` 与 `wiki/concepts/visuo-tactile-fusion.md` 中补"抓取→插装→精细操作"的级联引用，把 P1 触觉链路与 P2 抓取链路打通。

## P3: 交互层"关系视角"增强 (UX/UI)

- [ ] **详情页"关联类型分布"小条形图**：
    - [ ] 在 `docs/detail.html` 的"关联页面"区块新增按 `type`（method/concept/entity/formalization/...）聚类的横向条形小图，让读者一眼判断当前节点偏理论、偏工程还是偏实体。
- [ ] **图谱页"专题视图"切换器**：
    - [ ] `docs/graph.html` 增加下拉菜单，可选"全量 / 动作重定向 / 抓取 / 触觉与通信"三个子图过滤模式，复用 V21 微地图的同套 `path → type` 元数据。

---

## 验收标准 (Definition of DoD)

- [ ] `make lint`: 0 errors（含新引入的 `methods_without_practitioner_query` 检查全通过）。
- [ ] 知识图谱节点数 **≥ 312**，边数 **≥ 2050**（见 `exports/graph-stats.json`）。
- [ ] 事实库扩展至 **155 条** 以上（重点补 motion-retargeting / grasp-pose 矛盾检测规则）。
- [ ] `community_quality_warning` 在 `exports/graph-stats.json` 中变为 `false`。
- [ ] `log.md` 记录 V22 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
