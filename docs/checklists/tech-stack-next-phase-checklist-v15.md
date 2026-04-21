# 技术栈项目执行清单 v15

最后更新：2026-04-21（V15 启动，基于 V14 完整交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v14.md`](tech-stack-next-phase-checklist-v14.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md)

---

## V14 交付基线 (V15 起点)

| 维度 | V14 状态 | V15 目标 |
|------|-----------|---------|
| 知识图谱节点 | 116 | **≥ 130** |
| 知识图谱边数 | 720 | **≥ 850** |
| 社区分布 | 4 个 (Loco/WBC/Manip/LQR) | **强化 Manipulation 社区 (≥ 25 节点)** |
| 事实库 (CANONICAL_FACTS) | 53 条 | **≥ 70 条** |
| 薄弱页面 (< 200 字) | 5 个 | **0 个** |
| 搜索回归测试 | 26/26 (100%) | **新增 10+ 测试用例 (覆盖 V14 内容)** |
| 实体层覆盖 | 硬件平台为主 | **补全主流软件栈实体 (Drake/Pinocchio/MuJoCo)** |

---

## P0: 自动化与质量控制 (Engineering)

- [ ] **搜索质量基准扩展**：
    - [ ] 为 `scripts/eval_search_quality.py` 新增 10 条针对 V14 新页面的测试用例（如 ZMP、CMDP、Safe RL）。
    - [ ] 确保在 `make lint` 运行前自动执行质量评估。
- [ ] **CANONICAL_FACTS 深度扩展**：
    - [ ] 扩展至 70 条事实，重点覆盖：Impedance Control 原理、Trajectory Optimization 家族、ANYmal 硬件特征、WBC 调参经验。
- [ ] **自动化内链检查增强**：
    - [ ] `scripts/lint_wiki.py` 新增检测：`methods/` 页面必须包含指向其对应 `formalizations/` 或 `concepts/` 的链接。

## P1: 社区深度强化 (Quality)

- [ ] **修复 V14 遗留薄弱页面**（每页扩充至 ≥ 400 字）：
    - [ ] `wiki/entities/boston-dynamics.md` (当前 148 字)
    - [ ] `wiki/entities/anymal.md` (当前 178 字)
    - [ ] `wiki/methods/trajectory-optimization.md` (当前 180 字)
    - [ ] `wiki/methods/safe-rl.md` (当前 190 字)
    - [ ] `wiki/formalizations/cmdp.md` (当前 197 字)
- [ ] **建立操作 (Manipulation) 知识子链**：
    - [ ] 新增 `wiki/concepts/tactile-sensing.md` (触觉感知)。
    - [ ] 新增 `wiki/formalizations/behavior-cloning-loss.md` (模仿学习损失函数形式化)。
    - [ ] 新增 `wiki/methods/visual-servoing.md` (视觉伺服控制)。

## P2: 知识图谱软件栈生长 (Quantity)

- [ ] **新增软件栈实体页 (Software Stack Entity, +3)**:
    - [ ] `wiki/entities/drake.md` (TRI 开发的优化与仿真框架)。
    - [ ] `wiki/entities/pinocchio.md` (高性能刚体动力学库)。
    - [ ] `wiki/entities/mujoco.md` (物理引擎核心特性与机器人研究地位)。
- [ ] **新增对比页 (Comparison, +2)**:
    - [ ] `wiki/comparisons/mujoco-vs-isaac-sim.md` (物理引擎选型：接触建模、并行度、API 友好度)。
    - [ ] `wiki/comparisons/ros2-vs-lcm.md` (机器人中间件：实时性、吞吐量、运控场景适用度)。
- [ ] **新增实践指南 (Query, +2)**:
    - [ ] `wiki/queries/real-time-control-middleware-guide.md` (运控场景下如何配置实时操作系统与中间件)。
    - [ ] `wiki/queries/tactile-feedback-in-rl.md` (如何在 RL 中利用触觉反馈提升操作鲁棒性)。

## P3: UI/UX 深度优化 (Frontend)

- [ ] **图谱搜索联想 (Autocomplete)**：在图谱搜索框增加基于节点标题的实时联想列表。
- [ ] **对比页布局优化**：在 `docs/main.js` 中增加表格特定样式，优化 `comparison` 类型页面在移动端的显示效果。
- [ ] **详情页侧边栏增强**：在 `detail.html` 的 TOC 中增加“关联图谱节点”快捷入口。

---

## 验收标准 (Definition of Done)

- [ ] `make lint`: 0 errors。
- [ ] 知识图谱节点数 **≥ 130**，边数 **≥ 850**。
- [ ] `python3 scripts/eval_search_quality.py` 通过率 ≥ 90% (含新增用例)。
- [ ] 所有 `comparison` 类型页面均包含完整对比表。
- [ ] `log.md` 记录 V15 关键改动。

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
