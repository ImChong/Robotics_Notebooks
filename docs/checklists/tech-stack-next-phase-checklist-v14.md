# 技术栈项目执行清单 v14

最后更新：2026-04-20（V14 启动，基于 V13 全部交付）
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`tech-stack-next-phase-checklist-v13.md`](tech-stack-next-phase-checklist-v13.md)
方法论参考：[Karpathy LLM Wiki](../../wiki/overview/robot-learning-overview.md)

---

## P0: 核心脚本与性能修复 (Must-have)

- [x] `scripts/search_wiki.py` 中 `numpy` 改为 **Lazy Import**（解决无 numpy 环境下 BM25 搜索报错）
- [x] 修复 `search_wiki.py` 在向量索引缺失时的优雅降级逻辑
- [x] 验证 `python3 scripts/search_wiki.py PPO` 在纯 BM25 模式下正常运行

## P1: 薄弱页面深度扩充 (Quality)

- [x] `wiki/entities/humanoid-robot.md` 扩充（增加：核心特征、主流平台对比、与四足区别、核心挑战，>400字）
- [x] `wiki/concepts/hybrid-force-position-control.md` 深度扩充（增加：Task Frame 形式化、选择矩阵 S 定义、与阻抗区别，>300字）
- [x] `wiki/concepts/impedance-control.md` 深度扩充（增加：质量-弹簧-阻尼模型公式、导纳 vs 阻抗对比、调参建议，>300字）

## P2: 知识图谱横向生长 (Quantity)

- [x] **New Formalization (+3)**:
    - [x] `wiki/formalizations/cmdp.md` (Constrained MDP)
    - [x] `wiki/formalizations/zmp-lip.md` (ZMP + LIP 形式化)
    - [x] `wiki/formalizations/friction-cone.md` (摩擦锥线性化)
- [x] **New Method (+2)**:
    - [x] `wiki/methods/safe-rl.md` (Lagrangian, CPO, Safety Layer)
    - [x] `wiki/methods/trajectory-optimization.md` (Direct Collocation, iLQR)
- [x] **New Entity (+2)**:
    - [x] `wiki/entities/anymal.md` (RSL/ANYbotics 标杆平台)
    - [x] `wiki/entities/boston-dynamics.md` (Atlas/Spot/WBC 背景)
- [x] **New Query (+3)**:
    - [x] `wiki/queries/reward-shaping-guide.md` (Locomotion RL 奖励函数设计)
    - [x] `wiki/queries/locomotion-failure-modes.md` (原地踏步、抖动、起跳等失败模式诊断)
    - [x] `wiki/queries/wbc-tuning-guide.md` (WBC 权重、松弛、热启动调参)
- [x] **New Comparison (+1)**:
    - [x] `wiki/comparisons/trajectory-opt-vs-rl.md` (轨迹优化 vs RL)

## P3: 自动化 Lint 增强 (Engineering)

- [x] `scripts/lint_wiki.py` 新增 **CANONICAL_FACTS** 事实检测：
    - [x] CMDP 约束形式检测
    - [x] LIP 核心假设（高度恒定、忽略角动量）检测
    - [x] 摩擦锥线性化（QP 友好型）检测
- [x] `scripts/lint_wiki.py` 支持检测 `formalizations/` 目录下是否包含公式块 (`$$ ... $$`)

## P4: 交付与同步 (Delivery)

- [x] 运行 `make graph` 更新图谱数据
- [x] 运行 `make lint` 确保 0 错误
- [x] 运行 `make badge` 更新 README 统计
- [x] 更新 `log.md` 记录 V14 关键改动

---

## 验收标准 (Definition of Done)

- [x] `make lint`: 0 errors (忽略 sources 覆盖率中的 stub 警告)
- [x] `python3 scripts/eval_search_quality.py` 通过率 ≥ 80%
- [x] README 正文节点/边数与 `graph-stats.json` 一致 (115 nodes, 723 edges)
- [x] `log.md` 最近记录距今 ≤ 7 天
- [x] `wiki/comparisons/` 下有 ≥ 11 个对比页 (实际已达 10+，包含新加的 trajectory-opt-vs-rl.md)

---

## 状态说明

- `[ ]` 待执行
- `[~]` 进行中
- `[x]` 已完成
- `[−]` 已评估，决定跳过（附理由）
