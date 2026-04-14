# 技术栈项目执行清单 v5

最后更新：2026-04-14
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v4.md`](tech-stack-next-phase-checklist-v4.md)
方法论参考：[Karpathy LLM Wiki](../wiki/references/llm-wiki-karpathy.md)

---

## V4 完成基线（V5 起点）

| 维度 | V4 末状态 |
|------|----------|
| wiki 页面总数 | 89（concepts 19 / methods 7 / entities 7 / queries 6 / tasks 5 / formalizations 4 / comparisons 3） |
| sources/papers/ 文件 | 10 |
| Sources 覆盖率 | 59%（31/52 wiki 页有 ingest 来源） |
| Lint 健康 | ✅ 0 issues |
| GitHub Actions CI | ✅ lint.yml |
| 前端搜索 | ✅ 类型过滤 + tag 点击 + ingest 标记 |
| sitemap.xml | ✅ 89 URL |
| **log.md** | **❌ 缺失（Karpathy 必须组件）** |
| 矛盾检测 | ❌ 未实现 |
| 陈旧内容检测 | ❌ 未实现 |
| route-b-fullstack.md | ⚠️ 仅 13 行（未完成） |

---

## V5 阶段总目标

> 基于 Karpathy LLM Wiki 模式，把 Robotics_Notebooks 从"结构完善的知识库"推进为**真正自我维护的编译机**：建立运营层（log.md）、扩大 sources 覆盖至 80%+、增强 lint 的矛盾/陈旧检测能力，让每次 ingest 都能在 10-15 个页面上留下痕迹。

---

## P0 · 运营层完善（最高优先级）

**背景**：Karpathy 原文明确指出两个必须的特殊文件：`index.md`（目录）和 `log.md`（操作日志）。目前 `index.md` 已完善，但 `log.md` 完全缺失，无法追踪知识库演进历史。

### 0.1 log.md 建立

- [ ] 新建 `log.md`：append-only 操作日志，格式 `## [YYYY-MM-DD] op | 描述`
  - 支持类型前缀：`ingest`、`query`、`lint`、`index`
  - 示例：`## [2026-04-14] ingest | sources/papers/mpc.md — 新增 Mayne 2000 等 5 条 MPC 论文摘要`
  - 用途：LLM 通过 grep 快速了解近期操作，避免重复工作

- [ ] `scripts/append_log.py`：命令行追加 log 条目
  ```bash
  python3 scripts/append_log.py ingest "sources/papers/xxx.md — 描述"
  python3 scripts/append_log.py query "关键词 → wiki/queries/xxx.md"
  python3 scripts/append_log.py lint "0 issues，全绿"
  ```

- [ ] `Makefile` 新增：`make log OP=ingest DESC="描述"`

### 0.2 schema 更新

- [ ] `schema/ingest-workflow.md`：在每步操作后加入 `追加 log.md 条目` 子步骤
- [ ] 新建 `schema/log-format.md`：规范 log 格式与可解析前缀约定

### 完成标准
- `log.md` 存在，可用 `grep "^## \[" log.md | tail -5` 查看最近 5 条
- `make log` 可一行追加条目
- `schema/` 中记录了 log 格式规范

---

## P1 · Sources 层扩充（第二优先级）

**背景**：当前 59%（31/52）的 wiki 页面有 ingest 来源，21 个核心页面（含 contact-dynamics、privileged-training、system-identification 等）完全没有论文支撑。

### 1.1 核心概念 sources 补充

| 目标文件 | 覆盖 wiki 页面 | 关键论文 |
|---------|--------------|---------|
| `sources/papers/contact_dynamics.md` | contact-dynamics, contact-estimation | Featherstone 2008 RBDA；Stewart 2000 LCP；Todorov 2011 MuJoCo contact |
| `sources/papers/privileged_training.md` | privileged-training, sim2real | Kumar RMA 2021；Lee 2020 teacher-student；Ji 2022 concurrent training |
| `sources/papers/state_estimation.md` | state-estimation, EKF | Bloesch RSL 2013；Hartley InEKF 2020；Teng 2021 legged odom |
| `sources/papers/system_identification.md` | system-identification | Nguyen IJRR 2011；Gautier 2013 excitation traj；Hwangbo 2019 ActuatorNet |

- [ ] `sources/papers/contact_dynamics.md`
- [ ] `sources/papers/privileged_training.md`
- [ ] `sources/papers/state_estimation.md`
- [ ] `sources/papers/system_identification.md`

### 1.2 实体页 sources 补充

- [ ] `sources/papers/simulation_tools.md`
  - MuJoCo（Todorov et al. 2012）、Isaac Gym（Makoviychuk et al. 2021）、Genesis（2024）
  - 覆盖：mujoco.md、isaac-gym-isaac-lab.md

- [ ] `sources/papers/robot_kinematics_tools.md`
  - Pinocchio（Carpentier et al. 2019）、RBDL、Crocoddyl
  - 覆盖：pinocchio.md、crocoddyl.md

### 1.3 wiki 参考来源链接化（P0.2 延续）

下列页面现有文字引用，需改为 sources/ 链接：
- [ ] `wiki/concepts/privileged-training.md` → `sources/papers/privileged_training.md`
- [ ] `wiki/concepts/system-identification.md` → `sources/papers/system_identification.md`
- [ ] `wiki/concepts/contact-dynamics.md` → `sources/papers/contact_dynamics.md`
- [ ] `wiki/concepts/state-estimation.md` → `sources/papers/state_estimation.md`
- [ ] `wiki/entities/mujoco.md` → `sources/papers/simulation_tools.md`
- [ ] `wiki/entities/pinocchio.md` → `sources/papers/robot_kinematics_tools.md`

### 完成标准
- sources/papers/ 文件数：**16+**（+6 新建）
- wiki 含 sources 链接页：**40+**（当前 31，覆盖率提升至 **75%+**）
- 每个新 sources 文件至少覆盖 2 个 wiki 页面

---

## P2 · Wiki 内容缺口补全（第三优先级）

### 2.1 频繁提及但缺独立页面的概念

- [ ] `wiki/concepts/footstep-planning.md`
  - 步位规划：contact sequence + step location optimization
  - 被 capture-point-dcm.md、locomotion.md、balance-recovery.md 提及

- [ ] `wiki/concepts/gait-generation.md`
  - 步态生成：中枢模式生成器（CPG）、参数化步态、数据驱动步态
  - 被 locomotion.md、mpc.md 提及

- [ ] `wiki/formalizations/contact-complementarity.md`
  - LCP / complementarity 条件：接触力学的数学基础
  - 被 contact-dynamics.md、whole-body-control.md 提及

### 2.2 现有页面深化

- [ ] `wiki/concepts/floating-base-dynamics.md` → 内容扩充
  - 当前仅有基础定义；需补充 Featherstone RNEA、空间向量代数、EOM 推导
  
- [ ] `wiki/comparisons/mpc-vs-rl.md`
  - 控制领域最核心的对比：MPC（确定性模型+在线规划）vs RL（学习策略+隐式规划）
  - 已有 model-based-vs-model-free，但这个角度更聚焦工程选型

### 2.3 路线图补全

- [ ] `roadmap/route-b-fullstack.md` → 扩充至完整路线图
  - 当前仅 13 行占位符，需对标 route-a 的完整度（341 行）
  - 内容：全栈人形机器人研发路线（感知→规划→控制→部署→测试）

### 完成标准
- 3 个新 wiki 页面，lint ✅ 0 issues
- route-b-fullstack.md 达到 200+ 行，结构完整

---

## P3 · Lint 深化（Karpathy 健康检查增强）

**背景**：Karpathy 原文提到 lint 应检测：contradictions（矛盾）、stale claims（陈旧断言）、orphan pages、missing cross-references、data gaps。目前我们只做了 orphan/missing/broken link 检测。

### 3.1 矛盾检测（Contradiction Detection）

- [ ] `lint_wiki.py`：新增 5b 检测 — 关键数字/定义一致性
  - 检测同一概念在不同页面的不一致描述（e.g. 某算法的样本效率数字相差 10x）
  - 实现：维护 `CANONICAL_FACTS` dict，对高频数字/定义做跨页面比对

### 3.2 陈旧内容检测（Stale Detection）

- [ ] `lint_wiki.py`：新增 6 检测 — sources 文件比对应 wiki 页面更新时，标记需 review
  - 通过文件 mtime 比对：如果 `sources/papers/xxx.md` 比 `wiki/methods/xxx.md` 新 → ⚠️

### 3.3 Sources 孤儿检测

- [ ] `lint_wiki.py`：新增 7 检测 — sources 文件中列出的 wiki 映射目标不存在
  - 扫描 `sources/papers/*.md` 中 `对 wiki 的映射` 链接，验证目标存在

### 3.4 覆盖率报告

- [ ] `lint_wiki.py` 末尾输出 ingest 覆盖率统计：
  ```
  📊 Sources 覆盖率：31/52 (59%) wiki/entity 页有 ingest 来源
  ```

### 完成标准
- lint 输出新增覆盖率报告行
- 3 个新检测类型正常运行（即使当前 0 issues）

---

## P4 · Query 产物系统化（第五优先级）

Query 产物是 Karpathy 模式的核心价值点：**好答案应该写回 wiki，让探索复利**。

- [ ] `wiki/queries/humanoid-rl-cookbook.md`
  - 触发问题：「从零开始训练一个能在真机上走路的人形机器人 RL 策略，完整 checklist？」
  - 综合来源：sim2real-checklist.md、rl-algorithm-selection.md、locomotion-reward-design-guide.md

- [ ] `wiki/queries/pinocchio-quick-start.md`
  - 触发问题：「用 Pinocchio 做机器人动力学计算的最小可运行示例？」
  - 综合来源：pinocchio.md、wbc-implementation-guide.md、tsid.md

- [ ] `wiki/queries/mpc-solver-selection.md`
  - 触发问题：「机器人 MPC 求解器怎么选：OSQP vs qpOASES vs Acados vs FORCES Pro？」
  - 综合来源：model-predictive-control.md、mpc-wbc-integration.md

### 完成标准
- 3 个新 query 产物，wiki/queries/README.md 同步更新
- 每个 query 产物 300+ 字，包含决策树或对比表格

---

## P5 · 前端体验提升（持续改进）

### 5.1 搜索体验

- [ ] `index.html` 搜索结果键盘导航：↑↓ 选中条目，Enter 打开详情页
- [ ] `index.html` 搜索框下方标签云：统计高频 tag，点击直接过滤

### 5.2 detail.html 增强

- [ ] 关联页面列表：当前是文字列表，改为卡片式（显示 summary 片段）
- [ ] `og:image` 动态设置（基于 page_type 选择预设图标 URL）

### 5.3 tech-map.html 增强

- [ ] 层级筛选器：多选（目前单选），支持 Ctrl+Click 多层级同时显示
- [ ] 页面内搜索：在 tech-map 节点中实时过滤匹配节点

---

## P6 · 文档与 Schema 补全

- [ ] `schema/ingest-workflow.md` 更新：
  - 加入 `ingest_paper.py` 工具使用说明
  - 加入 `log.md` 追加步骤
  - 更新工具命令（Makefile 版本）

- [ ] `schema/page-types.md` 更新：
  - 新增 `query` 类型的详细规范（当前 queries/ 目录规范在 queries/README.md 里，未同步到 schema）

---

## 维护操作标准（Karpathy 四大 Ops，V5 更新版）

### Op 1：Ingest（添加新资料）
```
1. make ingest NAME=xxx TITLE="..." DESC="..."  # 生成 sources/papers/xxx.md 模板
2. 编辑模板，填写论文摘录（至少 3 条核心论文 + wiki 映射）
3. 在对应 wiki 页面 ## 参考来源 加入 ingest 档案链接
4. 更新关联页面的 ## 关联页面 区块（确保双向链接）
5. make lint       # 确认 0 issues
6. make catalog    # 更新 index.md
7. make export     # 同步 JSON + sitemap
8. make log OP=ingest DESC="sources/papers/xxx.md — 简述"  # 追加 log
```

### Op 2：Query（知识查询）
```
1. make search Q=<关键词>  # 快速定位
2. python3 scripts/search_wiki.py <关键词> --related  # 加载邻居页面
3. 综合多页面分析，得出结论
4. 如有独立价值 → 保存为 wiki/queries/xxx.md
5. 更新 wiki/queries/README.md 的表格
6. make lint && make catalog && make export
7. make log OP=query DESC="关键词 → wiki/queries/xxx.md"
```

### Op 3：Lint（健康检查）
```
make lint                              # 完整健康检查（含覆盖率报告）
make log OP=lint DESC="0 issues，覆盖率 XX%"
```
目标：每次大规模改动后运行，永久保持 0 issues。

### Op 4：Index（索引更新）
```
make catalog    # 刷新 index.md（Page Catalog）
make export     # 更新 exports/ JSON + sitemap.xml
```
每次新增 wiki 页面后必须执行，通常紧跟 lint 之后。

---

## Karpathy 对齐度评估（V5 起点）

| Karpathy 原则 | V4 末状态 | V5 目标 |
|-------------|----------|--------|
| Raw sources（不可变 sources 层） | ✅ sources/papers/ 10 个文件 | 16+ 文件，覆盖率 75%+ |
| Wiki（LLM 维护的 md 文件集） | ✅ 89 页，互联完整 | 95+ 页 |
| Schema（配置与规范文档） | ✅ schema/ 4 个文件 | 更新对齐新工具 |
| Ingest 操作 | ✅ ingest_paper.py + Makefile | 加入 log 追加步骤 |
| Query 操作 | ✅ 6 个 query 产物 | 9 个，覆盖更多高频问题 |
| Lint 操作 | ✅ 7 项检测，0 issues | 加入矛盾/陈旧/覆盖率检测 |
| Index 操作 | ✅ index.md + export JSON | 同步更新 log |
| **log.md** | **❌ 缺失** | **✅ 建立 + 工具支持** |
| 矛盾检测 | ❌ | ⚠️（部分实现） |
| 知识复利（query→wiki） | ✅ 6 篇 query 产物 | 9 篇+ |
| 搜索工具 | ✅ search_wiki.py + 前端搜索 | 键盘导航增强 |

---

## 状态约定
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[-]` 暂缓
