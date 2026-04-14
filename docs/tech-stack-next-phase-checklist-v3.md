# 技术栈项目执行清单 v3

最后更新：2026-04-14
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v2.md`](tech-stack-next-phase-checklist-v2.md)
方法论参考：[Karpathy LLM Wiki](wiki/references/llm-wiki-karpathy.md)

---

## 当前项目状态（V3 起点）

### 已完成的核心基础设施

**知识组织层（wiki/）**

| 类别 | 已有页面 |
|------|---------|
| 概念页 | lip-zmp, centroidal-dynamics, tsid, whole-body-control, state-estimation, system-identification, floating-base-dynamics, contact-dynamics, capture-point-dcm, optimal-control, sim2real, domain-randomization, mpc-wbc-integration, **reward-design**, **hqp** |
| 方法页 | reinforcement-learning, model-predictive-control, imitation-learning, trajectory-optimization, policy-optimization, diffusion-policy |
| 任务页 | locomotion, manipulation, loco-manipulation, ultra-survey |
| 形式化 | mdp, bellman-equation, **lqr**, **ekf** |
| 对比页 | wbc-vs-rl |
| 实体页 | mujoco, isaac-gym-isaac-lab, legged-gym, pinocchio, crocoddyl, unitree |
| 总览页 | robot-learning-overview |

**工具层（scripts/）**

| 工具 | 功能 |
|------|------|
| `lint_wiki.py` | 健康检查：孤儿页 / 断链 / 缺参考来源 / 缺关联页面 / 概念缺口建议 |
| `search_wiki.py` | CLI 搜索：关键词 AND / 类型 / 标签 / 上下文行数过滤，ANSI 高亮 |
| `generate_page_catalog.py` | 自动生成 index.md Page Catalog |
| `export_minimal.py` | 导出 JSON 供前端消费 |

**Karpathy 对齐层**

| 原则 | 状态 |
|------|------|
| 三层架构（sources→wiki→schema） | ✅ 已实现 |
| Ingest / Query / Lint / Index 操作规范 | ✅ 已文档化（schema/ingest-workflow.md） |
| wiki 页面互联（cross-references） | ✅ 每页有"关联页面"区块，lint 强制检测 |
| 追加日志（append-only log.md） | ✅ |
| Sources → Wiki 可溯源性（参考来源区块） | ✅ 所有 wiki 页面已补全 |
| 自动化 Lint | ✅ lint_wiki.py，0 孤儿页 / 0 断链 |
| 查询产物系统化 | ✅ wiki/queries/ 目录已建立 |
| YAML Frontmatter + Dataview 支持 | ✅ 所有核心页面已补全 |
| index.md Page Catalog 自动生成 | ✅ generate_page_catalog.py |
| "提及但缺页"概念主动检测 | ✅ lint 💡 建议机制 |

**lint 健康状态**：`✅ 0 孤儿页 / 0 断链 / 0 缺参考来源 / 0 缺关联页面 / 0 概念缺口`

---

## V3 阶段总目标

> 把 Robotics_Notebooks 从"结构完整的知识库骨架"推进为**真正持续活跃的知识编译机**：sources 层有真实内容，wiki 层持续被新资料滋养，Query 产物不断积累为新页面。

---

## P0 · Sources 层激活（最高优先级）

**背景**：Karpathy gist 的核心原则是 "compilation beats retrieval"——wiki 的价值来自对原始资料的编译。目前 sources/ 只是占位列表，不是真正被消费的资料层。

### 任务

- [~] 将 `sources/papers/` 中的论文列表升级为实际 ingest 笔记（每篇至少：标题、核心贡献、与 wiki 的映射关系）
  - [x] `locomotion_rl.md`：补充 AMP / ASE / PPO-for-locomotion 核心摘要
  - [x] `whole_body_control.md`：补充 TSID / HQP / Crocoddyl 核心摘要
  - [x] `imitation_learning.md`：补充 DAgger / ACT / Diffusion Policy 核心摘要
  - [x] `sim2real.md`：补充域随机化 / RMA / InEKF 核心摘要
- [x] 在 sources/README.md 中为每条资料增加状态标记
  - `[x]` 已提炼进 wiki
  - `[ ]` 待提炼
  - `[-]` 暂不提炼
- [~] 将 wiki 页面的 `## 参考来源` 中的论文引用，逐步从纯文本改为链接到 `sources/papers/xxx.md` 具体条目（已有源文件的优先补）

### 完成标准
- [x] sources/papers/ 至少 3 个文件有真实 ingest 内容（不只是链接列表）
- [x] 至少 5 个 wiki 页面的参考来源链接到 sources/ 实体文件

---

## P1 · Wiki 内容深挖（第二优先级）

### 5.1 缺失的高价值概念页

- [x] `wiki/concepts/privileged-training.md`
  - 特权信息训练（teacher-student）、asymmetric actor-critic
  - 被 sim2real.md、reinforcement-learning.md、loco-manipulation.md 频繁提及
- [x] `wiki/methods/model-based-rl.md`
  - Dreamer / MBPO / PETS 等；与 model-free 对比
  - 被 reinforcement-learning.md 提及但无专页
- [x] `wiki/tasks/balance-recovery.md`
  - 扰动恢复、推一下站回来；capture point 的实际应用场景
  - 被 locomotion.md、capture-point-dcm.md 提及

### 5.2 缺失的对比页

- [x] `wiki/comparisons/rl-vs-il.md`
  - RL vs 模仿学习：数据效率 / 泛化 / 行为质量 / 实际应用场景对比
  - 被多个方法页提及，目前只有 wbc-vs-rl.md

### 5.3 增强已有页面深度

- [ ] `wiki/concepts/sim2real.md`：补充 RMA（Rapid Motor Adaptation）具体步骤
- [ ] `wiki/methods/reinforcement-learning.md`：补充 model-based RL 与 model-free 对比表格
- [ ] `wiki/comparisons/wbc-vs-rl.md`：状态从 `draft` 升级到 `complete`，补充更多融合架构实例

---

## P2 · Query 产物积累（第三优先级）

**背景**：Karpathy gist 明确说 "good answers can be filed back into the wiki"。wiki/queries/ 目录已建立，但还没有实际产物。

### 任务

- [ ] 每次复杂 Query 操作后，将答案整理为 wiki/queries/xxx.md
  - 建议格式：`## 触发问题` + 正文 + `## 参考页面`
- [ ] 候选 Query 产物（下次回答这些问题时保存结果）：
  - `wiki/queries/rl-algorithm-selection.md`：在足式机器人里选 PPO / SAC / TD3 的决策指南
  - `wiki/queries/sim2real-checklist.md`：从仿真到真机部署的工程 checklist
  - `wiki/queries/control-architecture-comparison.md`：各主流人形控制架构（MPC-WBC / End2End RL / Hierarchical IL）的综合对比

---

## P3 · 工具链与自动化（持续改进）

### 3.1 Makefile 一键操作

- [x] 新建根目录 `Makefile`，提供常用操作的快捷方式：
  ```makefile
  make lint        # python3 scripts/lint_wiki.py
  make catalog     # python3 scripts/generate_page_catalog.py
  make export      # python3 scripts/export_minimal.py
  make search Q=   # python3 scripts/search_wiki.py $(Q)
  ```

### 3.2 搜索工具增强

- [ ] 在 `scripts/search_wiki.py` 中增加 `--related` 选项：输出匹配页面的关联页面（用于 Query 时快速找邻居）
- [ ] 考虑：是否引入 [qmd](https://github.com/tobi/qmd) 做 BM25/向量混合搜索（Karpathy 推荐工具）

### 3.3 Lint 规则扩展

- [ ] 新增检测："推荐继续阅读" 中含有失效外链（HTTP 404）
- [ ] 新增检测：wiki 页面引用了 sources/ 文件但该文件不存在

---

## P4 · 前端与部署（延续 V2 方向）

延续 V2 已完成的 detail.html / tech-map.html / module.html / roadmap.html 体系，进一步完善：

- [ ] 给 `detail.html` 增加"相关页面"侧边栏（从 `related` 字段渲染）
- [ ] 给 `tech-map.html` 增加从某个节点出发的"最短路径"视图
- [ ] 为 `index.html` 添加快速搜索框（消费 `index-v1.json` 做客户端全文搜索）

---

## 维护操作标准（Karpathy 四大 Ops）

### Op 1：Ingest（添加新资料）
```
1. 进入 sources/，写 ingest 笔记（来源、摘要、为什么值得保留）
2. 判断是否沉淀到 wiki/（能解释概念 / 补全方法 / 影响路线判断）
3. 在 wiki 页面写/更新 ## 参考来源，链接到 sources/ 文件
4. 更新相关页面的关联区块
5. 运行 python3 scripts/generate_page_catalog.py 更新 index.md
6. 运行 python3 scripts/export_minimal.py 同步导出
7. 追加 log.md 条目
```

### Op 2：Query（知识查询）
```
1. python3 scripts/search_wiki.py <关键词> 快速定位相关页面
2. 读取相关页面，综合分析
3. 如果答案有独立价值，保存为 wiki/queries/xxx.md
4. 更新 wiki/queries/README.md 的查询产物表格
```

### Op 3：Lint（健康检查）
```
python3 scripts/lint_wiki.py            # 快速检查
python3 scripts/lint_wiki.py --write-log # 结果写入 log.md
```

目标：每次大规模改动后运行，保持 0 孤儿页 / 0 断链 / 0 概念缺口

### Op 4：Index（索引更新）
```
python3 scripts/generate_page_catalog.py  # 生成 Page Catalog 片段
python3 scripts/export_minimal.py          # 更新 exports/ JSON
```

每次新增 wiki 页面后必须执行。

---

## 本次推进记录（V3 起点，2026-04-14）

### P0 Sources 激活推进（2026-04-14）
- [x] `sources/papers/locomotion_rl.md` 增加 AMP/ASE/Heess 等 ingest 摘要与 wiki 映射
- [x] `sources/papers/whole_body_control.md` 增加 TSID/HQP/Crocoddyl ingest 摘要与 wiki 映射
- [x] `sources/papers/imitation_learning.md` 增加 DAgger/ACT/Diffusion ingest 摘要与 wiki 映射
- [x] `sources/papers/sim2real.md` 增加 DR/RMA/InEKF ingest 摘要与 wiki 映射
- [x] `sources/README.md` 增加 `[x]/[ ]/[-]` 提炼状态标记
- [x] 6 个 wiki 页面参考来源补充 `sources/papers/*.md` 实体链接（RL/IL/Sim2Real/WBC/TSID/HQP）

### Karpathy 对齐专项（2026-04-14）
- [x] 为 16+ 核心 wiki 页面补全 `## 参考来源` 区块（含 entities、concepts、methods、formalizations）
- [x] 新增 `scripts/lint_wiki.py`：0 孤儿页、0 断链、💡 概念缺口建议机制
- [x] 新增 `scripts/search_wiki.py`：关键词 / 类型 / 标签过滤，ANSI 高亮输出
- [x] 修复 `scripts/generate_page_catalog.py` frontmatter 剥离 bug
- [x] 为所有核心 wiki 页面补全 YAML frontmatter（type / tags / status）
- [x] 新增 P2 页面：`policy-optimization.md`、`diffusion-policy.md`、`loco-manipulation.md`
- [x] 新建 `wiki/queries/README.md`，建立 query 产物追踪机制
- [x] 更新 `schema/page-types.md`：参考来源列为必填，YAML frontmatter 规范
- [x] 更新 `schema/ingest-workflow.md`：步骤 6 明确 generate_page_catalog.py 调用
- [x] 修复错误链接：`mdp.md` 中 Reward Design 指向 domain-randomization（已修正）
- [x] 新增页面：`wiki/concepts/reward-design.md`（奖励函数设计）
- [x] 新增页面：`wiki/formalizations/lqr.md`（LQR / iLQR）
- [x] 新增页面：`wiki/formalizations/ekf.md`（EKF / InEKF）
- [x] 新增页面：`wiki/concepts/hqp.md`（分层 QP）
- [x] 在 `index.md` 添加 Obsidian Dataview 查询示例块

---

## 状态约定
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[-]` 暂缓
