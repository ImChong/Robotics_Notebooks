# 技术栈项目执行清单 v4

最后更新：2026-04-14
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v3.md`](tech-stack-next-phase-checklist-v3.md)
方法论参考：[Karpathy LLM Wiki](../wiki/references/llm-wiki-karpathy.md)

---

## V3 阶段完成状态（V4 起点）

### 已完成的全部工作

**知识组织层（wiki/）**

| 类别 | 页面数 | 新增（V3 阶段） |
|------|--------|--------------|
| 概念页 | 16 | privileged-training, reward-design, hqp |
| 方法页 | 8 | model-based-rl, policy-optimization, diffusion-policy |
| 任务页 | 6 | loco-manipulation, balance-recovery, ultra-survey |
| 形式化 | 4 | lqr, ekf |
| 对比页 | 2 | rl-vs-il（wbc-vs-rl 升级至 complete） |
| 实体页 | 6 | — |
| 总览页 | 1 | — |
| **Query 产物** | **4** | **rl-algorithm-selection, sim2real-checklist, control-architecture-comparison** |

**工具层（scripts/）**

| 工具 | 状态 | V3 新增功能 |
|------|------|-----------|
| `lint_wiki.py` | ✅ | `broken_source_refs` 检测（sources/ 引用失效） |
| `search_wiki.py` | ✅ | `--related` 选项（输出关联页面邻居） |
| `generate_page_catalog.py` | ✅ | — |
| `export_minimal.py` | ✅ | 新增 formalizations/queries 目录扫描，81 页 |
| `Makefile` | ✅ | 新建，`make lint/catalog/export/search` |

**Karpathy 对齐层**

| 原则 | V3 末状态 |
|------|----------|
| 三层架构（sources→wiki→schema） | ✅ |
| Ingest / Query / Lint / Index 操作规范 | ✅ |
| wiki 页面互联（cross-references） | ✅ lint 强制 |
| Sources → Wiki 可溯源性 | ⚠️ 4 个 sources 文件有 ingest 内容，仍有大量纯文本引用 |
| 自动化 Lint | ✅ 0 issues |
| Query 产物积累 | ✅ 4 篇，目录机制完善 |
| CI 集成（自动 lint on push） | ❌ 尚未接入 GitHub Actions |
| Sources 覆盖率 | ❌ 核心论文仍大量缺失 ingest 条目 |

**lint 健康状态**：`✅ 0 孤儿页 / 0 断链 / 0 缺参考来源 / 0 缺关联页面 / 0 概念缺口 / 0 sources 引用失效`

---

## V4 阶段总目标

> 把 Robotics_Notebooks 从"骨架与机制完善的知识库"推进为**真正以论文为原材料的知识编译机**：每个 wiki 页面的核心观点都能追溯到具体的 sources/papers/ 条目，sources 层成为不可变的事实基础。

---

## P0 · Sources 层深度激活（最高优先级）

**背景**：Karpathy gist 最核心的一句话是 "compilation beats retrieval"。目前 wiki 页面大多是独立创作，而不是从论文中编译而来。sources/ 层覆盖率不足严重削弱了知识库的可信度和溯源能力。

### 任务 0.1：核心论文 ingest 条目扩充

为以下高频引用论文建立 `sources/papers/` 对应条目（每条至少：标题/年份/核心贡献/与 wiki 映射）：

**控制类**
- [x] `sources/papers/whole_body_control.md`：补充 Koolen et al. N-step Capture Point（平衡恢复理论）
- [x] `sources/papers/optimal_control.md`（新建）：Bellman 1957、Pontryagin、Bryson & Ho — 最优控制奠基
- [x] `sources/papers/mpc.md`（新建）：Mayne et al. 2000（MPC 综述）、Di Carlo et al. 2018（MIT Cheetah 3 MPC）

**RL 类**
- [x] `sources/papers/locomotion_rl.md`：补充 Lee et al. 2020（Science Robotics 四足）、ETH ANYmal 系列
- [x] `sources/papers/policy_optimization.md`（新建）：PPO（Schulman 2017）、SAC（Haarnoja 2018）、TD3（Fujimoto 2018）
- [x] `sources/papers/model_based_rl.md`（新建）：DreamerV3、MBPO、PETS、TD-MPC2

**IL 类**
- [x] `sources/papers/imitation_learning.md`：补充 ACT（Zhao et al. 2023）、Chi et al. Diffusion Policy

**Sim2Real 类**
- [x] `sources/papers/sim2real.md`：补充 Kumar et al. RMA 2021 详细摘要

### 任务 0.2：wiki 参考来源链接化

将已有 sources/papers/ 文件对应的 wiki 页面参考来源，从纯文本引用改为 sources/ 链接：

- [x] `wiki/methods/model-based-rl.md` → 链接到 `sources/papers/model_based_rl.md`
- [x] `wiki/methods/policy-optimization.md` → 链接到 `sources/papers/policy_optimization.md`
- [x] `wiki/concepts/optimal-control.md` → 链接到 `sources/papers/optimal_control.md`
- [x] `wiki/concepts/capture-point-dcm.md` → 链接到 `sources/papers/whole_body_control.md`
- [x] `wiki/tasks/balance-recovery.md` → 链接到 `sources/papers/whole_body_control.md`
- [x] `wiki/methods/model-predictive-control.md` → 链接到 `sources/papers/mpc.md`

### 完成标准
- sources/papers/ 文件数达到 **10+**（目前 4 个）
- 至少 **15 个** wiki 页面的参考来源包含 sources/ 链接（目前约 6 个）
- lint 新增检测通过（无 `broken_source_refs`）

---

## P1 · Wiki 内容缺口补全（第二优先级）

### 1.1 缺失的高价值概念页

- [x] `wiki/concepts/curriculum-learning.md`
  - 课程学习：从简单到复杂的训练策略（地形课程、速度课程等）
  - 被 locomotion.md、sim2real.md、domain-randomization.md 频繁提及
  - 与 domain-randomization 密切相关但定位不同

- [x] `wiki/concepts/contact-estimation.md`
  - 足式机器人接触状态估计：接触检测、接触力估计、接触时序预测
  - 被 state-estimation.md、locomotion.md、contact-dynamics.md 提及
  - 与 EKF 的 contact-aided 滤波器直接相关

- [x] `wiki/concepts/motion-retargeting.md`
  - 人类动作到机器人动作的重定向：IK 重定向、物理约束重定向、风格保持
  - 被 imitation-learning.md、loco-manipulation.md 提及
  - 是 IL 数据获取链路的关键一步

### 1.2 缺失的对比页

- [x] `wiki/comparisons/model-based-vs-model-free.md`
  - 基于模型 vs 无模型：在机器人任务中的实际差异、选型指南
  - 补充 reinforcement-learning.md 和 model-based-rl.md 中对比表格的深度

### 1.3 实体页补全

- [x] `wiki/entities/drake.md`
  - Drake（Tedrake lab）：机器人轨迹优化和控制工具链
  - 被 trajectory-optimization.md、optimal-control.md 提及

---

## P2 · Query 产物持续积累（第三优先级）

Query 产物遵循 schema：`## 触发问题` + 正文 + `## 关联页面` + `## 参考来源`

- [x] `wiki/queries/humanoid-hardware-selection.md`
  - 触发问题：「人形机器人研究平台怎么选？Unitree G1/H1 vs Fourier GR1 vs 自研？」
  - 综合来源：unitree.md、locomotion.md、sim2real.md

- [x] `wiki/queries/wbc-implementation-guide.md`
  - 触发问题：「从 URDF 到能跑起来的 TSID/WBC，要走哪些步骤？」
  - 综合来源：tsid.md、hqp.md、pinocchio.md、crocoddyl.md

- [x] `wiki/queries/locomotion-reward-design-guide.md`
  - 触发问题：「足式 locomotion 的 reward 怎么设计，有哪些关键项？」
  - 综合来源：reward-design.md、locomotion.md、rl-algorithm-selection.md

---

## P3 · 自动化与 CI（第四优先级）

### 3.1 GitHub Actions：Lint on Push

- [x] 新建 `.github/workflows/lint.yml`：
  ```yaml
  name: Wiki Lint
  on: [push, pull_request]
  jobs:
    lint:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version: '3.11' }
        - run: python3 scripts/lint_wiki.py
  ```
  目标：每次 push 自动运行 lint，PR 中可见健康状态

### 3.2 Ingest 辅助工具

- [x] `scripts/ingest_paper.py`：给定论文标题 + 年份 + 作者，自动生成 `sources/papers/` 模板文件，包含：
  - YAML frontmatter（topic/year/status）
  - 模板区块：核心贡献、与 wiki 的映射关系、ingest 状态
- [x] `Makefile` 新增：`make ingest TITLE="xxx" YEAR=20xx`

### 3.3 Export 增强

- [x] `export_minimal.py` 支持生成 `sitemap.xml`，便于 GitHub Pages SEO
- [x] `detail.html` 中从 `summary` 字段动态填充 `<meta description>`

---

## P4 · 前端体验深化（持续改进）

- [x] `index.html` 搜索框增强：支持按类型过滤（concept/method/task 下拉选择）
- [x] `detail.html`："在所有页面中搜索此关键词"快捷链接（点击标签跳转搜索）
- [x] `tech-map.html`：节点详情弹窗中加入 sources/ ingest 状态标记

---

## 维护操作标准（Karpathy 四大 Ops）

### Op 1：Ingest（添加新资料）
```
1. 进入 sources/，写 ingest 笔记（来源、摘要、与 wiki 映射关系）
   可使用：python3 scripts/ingest_paper.py --title "xxx" --year 20xx
2. 判断是否沉淀到 wiki/（能解释概念 / 补全方法 / 影响路线判断）
3. 在 wiki 页面写/更新 ## 参考来源，链接到 sources/ 文件
4. 更新相关页面的关联区块
5. make catalog   → 更新 index.md
6. make export    → 同步导出 JSON
7. 追加 log.md 条目
```

### Op 2：Query（知识查询）
```
1. python3 scripts/search_wiki.py <关键词> --related  快速定位 + 找邻居
2. 读取相关页面，综合分析
3. 如果答案有独立价值 → 保存为 wiki/queries/xxx.md
4. 更新 wiki/queries/README.md 的查询产物表格
5. make catalog && make export
```

### Op 3：Lint（健康检查）
```
make lint                              # 快速检查
python3 scripts/lint_wiki.py --write-log  # 结果写入 log.md
```
目标：每次大规模改动后运行，保持全零状态

### Op 4：Index（索引更新）
```
make catalog   # 生成 Page Catalog → index.md
make export    # 更新 exports/ JSON（81 页）
```
每次新增 wiki 页面后必须执行。

---

## 本次推进记录（V4 起点，2026-04-14）

### V3 完成汇总（2026-04-14）

**P1.3 页面深化：**
- [x] `wiki/concepts/sim2real.md`：补充 RMA 两阶段流程详细说明（Adaptation Module 机制）
- [x] `wiki/methods/reinforcement-learning.md`：新增 Model-Free vs Model-Based 对比表格
- [x] `wiki/comparisons/wbc-vs-rl.md`：`draft` → `complete`，补全 5 种融合架构（RL HLC+WBC LLC、AMP/ASE LLC+HLC、WBC 生成演示+IL、MPC-WBC、VLA）

**P2 Query 产物（3 篇）：**
- [x] `wiki/queries/rl-algorithm-selection.md`：PPO / SAC / TD3 在足式机器人中的选型决策指南
- [x] `wiki/queries/sim2real-checklist.md`：从仿真到真机部署的 5 阶段工程 checklist（含域随机化配置、系统辨识、安全部署）
- [x] `wiki/queries/control-architecture-comparison.md`：6 种主流人形控制架构综合对比（ZMP/MPC-WBC/端到端RL/AMP/层次IL/VLA）

**P3 工具增强：**
- [x] `scripts/search_wiki.py`：新增 `--related` 选项，输出匹配页面的关联页面邻居
- [x] `scripts/lint_wiki.py`：新增 `broken_source_refs` 检测（wiki 引用了不存在的 sources/ 文件）
- [x] `scripts/export_minimal.py`：新增 `formalizations/`、`queries/` 目录扫描，导出从 74 → 81 页；新增 `query` / `formalization` 类型映射

**P4 前端：**
- [x] `docs/index.html`：新增客户端全文搜索框（消费 `index-v1.json`，多词 AND，200ms 防抖，最多 12 条结果）
- [x] `docs/style.css`：搜索框样式（聚焦时高亮边框，主题色适配）

---

## 状态约定
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[-]` 暂缓

---

## V4 完成汇总（2026-04-14）

### P0 Sources 层深度激活
- [x] 新建 4 个 `sources/papers/`：`optimal_control` / `mpc` / `policy_optimization` / `model_based_rl`（各含 5 条核心论文摘录）
- [x] 扩充 `whole_body_control.md`（Koolen N-step Capture Point）+ `locomotion_rl.md`（Lee 2020 / ANYmal 系列）
- [x] 6 个 wiki 页面参考来源新增 `ingest 档案` 双向链接（model-based-rl / policy-optimization / optimal-control / capture-point-dcm / balance-recovery / model-predictive-control）
- [x] sources/papers/ 文件数：10，wiki 含 sources 链接页：31（`has_ingest` 字段追踪）

### P1 Wiki 内容缺口
- [x] `curriculum-learning.md`（手动/自适应/地形课程，与 RL+sim2real 关系）
- [x] `contact-estimation.md`（力矩估计/F/T传感器/分类器，WBC 依赖分析）
- [x] `motion-retargeting.md`（MoCap→机器人骨架，AMP/ASE 连接）
- [x] `comparisons/model-based-vs-model-free.md`（8 维对比表 + 机器人选型决策树）
- [x] `entities/drake.md`（MultibodyPlant / MathProg / 轨迹优化工具链）
- [x] exports 89 页（+8 页），lint ✅ 0 issues

### P2 Query 产物
- [x] `humanoid-hardware-selection.md`（G1/H1/GR1 平台对比与决策树）
- [x] `wbc-implementation-guide.md`（Pinocchio + OSQP 从零 WBC 工程步骤）
- [x] `locomotion-reward-design-guide.md`（奖励函数分类/调参/失败模式诊断）

### P3 自动化与 CI
- [x] `.github/workflows/lint.yml`（所有分支 push/PR 自动运行 lint）
- [x] `scripts/ingest_paper.py`（一行生成 sources/papers/ 模板）
- [x] `Makefile` 新增 `make ingest NAME=xxx TITLE=yyy DESC=zzz`
- [x] `export_minimal.py` 支持生成 `docs/sitemap.xml`（89 wiki/entity URL + 4 静态页）
- [x] `detail.html` 动态填充 `<meta description>`（从 summary 字段，截取前 160 字符）

### P4 前端体验
- [x] `index.html` 搜索框新增类型过滤下拉（concept/method/task/comparison/entity/query）
- [x] `data-wiki-tag` 点击委托→填入搜索框并触发搜索
- [x] tech-map 节点卡片：`📄 ingest` / `— no ingest` 标记（联动 `has_ingest` 字段）
- [x] `ingest-badge` CSS 样式（主题色徽章 + muted 缺失态）
