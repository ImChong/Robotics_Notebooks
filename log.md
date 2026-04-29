# Change Log

> 这是 Robotics_Notebooks 的 append-only 操作日志。
> 每个条目格式：`## [date] op | category | title`
> 可用 unix 工具解析：`grep "^## \[" log.md | tail -10`
>
> 与 `docs/change-log.md` 的分工：
> - `log.md` = 每次操作的时间线（ingest / query / lint / structural）
> - `docs/change-log.md` = 重要结构性变更的里程碑记录

---

## [2026-04-13] ingest | paper | ULTRA: Survey 完成

- 新建 `wiki/tasks/ultra-survey.md`（ULTRA 统一多模态 loco-manipulation 控制器笔记）
- 在 `wiki/tasks/locomotion.md` 的关联系统区块新增 ULTRA 链接
- 同步更新 `index.md` Page Catalog，新增 ULTRA 条目（Wiki Tasks 任务页）

## [2026-04-13] ops | page-catalog-automation | Page Catalog 自动化脚本

- 新建 `scripts/generate_page_catalog.py`：自动扫描 wiki/、roadmap/、tech-map/、references/ 目录生成 Page Catalog markdown
- 替换 `index.md` 中手动维护的 Page Catalog 为脚本生成版本
- 以后新增页面后运行 `python3 scripts/generate_page_catalog.py >> index.md` 即可同步

## [2026-04-13] ingest | paper | ULTRA: Survey of Humanoid Locomotion RL ← 计划但未执行

- 规划了 ULTRA 论文 ingest：在 wiki/concepts/ 建立 ultra-survey 页面，更新 wiki/tasks/locomotion 的 related，新增 tags: rl, survey, humanoid
- **实际未执行**（ultra-survey 页面不存在，locomotion related 链接未建立）
- 待后续有空闲窗口时补上

---

## [2026-04-13] ingest | wiki-quality | 补全所有 entity 页的关联页面区块

- 系统检查发现 6 个 entity 页（crocoddyl / isaac-gym-isaac-lab / legged-gym / mujoco / pinocchio / unitree）缺少"关联页面"区块
- 为每个 entity 页在"推荐继续阅读"和"一句话记忆"之间新增"关联页面"区块，包含 2-3 个相关 wiki 页链接
- 这是 schema/page-types.md 质量标准落地的重要一步

---

## [2026-04-13] ingest | sources-backfill | 补全 sources/ 资料层结构

- 系统检查发现 sources/ 资料层严重缺失（193 条 git commit 没有任何 sources 条目）
- 建立 `sources/papers/`、`sources/repos/`、`sources/blogs/` 三个子目录骨架
- 补全实体来源归档：mujoco.md、isaac_gym_isaac_lab.md、pinocchio.md、crocoddyl.md、unitree.md、legged_gym.md
- 补全论文来源归档：locomotion_rl.md、sim2real.md、survey_papers.md、imitation_learning.md、whole_body_control.md、humanoid_hardware.md
- 重写 `sources/README.md`，按 papers/repos/blogs/notes/根目录散文件 五区重组

---

## [2026-04-13] ingest | paper | Detail page 相关推荐模块

- 基于 Karpathy LLM Wiki 分析，开始补齐项目 Wiki-Ops 体系
- 在 `detail.html` 新增"相关推荐" section，基于 tag 匹配自动推荐 5 条相关 detail pages
- 在 `main.js` 新增 `findRelatedByTags()` 函数，按 tag 重合数排序
- 更新 `docs/change-log.md` 记录本次 detail page 锚点深链闭环

---

## [2026-04-13] ingest | wiki-ops | 开始补齐 LLM Wiki Ops 体系

- 基于 Karpathy LLM Wiki 模式，分析项目现状与 LLM Wiki 三层架构的差距
- 发现已有部分基础设施（`schema/ingest-workflow.md`、`log.md`、`index.md`）但未真正使用
- 制定 LLM Wiki Ops 补齐计划：log.md（append-only）/ ingest-workflow.md（补 Query/Lint）/ index.md（page catalog）/ AGENTS.md（ops 约束）
- 同步更新 `docs/change-log.md` 的版本演进记录

---

## [2026-04-13] lint | health-check | Tech-stack checklist v2 定期检查

- 审视 P0/P1/P2 待办完成度
- 确认 detail page / module page / roadmap page / tech-map page 四类核心页面均已进入可浏览状态
- 更新 `docs/tech-stack-next-phase-checklist-v2.md` 的当前状态判断

---

## [2026-04-13] ingest | wiki-content | Detail page 阅读体验全链路升级

- 将 detail page 正文从 raw markdown `<pre>` 升级为基础 markdown 渲染容器（支持标题、列表、引用、代码块、粗体、链接）
- 接入 KaTeX CSS/JS 与 auto-render，detail page 公式从最小样式高亮升级为真正数学排版
- 新增 detail 正文侧边目录导航区块（TOC），根据正文自动生成
- 给 detail 标题追加锚点复制按钮，TOC 根据滚动位置自动高亮当前章节
- 建立基于 path 的站内 markdown 内链解析，wiki/references/roadmap 相对链接统一回流到 detail.html?id=... 或 roadmap.html?id=...
- 这一系列使 detail page 从"能看"推进到"能稳定导航长文"

---

## [2026-04-12] structural | page-routing | 统一 detail / module / roadmap route

- 将 `docs/tech-map.html` 从静态说明页升级为 data-driven 页面
- 新增 `docs/module.html`（统一 module route）和 `docs/roadmap.html`（统一 roadmap route）
- 扩展 `docs/main.js`，新增 tech-map / module / roadmap page 渲染器
- 统一四类页面之间的跳转方式到 `detail.html?id=...` / `module.html?id=...` / `roadmap.html?id=...`
- 在 `scripts/export_minimal.py` 增加 `docs/exports/` 镜像导出，打通 GitHub Pages 部署链路

---

## [2026-04-12] structural | page-routing | tech-map 筛选与分组增强

- tech-map 页面增加 layer filter，最小筛选能力
- tech-map 当前 layer 同步到 URL 查询参数 `?layer=...`
- tech-map 节点列表改为按 layer 分组的可折叠 `details` 区块
- 这一步让 tech-map 从"全量节点展示"变为"可按层筛选"

---

## [2026-04-12] structural | export-layer | 页面级聚合导出落地

- 扩展 `scripts/export_minimal.py`，生成 `exports/site-data-v1.json`，包含 5 类页面聚合结果
- 新增 `docs/site-data-preview.html` 作为页面级聚合导出的最小验证页
- 直接消费 `site-data-v1.json` 渲染首页摘要、模块页数据、路线页数据
- 扩展 `scripts/export_minimal.py`，为 `detail_pages` 新增 `content_markdown` 字段，同步源 markdown 正文
- 项目从"对象 schema 导出"推进到"页面消费层聚合导出"

---

## [2026-04-11] ingest | wiki-entities | 第一批实体页补全

- 新增 `wiki/entities/legged_gym.md`（ETH RSL 足式机器人 RL 训练框架）
- 新增 `wiki/entities/mujoco.md`（Google DeepMind 物理引擎）
- 新增 `wiki/entities/isaac-gym-isaac-lab.md`（NVIDIA GPU 加速仿真框架）
- 新增 `wiki/entities/pinocchio.md`（机器人运动学/动力学底层引擎）
- 新增 `wiki/entities/crocoddyl.md`（Pinocchio 之上的最优控制框架）
- 新增 `wiki/entities/unitree.md`（Unitree 硬件与 SDK 平台）
- 同步在 `sources/repos/` 建立对应来源归档条目

---

## [2026-04-11] ingest | wiki-content | 第二批关键概念页深化

- 新增 `wiki/concepts/capture-point-dcm.md`（Capture Point 与 DCM 步行平衡方法）
- 新增 `wiki/concepts/floating-base-dynamics.md`（浮动基动力学）
- 新增 `wiki/concepts/contact-dynamics.md`（接触动力学）
- 补全 `wiki/concepts/centroidal-dynamics.md` / `lip-zmp.md` / `tsid.md` / `state-estimation.md` / `system-identification.md`

---

## [2026-04-11] structural | references-cleanup | 梳理 references 与 sources 职责边界

- 重写 `references/README.md` 和 `references/papers/`、`references/repos/`、`references/benchmarks/` 各子目录 README
- 明确三层职责边界：sources（原始资料输入）→ references（论文/仓库导航索引）→ wiki（结构化知识）
- references 层现在具备快速入口和主线映射能力

---

## [2026-04-11] ingest | wiki-content | 第一批关键概念页深化

- 深化 `wiki/concepts/` 主干，补全 Capture Point / DCM / Floating Base Dynamics / Contact Dynamics 等关键节点
- 为关键实体页（Isaac Gym / Isaac Lab、MuJoCo、legged_gym、Pinocchio、Crocoddyl、Unitree）补充 references 入口
- 更新 `references/` 三个子目录（papers / repos / benchmarks）为具备快速入口和主线映射的导航层

---

## [2026-04-11] structural | v2-phase | V2 阶段正式启动

- 重写 `README.md` 为真正入口指南（适合谁 / 怎么用 / 从哪开始 / 项目结构）
- 重写 `index.md` 为导航总入口（快速入口表 / 四模块分工 / 推荐阅读顺序 / 主线知识链）
- 重写 `roadmap/route-a-motion-control.md`（L0 → L6 完整执行路线）
- 重写两个 learning path（Locomotion RL / Imitation Learning 各 6 Stage）
- 重写 `tech-map/overview.md` 和 `tech-map/dependency-graph.md`
- 建立 `docs/tech-stack-next-phase-checklist-v2.md`
- V1 阶段完成：Wiki 主干基本成型，目录体系确立

---

## [2026-04-07] structural | project-kickoff | 重构启动

### 重构启动
- 将项目方向从"思维导图资源堆叠"转向"机器人研究与工程知识库"
- 新增 `AGENTS.md`
- 新增 `schema/` 规则目录
- 新增 `sources/`、`wiki/`、`exports/` 骨架
- 新增首批 MVP wiki 页面
- 暂不修改当前思维导图网页渲染逻辑

### 设计决策
- 底层知识组织采用 wiki / 图结构，而不是强制树结构
- 展示层（思维导图）后续通过 `exports/` 从 `wiki/` 导出
- [`Humanoid_Robot_Learning_Paper_Notebooks`](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks) 与 [`Robotics_Notebooks`](https://github.com/ImChong/Robotics_Notebooks) 分工明确：前者深读论文，后者整理知识图谱

### README 瘦身
- 将原根目录 README 的大型资源链接列表归档到 `sources/notes/legacy-readme-resource-map.md`
- 新增 `sources/README.md` 作为资料层入口
- 将根目录 `README.md` 改成项目说明 + 知识入口 + 资料入口，减少首页堆叠

---

## [2026-04-14] structural | karpathy-alignment | 对齐 Karpathy LLM Wiki 方法论

### 核心改进内容

**1. YAML Frontmatter 规范化（Obsidian Dataview 支持）**
- 为 12+ 核心 wiki 页面补全 `type`/`tags`/`status` frontmatter 字段
- 更新 `schema/page-types.md`：新增 YAML Frontmatter 规范区块，含 Dataview 查询示例

**2. 参考来源（Sources 溯源）**
- 为 10+ 核心 wiki 页面新增 `## 参考来源` 区块
  - wiki/concepts/: lip-zmp, tsid, whole-body-control, centroidal-dynamics, sim2real, state-estimation
  - wiki/methods/: reinforcement-learning, model-predictive-control, imitation-learning, policy-optimization, diffusion-policy
  - wiki/tasks/: locomotion, manipulation, loco-manipulation
  - wiki/formalizations/: bellman-equation, mdp
  - wiki/comparisons/: wbc-vs-rl
  - wiki/overview/: robot-learning-overview
- 更新 `schema/page-types.md` 各模板：`参考来源` 列为 Must-Have 必填项
- 更新 `schema/ingest-workflow.md`：Ingest Step 5 和 Query Step 5 明确要求写 sources
- 更新 `AGENTS.md`：lint checklist 新增"缺失参考来源"

**3. 自动化 Lint 脚本**
- 新建 `scripts/lint_wiki.py`
  - 检测孤儿页（无入链）、缺失关联页面、缺失参考来源、断链、空壳页（< 200 字）
  - `--write-log` 标志将报告追加到 log.md
  - lint 结果：0 孤儿页、0 断链、0 空壳页；21 个"缺失参考来源"待后续补全

**4. 自动化搜索工具**
- 新建 `scripts/search_wiki.py`
  - 支持关键词（AND 逻辑）、`--type`、`--tag`、`--context` 过滤
  - ANSI 高亮输出；用法：`python3 scripts/search_wiki.py MPC locomotion`

**5. P2 内容扩展（3 个新页面）**
- 新建 `wiki/methods/policy-optimization.md`（PPO/SAC/TRPO/AWR 对比与机器人应用）
- 新建 `wiki/methods/diffusion-policy.md`（扩散策略、DDPM vs DDIM、模仿学习应用）
- 新建 `wiki/tasks/loco-manipulation.md`（移动操作任务定义、挑战、四种方法路线）

**6. Query 产物规范化**
- 新建 `wiki/queries/README.md`：说明 query 目录定位，追踪现有 query 产物
- 更新 `schema/ingest-workflow.md`：独立 query 洞见路由到 `wiki/queries/`

**7. 其他 bug 修复**
- 修复 `scripts/generate_page_catalog.py`：`extract_first_sentence()` 增加 frontmatter 剥离，避免描述显示"---"
- 修复 `wiki/formalizations/bellman-equation.md` 和 `mdp.md` 的断链（`./optimal-control.md` → `../concepts/optimal-control.md`）
- 修复 `wiki/entities/pinocchio.md` 的断链（`./tsid.md` → `../concepts/tsid.md`）
- 修复 `wiki/concepts/state-estimation.md` 的关联页面区块层级（`###` → `##`）
- 更新 `index.md` Page Catalog，包含三个新建 P2 页面

### lint 历史对比
- 改进前：82 个问题（大量断链、孤儿页）
- 改进后：21 个问题（0 断链，0 孤儿页，0 空壳页；剩余均为"缺失参考来源"可逐步回填）

---

## [2026-04-14] ingest | papers | P0 Sources 层激活（checklist v3 推进）

- 完成 `sources/papers/locomotion_rl.md`、`whole_body_control.md`、`imitation_learning.md`、`sim2real.md` 的 ingest 摘要升级（标题/贡献/wiki 映射）
- 更新 `sources/README.md`：为 papers 列表增加 `[x]/[ ]/[-]` 提炼状态标记
- 在 6 个 wiki 页面的 `## 参考来源` 中加入 `sources/papers/*.md` 实体链接（RL/IL/Sim2Real/WBC/TSID/HQP）
- 同步更新 `docs/tech-stack-next-phase-checklist-v3.md` 的 P0 任务状态与推进记录

---

## [2026-04-14] ingest | sources/papers/optimal_control.md — Bellman 1957 / Pontryagin 1962 / Bryson & Ho 1975 / DDP / Bertsekas 2019，覆盖 optimal-control / trajectory-optimization

## [2026-04-14] ingest | sources/papers/mpc.md — Mayne 2000 / Di Carlo 2018 / Sleiman 2021 / Wieber 2006 / MPPI，覆盖 model-predictive-control / mpc-wbc-integration

## [2026-04-14] ingest | sources/papers/policy_optimization.md — PPO / SAC / TD3 / TRPO / Rudin 2022，覆盖 reinforcement-learning / locomotion

## [2026-04-14] ingest | sources/papers/model_based_rl.md — DreamerV3 / MBPO / PETS / TD-MPC2 / Dyna，覆盖 model-based-rl / reinforcement-learning

## [2026-04-14] structural | v4 清单推进 — 新增 wiki 概念页 curriculum-learning / contact-estimation / motion-retargeting；新增 comparisons/model-based-vs-model-free；新增实体页 drake.md；query 产物 humanoid-hardware-selection / wbc-implementation-guide / locomotion-reward-design-guide

## [2026-04-14] structural | .github/workflows/lint.yml — GitHub Actions CI，每次 push 自动运行 lint_wiki.py

## [2026-04-14] structural | scripts/ingest_paper.py + Makefile ingest — 快速生成 sources/papers/ 模板工具

## [2026-04-14] index | make export → sitemap.xml（89 URL）；前端新增类型过滤 + ingest 标记；rebase 合并 main 的 sitemap/meta 改动

## [2026-04-14] structural | docs/tech-stack-next-phase-checklist-v5.md — 新建 V5 执行清单（P0-P6），深度对齐 Karpathy LLM Wiki；README 同步更新至 89 页 / v5 清单 / make ingest 命令

## [2026-04-14] structural | P0 运营层建立 — log.md 激活（已有文件补全近期条目）+ scripts/append_log.py + make log + schema/log-format.md

## [2026-04-14] lint | P0 验证通过 — log.md 建立 + append_log.py + make log + schema/log-format.md

## [2026-04-14] ingest | sources/papers/contact_dynamics.md + privileged_training.md + state_estimation.md + system_identification.md + simulation_tools.md + robot_kinematics_tools.md — P1 Sources 层扩充，覆盖 7 个 wiki 页面

## [2026-04-14] structural | P2 Wiki 缺口补全 — footstep-planning / gait-generation / contact-complementarity（3 个新页面）；P3.4 lint 覆盖率报告；P4 3 个 Query 产物（humanoid-rl-cookbook / pinocchio-quick-start / mpc-solver-selection）

## [2026-04-14] lint | 0 issues，覆盖率 50%（29/58），wiki 95 页

## [2026-04-14] structural | v5 所有 P0–P6 完成：log.md 建立 / 6 个 sources 文件 / 6 个 wiki 页面 / lint 10 项检测 / 9 个 query 产物 / route-b-fullstack 扩充

## [2026-04-14] structural | v6 checklist 新建：Ingest 深度 / TF-IDF 搜索 / Sources 75% 目标 / Marp / link-graph / fetch_to_source

## [2026-04-14] lint | lint 10 项全绿，2 contextual findings（PPO/MPC 语境性矛盾），覆盖率 53%（31/59）

## [2026-04-14] structural | v6 P0-P6 完成：ingest_coverage / TF-IDF 搜索 / 5 新 sources 文件 / 3 新 wiki 页面 / Marp+link-graph+fetch 工具 / CANONICAL_FACTS 2→6 条 / 覆盖率 53%→73%

## [2026-04-17] query | manipulation/vla guides | wiki/queries/il-for-manipulation.md + wiki/queries/vla-deployment-guide.md

## [2026-04-17] structural | v10-p1-p2 | 新增 methods/concepts/query 页面，补 summary frontmatter，并扩展 lint_wiki.py 到 14 项检查/30 canonical facts

## [2026-04-17] structural | V10 全量完成：向量搜索/内容扩充/lint增强/前端离线搜索/图谱社区/Anki导出/README与清单同步

## [2026-04-17] structural | 建立 V11 执行清单并同步 README 到 V11 入口

## [2026-04-17] structural | 将 V10/V11 执行清单移动到 docs/checklists/ 并同步更新 README

## [2026-04-18] lint | ultra-survey follow-up | 回写 2026-04-13 历史日志状态

- 核对确认 `wiki/tasks/ultra-survey.md` 已存在，`wiki/tasks/locomotion.md` 已包含 ULTRA 关联链接
- 说明 2026-04-13 的“计划但未执行”条目已被后续同日 ingest 实际完成，当前不构成积压
- 保留原历史日志不改写，仅通过 follow-up 追加状态说明，符合 append-only 约束

## [2026-04-18] lint | sources-backfill follow-up | 回写 2026-04-14 缺失参考来源积压状态

- 核对确认 2026-04-14 `karpathy-alignment` 条目中记录的“21 个缺失参考来源待后续补全”已被后续工作消化
- 证据一：同日后续已有 `## [2026-04-14] lint | 0 issues，覆盖率 50%（29/58），wiki 95 页` 条目，说明缺失参考来源问题已清零
- 证据二：当前再次运行 `python3 scripts/lint_wiki.py`，结果仍为 0 issues，且“缺少参考来源区块”为 0、Sources 覆盖率为 81/81（100%）
- 保留原历史日志不改写，仅通过 follow-up 追加状态说明，符合 append-only 约束

## [2026-04-18] query | motion-control-projects | 飞书公开文档结构化摘要 + PDF Sources 入库

- 新建 `sources/papers/motion_control_projects.md`：将飞书公开文档《【开源】小而美的运动控制项目》中可见的 14 个 PDF 附件统一归档为 sources
- 新建 `wiki/queries/open-source-motion-control-projects.md`：按训练机制优化 / Parkour / 动作模仿 / 物体交互 / 动作重定向五条主线整理结构化摘要
- 更新 `wiki/queries/README.md` 与 `sources/README.md` 索引
- 回填 `contact-estimation.md`、`curriculum-learning.md`、`motion-retargeting.md` 的来源与更新时间，消除 sources 比 wiki 更新的陈旧提示
- 运行 `make lint && make export && make graph && make badge`，结果：lint 0 issues，Sources 覆盖率 82/82（100%），知识图谱更新为 84 节点 / 498 边

## [2026-04-18] query | humanoid-motion-control-know-how | 飞书 Know-How 结构化摘要入库

- 新建 `sources/papers/humanoid_motion_control_know_how.md`：把飞书公开文档《人形机器人运动控制 Know-How》整理为 sources 输入，提炼趋势 / 路线 / 问题框架 / 传统控制主线
- 新建 `wiki/queries/humanoid-motion-control-know-how.md`：提炼成适合 `Robotics_Notebooks` 的结构化摘要，强调路线层、问题层、方法层三层组织
- 更新 `wiki/queries/README.md` 与 `sources/README.md` 索引；保留 `sources/notes/know-how.md` 作为旧资源树归档
- 运行 `make lint && make export && make graph && make badge`，结果：lint 0 issues，Sources 覆盖率 83/83（100%），知识图谱更新为 85 节点 / 510 边

## [2026-04-18] structural | route-a-and-method-templates | 路线页增强 + 方法骨架补齐

- 更新 `roadmap/route-a-motion-control.md`：显式加入 Know-How 的读法，强调路线层 / 问题层 / 方法层三层阅读顺序，并把传统控制主线与 learning-based 主线更明确拆开
- 更新 `wiki/concepts/optimal-control.md`、`lip-zmp.md`、`whole-body-control.md` 与 `wiki/methods/model-predictive-control.md`
- 为上述方法页补入统一的“最小代码骨架 / 方法局限性”结构，使其更符合“原理 → 最小代码 → 局限性”的学习模板
- 运行 `make lint && make export && make graph && make badge`，结果：lint 0 issues，Sources 覆盖率 83/83（100%），知识图谱维持 85 节点 / 510 边

## [2026-04-18] structural | control-chain-template-phase-2 | 补齐传统控制主线剩余方法模板

- 更新 `wiki/concepts/centroidal-dynamics.md`、`tsid.md`、`state-estimation.md`
- 为三页补入统一的“最小代码骨架 / 方法局限性 / 学这个方法时最该盯住的点”模板，使传统控制主线从 OCP → LIP/ZMP → Centroidal Dynamics → MPC → TSID/WBC → State Estimation 的页面风格更一致
- 运行 `make lint && make export && make graph && make badge`，结果：lint 0 issues，Sources 覆盖率 83/83（100%），知识图谱维持 85 节点 / 510 边

## [2026-04-18] structural | 同步 README 顶部 badge 到实际仓库统计，并补强 lint 对 README badge/checklist 漏报检测

## [2026-04-18] structural | 首页 Hero 节点/边/coverage 统计改为读取导出 JSON 自动同步，避免手写数值过期

## [2026-04-18] structural | 修正 Graph View 筛选逻辑：按类型模式筛类型，按社区模式筛社区

## [2026-04-18] structural | Graph View 默认排斥力从 400 调整为 800

## [2026-04-18] structural | 修复 Graph View PC 端 hover 卡片失效：筛选逻辑重构后残留 activeTypes 引用导致 mouseenter 报错

## [2026-04-18] structural | 首页知识图谱预览改为跟随全局白天/黑夜主题切换

## [2026-04-19] feat | v12-execution | P0-P5 推进中

- P0: 消除 3 个孤立社区（singleton_communities=[]），图谱社区从 6 → 4；节点健康着色（health_score 0-3）加入 graph.html
- P1: robot-policy-debug-playbook + simulator-selection-guide + demo-data-collection-guide + ppo-vs-sac-for-robots 四个新 Query 页（学习路径在 main 分支已存在）
- P2: CANONICAL_FACTS 30 → 40 条，搜索回归 12 → 18 条（100% 通过）
- P3: ingest_paper.py 新增 --suggest-updates，check_export_quality.py 新增 index.md 同步检测（11 项全通过）
- P4: PWA manifest.json + sw.js 创建，index.html/graph.html 注册 Service Worker
- P5: mujoco-vs-isaac-lab + ppo-vs-sac 对比页（新建中）
- 运行 make lint: 0 errors（7 陈旧警告来自 main branch sources 更新），sources 覆盖率 100%，图谱 92 节点 560 边

## [2026-04-20] structural | v13-execution | V13 启动，P0-P5 规划

- 基于 V12 完成交付启动 V13：目标聚焦图谱健康、知识积累与检索质量三条主线
- P0：补齐孤儿节点入链，修复 `roadmap/reference` 页面缺失 `type:`，把 `unknown` 降到 0
- P1：扩展 `CANONICAL_FACTS` 到 50 条，覆盖 PPO on-policy、CLF/CBF、VLA、contact-rich manipulation、Isaac Lab 并行训练等事实
- P2：加深 manipulation / contact-rich / terrain / bimanual / sensor-fusion / behavior-cloning / VLA 等薄弱页面
- P3：规划新增 4 个高价值 Query 页（domain-randomization / clf-cbf / vla-low-level / contact-rich manipulation）
- P4：lint 增加孤儿节点计数检测，搜索回归扩展到 26 条
- P5：补齐安全控制与接触操作学习路径，扩展 overview / index / README 入口，并保持日志 append-only 更新

## [2026-04-20] fix | v14-execution | P0 搜索回归修复：numpy 延迟导入

- V14 P0 完成：`scripts/search_wiki.py` 移除 module 顶部 `import numpy as np`，改为在 `load_vector_resources` / `encode_query_vector` / `search` 内部使用 numpy 的分支做延迟导入
- 修复前：`python3 scripts/search_wiki.py PPO` 直接 `ModuleNotFoundError: No module named 'numpy'`，回归测试 0/26
- 修复后：`python3 scripts/eval_search_quality.py` 通过率 **26/26 (100%)**，BM25 路径不再依赖 numpy
- 无新增依赖、无行为变更；向量搜索分支在 numpy 可用时保持原逻辑

## [2026-04-21] structural | V14 执行清单完整交付 (新增 11 页 + 深度扩充 3 页 + 脚本优化)

## [2026-04-21] structural | V15 执行清单初始化

## [2026-04-21] structural | V15 执行清单完整交付 (聚焦操作社区与软件栈实体)

## [2026-04-21] structural | V16 执行清单初始化 (具身大模型深度化 + 灵巧操作补完)

## [2026-04-21] structural | V16 执行清单完整交付 (具身大模型深度化 + 灵巧操作补完)

## [2026-04-21] structural | V17 执行清单完整交付 & 初始化 V18 (具身数据流与交互闭环)

## [2026-04-21] structural | V20 执行清单完整交付 & 初始化 V21 (具身触觉专题 + 硬件通信链路形式化)

## [2026-04-22] ingest | sources/repos/fusion2urdf.md, sources/repos/marathongo.md — 接入新仓库资料并更新全站索引

## [2026-04-23] ingest | robot_lab (Repo) & CLAW (Blog) — 接入 IsaacLab 扩展框架与 G1 合成数据管线并更新全站索引

## [2026-04-24] ingest | sources/papers/policy_optimization.md — 补充 BRRL/BPO（Bounded Ratio RL）论文、项目页与代码仓库，并更新 Policy Optimization 方法页参考来源

- 新增来源条目：`Bounded Ratio Reinforcement Learning (Ao et al., 2026)`（arXiv / project page / GitHub）
- 更新 `wiki/methods/policy-optimization.md`：新增“BRRL / BPO（2026）”进展说明与参考来源
- 关联强化：将 BRRL 映射到 policy-optimization / reinforcement-learning / ppo-vs-sac / locomotion

## [2026-04-24] structural | V21 P1 推进 | wiki/formalizations/contact-wrench-cone.md

- V21 P1「触觉与力觉闭环」首项：新建 `wiki/formalizations/contact-wrench-cone.md`，把 Friction Cone 从 3D 点接触力推广到 6D 面接触力旋量（CWC / CWS / GWS）
- 涵盖 V-/H- 表示、ZMP/CoP 几何解读、多接触 Minkowski 和 与 Grasp Wrench Space 的统一视角，附最小 Python 骨架与方法局限性
- 回链：`wiki/formalizations/friction-cone.md`、`wiki/concepts/tactile-sensing.md`、`wiki/concepts/contact-dynamics.md` 的关联页面区块新增 CWC 入链
- V21 checklist 对应条目已勾选；follow-up：原同日 log 条目曾在合并 main 时按“冲突以 main 为准”规则被覆盖，此处以 append-only 方式补回

## [2026-04-24] feat | v21-execution | P0 智能拼写纠错（编辑距离）

- V21 P0 第二项推进：`scripts/search_wiki.py` 集成 Levenshtein 编辑距离算法，当查询无结果时自动推荐最接近的 Tag 或 标题
- 新增 `levenshtein_distance` / `collect_known_terms` / `suggest_terms` 三个辅助函数；阈值取 `max(2, ceil(len(query)/2))`，按距离升序返回 Top-5
- `print_results` 增加“您是否想搜索：”分区；`--json` 输出在有 notice 或 suggestions 时切换为 `{notice, suggestions, results}` 字典形式
- `search()` 返回签名保持 `(results, notice)` 不变，`scripts/eval_search_quality.py` 与 `scripts/debug_search.py` 调用方零侵入
- 验证：`python3 scripts/eval_search_quality.py` 通过率 **37/37 (100%)**；`python3 scripts/search_wiki.py "lokomotion" --json` 正确给出 `locomotion`（距离 1）建议
- V21 checklist 对应条目已勾选

## [2026-04-25] lint | 同步 sources/papers/policy_optimization.md 的最新进展（BRRL/BPO 2026）至全站 8 个相关 wiki 页面，消除陈旧预警并更新索引

## [2026-04-25] ingest | sources/repos/roboto_origin.md — 新增 Roboto Origin 及其五个官方模块仓库与文档资料归档，并沉淀 wiki/entities/roboto-origin.md

## [2026-04-25] ingest | sources/repos/atom01_hardware.md, sources/repos/atom01_deploy.md, sources/repos/atom01_train.md, sources/repos/atom01_description.md, sources/repos/atom01_firmware.md — 接入 Roboparty Atom01 五个官方模块仓库原始资料

## [2026-04-25] ingest | wiki/entities/roboto-origin.md — 回链 Atom01 五个 sources 条目并重新导出图谱索引，确保 sources 节点可见

## [2026-04-25] ingest | wiki/entities/atom01-*.md — 新增 Atom01 五个实体页并同步导出图谱（181 nodes / 1012 edges）

## [2026-04-25] feat | v21-execution | P0 自动化背链一致性 Lint（公式变量物理含义检测）

- V21 P0 第三项推进：`scripts/lint_wiki.py` 新增 `formalization_unexplained_vars` 检查；从 `wiki/formalizations/*.md` 的 `$$...$$` 显示公式中抽取单字母拉丁大写变量，逐一验证正文是否给出物理含义解释
- 启发式定义匹配：列表条目冒号、表格行、动词解释（是/为/表示/代表/denote）、其中/where 子句、等式或集合定义（=、\in、\succeq、\succ、\equiv、\triangleq）；并用 `(?![A-Za-z_(])` 排除函数调用形式（如 `R(s,a,s')`）以避免误报
- 修复 6 个被新规则命中的页面，补齐变量物理含义说明：
  - `wiki/formalizations/bellman-equation.md`：Q-learning 更新中的 $R$ 标量
  - `wiki/formalizations/control-lyapunov-function.md`：LQR-CLF 关系中的 $A, B$
  - `wiki/formalizations/ekf.md`：补全 $A, B, C, Q, R, P, K, I$ 整套矩阵物理含义
  - `wiki/formalizations/hjb.md`：LQR 特例段补 $A, B, Q, R, T$
  - `wiki/formalizations/lqr.md`：线性系统模型段补 $A, B, x, u$
  - `wiki/formalizations/tsid-formulation.md`：浮动基座动力学段补执行选择矩阵 $S$
- 验证：`make lint` 0 errors（所有检查通过）；`python3 scripts/eval_search_quality.py` 通过率 **37/37 (100%)**
- V21 checklist 对应条目已勾选

## [2026-04-26] structural | roadmap | 收敛为单一主路线并补齐阶段链接

- 将 `roadmap/route-a-motion-control.md` 从“路线A”调整为唯一主路线，明确 RL/IL/WBC/安全控制/接触操作是目标分支而非并列主路线。
- 删除 `roadmap/route-b-fullstack.md`，避免 roadmap 页面出现 A/B 两套主线。
- 为 L0-L6 阶段补充仓库内概念、方法、形式化、实体、query 页面链接，并同步更新索引与导出。

## [2026-04-26] structural | roadmap | 主路线 ID 去除 A/B 命名

- 将主路线文件从 `roadmap/route-a-motion-control.md` 重命名为 `roadmap/motion-control.md`。
- 将导出 ID 从 `roadmap-route-a-motion-control` 更新为 `roadmap-motion-control`，并同步 README、index、learning paths、wiki 回链与前端入口。
- 在 `docs/main.js` 保留旧 ID 到新 ID 的兼容映射，避免旧 URL 立即失效。

## [2026-04-26] feat | v21-execution | P0 图谱导出数据精简

- V21 P0 第三项收尾：精简 `exports/link-graph.json` 节点字段，去除每个节点上重复的 `community_label`（与 `communities` 数组中的 `label` 完全冗余）。
- `scripts/generate_link_graph.py` 仅在 `communities` 数组保留 `label`，节点端只输出 `community` id；`docs/graph.html` 的 tooltip 与侧栏改为通过 `communityLabelMap.get(d.community)` 查表。
- 体积变化：`exports/link-graph.json` 168 KB → 159 KB（5.7% 缩减），行数 5551 → 5370；`make lint` 与 `check_export_quality.py` 全通过。
- V21 checklist P0 全部完成，进入 P1 触觉与力觉闭环专题阶段。

## [2026-04-27] ingest | sources/repos/lingbot-map.md — 接入 LingBot-Map 流式 3D 重建基础模型方法页

## [2026-04-27] ingest | sources/repos/booster-robocup-demo.md — 接入 Booster Robotics RoboCup 演示框架

## [2026-04-27] ingest | 接入 htwk-gym 与 HumanoidSoccer (PAiD) 足球技能学习框架

## [2026-04-27] ingest | 补全飞书 Wiki 相关的运动控制进阶主题 (BeyondMimic, Any2Track, HAIC, AMP 等)

## [2026-04-27] feat | v21-execution | P1 触觉专题：视触觉融合 (Visuo-Tactile Fusion) 概念页

- V21 P1 触觉学习知识链推进第一项：新增 `wiki/concepts/visuo-tactile-fusion.md`，聚焦“接触瞬间如何在视觉全局与触觉局部之间动态切换权重”。
- 内容覆盖：视觉/触觉互补维度、阶段切换/软门控/注意力级三种融合范式、接触瞬间难点、训练数据采集要点、与现有相关页面的边界澄清（区分 `sensor-fusion`、`multimodal-fusion-tricks`）。
- 同步在 `tactile-sensing`、`contact-rich-manipulation`、`contact-wrench-cone`、`multimodal-fusion-tricks`、`tactile-feedback-in-rl` 中加入回链，避免新页成为孤儿；并刷新 `index.md` Page Catalog。
- 在 `docs/checklists/tech-stack-next-phase-checklist-v21.md` 中将该项打勾。

## [2026-04-28] ingest | 消化 ETH Zurich 关于扩散模型运动生成的论文 (Unitree G1 实时部署)

## [2026-04-28] ingest | 消化 awesome-humanoid-robot-learning 仓库，更新 Loco-Manipulation Wiki 页

## [2026-04-28] ingest | 接入 MimicKit 仓库及其核心算法系列 (DeepMimic, AMP, AWR, ASE, LCP, ADD, SMP)

## [2026-04-28] ingest | 深度更新 SMP (arXiv:2512.03028v3) 技术细节，补充 SDS/ESM/GSI 架构及 Unitree G1 真机验证

## [2026-04-29] ingest | sources/papers/humanoid_touch_dream.md — 消化 HTD / Touch Dreaming 论文并更新触觉增强人形移动操作知识节点

## [2026-04-29] structural | 更新待办列表与主页统计数据 (205 nodes, 1195 edges)

## [2026-04-29] ingest | sources/repos/embodied-ai-guide.md — 接入具身智能全栈百科并补全 RoboTwin/SAPIEN/ALOHA 实体

## [2026-04-29] ingest | sources/repos/xbotics-embodied-guide.md — 接入 Xbotics 工程指南，补全 LeRobot/Genesis 实体与数据飞轮概念

## [2026-04-29] ingest | sources/papers/zest.md — 接入 Boston Dynamics 跨形态技能迁移框架 ZEST

## [2026-04-29] ingest | sources/papers/sumo.md — 接入 RAI Institute 全身移动操作框架 Sumo (MPC-over-RL)
