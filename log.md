# Change Log

> 这是 Robotics_Notebooks 的 append-only 操作日志。
> 每个条目格式：`## [date] op | category | title`
> 可用 unix 工具解析：`grep "^## \[" log.md | tail -10`
>
> 与 `docs/change-log.md` 的分工：
> - `log.md` = 每次操作的时间线（ingest / query / lint / structural）
> - `docs/change-log.md` = 重要结构性变更的里程碑记录

---

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
