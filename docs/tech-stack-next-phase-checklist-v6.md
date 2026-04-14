# 技术栈项目执行清单 v6

最后更新：2026-04-14
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v5.md`](tech-stack-next-phase-checklist-v5.md)
方法论参考：[Karpathy LLM Wiki](../wiki/references/llm-wiki-karpathy.md)

---

## V5 完成基线（V6 起点）

| 维度 | V5 末状态 |
|------|----------|
| wiki 页面总数 | 96（concepts 21 / methods 7 / entities 8 / queries 9 / tasks 5 / formalizations 5 / comparisons 4 / 其他） |
| sources/papers/ 文件 | 16 |
| Sources 覆盖率 | 53%（31/59 wiki 页有 ingest 来源） |
| Lint 健康 | ✅ 10 项检测，2 contextual findings（PPO/MPC 语境性矛盾，非真正错误） |
| Query 产物 | 9 个（wiki/queries/ 目录） |
| log.md | ✅ 建立，append_log.py + `make log` 支持 |
| 矛盾检测 | ✅ CANONICAL_FACTS（2 条规则） |
| 陈旧检测 | ✅ mtime 比对（1 天容差） |
| ingest 平均覆盖页数 | ⚠️ ~2–3 页（Karpathy 目标 10–15 页） |
| Search | ⚠️ 纯关键词匹配，无排序 |
| 前端键盘导航 | ❌（P5 暂缓） |

---

## V6 阶段总目标

> 基于 Karpathy LLM Wiki 原文的三个核心洞见做深度对齐：
> 1. **"A single source might touch 10–15 wiki pages"** — 当前每次 ingest 平均只更新 2–4 个页面，远低于理想值。V6 把每次 ingest 的覆盖深度提升到 8–12 个页面。
> 2. **"Search engine with hybrid BM25/vector search"** — 当前 search_wiki.py 是纯关键词匹配，无排序。V6 引入 TF-IDF 加权排序。
> 3. **"Sources coverage 75%+"** — 从 53% 推进至 75%（目标 44/59 页有 ingest 来源）。

---

## P0 · Ingest 深度提升（"1 source → 10–15 pages" 目标）

**背景**：Karpathy 原文：*"A single source might touch 10–15 wiki pages."* 我们目前每个 sources 文件平均只直接链接 2–5 个 wiki 页面。Ingest 的价值应体现在：不只新建 2 个页面，而是在所有相关页面中都留下痕迹（新增 cross-reference、补充关联链接、标注 ingest 来源）。

### 0.1 ingest_coverage.py — 覆盖审计工具

- [x] 新建 `scripts/ingest_coverage.py`：给定一个 `sources/papers/*.md`，扫描其覆盖的关键词（从标题、wiki 映射中提取），找到所有提及这些关键词但未链接该 sources 文件的 wiki 页面，输出建议更新清单。

- [x] `Makefile` 新增：`make coverage F=<sources-path>`

### 0.2 回填现有 sources 文件的覆盖范围

对 16 个现有 sources 文件，逐一运行 coverage checker，补充缺失的 wiki 页面 ingest 链接。
目标：每个 sources 文件平均覆盖 **6+ 个** wiki 页面（当前平均 ~2–3 个）。

关键回填目标：

| sources 文件 | 当前链接页数 | 建议增补 wiki 页面 |
|-------------|------------|-------------------|
| `simulation_tools.md` | 2 | legged-gym.md, domain-randomization.md, sim2real.md |
| `state_estimation.md` | 3 | balance-recovery.md, locomotion.md, floating-base-dynamics.md |
| `privileged_training.md` | 5 | curriculum-learning.md, reward-design.md |
| `whole_body_control.md` | ? | tsid.md, hqp.md, mpc-wbc-integration.md |
| `policy_optimization.md` | ? | rl-algorithm-selection.md, wbc-vs-rl.md |
| `robot_kinematics_tools.md` | 4 | optimal-control.md, trajectory-optimization.md |

- [x] 回填 `sources/papers/simulation_tools.md`（+legged-gym, sim2real, domain-randomization）
- [x] 回填 `sources/papers/state_estimation.md`（+locomotion, balance-recovery, floating-base-dynamics）
- [x] 回填 `sources/papers/privileged_training.md`（+curriculum-learning, reward-design, gait-generation）
- [x] 回填 `sources/papers/whole_body_control.md`（已覆盖 8 页，含 tsid/hqp/mpc-wbc/capture-point/balance-recovery）
- [x] 回填 `sources/papers/policy_optimization.md`（+rl-algorithm-selection, curriculum-learning, reward-design）
- [x] 回填 `sources/papers/robot_kinematics_tools.md`（+optimal-control, trajectory-optimization, floating-base-dynamics）

### 完成标准
- `scripts/ingest_coverage.py` 可运行，输出格式正确
- 16 个 sources 文件平均覆盖 wiki 页面数从 ~2.5 提升到 **6+**
- Sources 覆盖率从 53% → **60%+**（不新建 sources 文件，只靠回填）

---

## P1 · Sources 层扩充（覆盖率 53% → 75%）

**背景**：目前 28 个 wiki 页面（47%）没有 ingest 来源，包括重要的 balance-recovery、reward-design、curriculum-learning、diffusion-policy、footstep-planning 等。

### 1.1 新增 sources 文件

| 目标文件 | 覆盖 wiki 页面 | 关键论文/资源 |
|---------|--------------|--------------|
| `sources/papers/reward_design.md` | reward-design.md, curriculum-learning.md, locomotion.md | Rudin 2021 reward shaping; Zhuang 2023 terrain curriculum; OpenAI RAPID; Escontrela 2022 |
| `sources/papers/footstep_and_balance.md` | footstep-planning.md, capture-point-dcm.md, balance-recovery.md, lip-zmp.md | Kajita 2003 ZMP; Pratt 2006 capture point; Koolen 2012 DCM; Herdt 2010 online walking |
| `sources/papers/diffusion_and_gen.md` | diffusion-policy.md, imitation-learning.md, motion-retargeting.md | Chi 2023 diffusion policy; Black 2023 π₀; Reuss 2023 BESO; Hansen 2024 TDMPC2 |
| `sources/papers/teleoperation.md` | loco-manipulation.md, motion-retargeting.md（+ 新建 wiki/tasks/teleoperation.md） | OmniH2O; ALOHA; ACT; UMI |
| `sources/papers/contact_planning.md` | contact-complementarity.md, footstep-planning.md, whole-body-control.md | Deits 2014 footstep regions; Tonneau 2018; Dai 2014 contact invariant optimization |

- [x] `sources/papers/reward_design.md`（覆盖 reward-design / curriculum-learning / locomotion / legged-gym）
- [x] `sources/papers/footstep_and_balance.md`（覆盖 lip-zmp / capture-point-dcm / balance-recovery / footstep-planning）
- [x] `sources/papers/diffusion_and_gen.md`（覆盖 diffusion-policy / imitation-learning / loco-manipulation / motion-retargeting）
- [x] `sources/papers/teleoperation.md`（覆盖 loco-manipulation / motion-retargeting + 新建 teleoperation.md）
- [x] `sources/papers/contact_planning.md`（覆盖 contact-complementarity / footstep-planning / whole-body-control）

### 1.2 wiki 参考来源链接化

下列 wiki 页面已有文字引用但未链接到 sources/：

- [x] `wiki/concepts/reward-design.md` → `sources/papers/policy_optimization.md` + `privileged_training.md`
- [x] `wiki/concepts/curriculum-learning.md` → `sources/papers/privileged_training.md` + `policy_optimization.md`
- [x] `wiki/concepts/capture-point-dcm.md` → `sources/papers/footstep_and_balance.md`
- [x] `wiki/concepts/lip-zmp.md` → `sources/papers/footstep_and_balance.md`
- [x] `wiki/tasks/balance-recovery.md` → `sources/papers/state_estimation.md`
- [x] `wiki/methods/diffusion-policy.md` → `sources/papers/diffusion_and_gen.md`
- [x] `wiki/concepts/footstep-planning.md` → `sources/papers/footstep_and_balance.md` + `contact_planning.md`
- [x] `wiki/formalizations/contact-complementarity.md` → `sources/papers/contact_planning.md`
- [x] `wiki/concepts/motion-retargeting.md` → `sources/papers/teleoperation.md` + `diffusion_and_gen.md`
- [x] `wiki/tasks/loco-manipulation.md` → `sources/papers/teleoperation.md` + `diffusion_and_gen.md`

### 完成标准
- sources/papers/ 文件数：**21**（+5 新建）✅
- Sources 覆盖率：**73%**（45/62 wiki 页有 ingest 来源）⚠️（接近 75% 目标）
- 每个新 sources 文件至少覆盖 **4 个** wiki 页面 ✅

---

## P2 · Wiki 内容缺口补全

### 2.1 新增 wiki 页面

| 文件 | 优先级 | 内容要点 |
|------|--------|---------|
| `wiki/tasks/teleoperation.md` | 高 | 遥操作系统设计；数据采集；ALOHA/ACT；延迟补偿；双臂控制 |
| `wiki/queries/reward-design-guide.md` | 中 | reward shaping 决策树；稀疏 vs dense；多目标 reward 权重调参 checklist |
| `wiki/queries/sim2real-gap-reduction.md` | 高 | sim2real 差距分类（model error / actuator / sensing）；DR 策略选择；参数对齐 checklist |
| `wiki/methods/imitation-learning.md` 深化 | 中 | 加入 GAIL 原理、DAgger 算法、对比 BC/IRL/IL+RL hybrid |

- [x] `wiki/tasks/teleoperation.md`（新建，满足 schema 最低质量标准）
- [x] `wiki/queries/reward-design-guide.md`（新建，500+ 字，含决策树 + 权重表格 + 陷阱对照表）
- [x] `wiki/queries/sim2real-gap-reduction.md`（新建，600+ 字，含 gap 分类 + 修复表格 + 完整 checklist）
- [-] `wiki/methods/imitation-learning.md` → 内容深化（当前 167 行已有 GAIL/DAgger 基础，推迟至 v7）

### 2.2 现有页面深化

- [-] `wiki/concepts/reward-design.md` → 已有 potential-based shaping 和多目标设计，新建 reward-design-guide.md query 产物补充
- [-] `wiki/formalizations/contact-complementarity.md` → 已添加 contact_planning.md ingest 链接，TO 示例推迟至 v7
- [x] `wiki/methods/model-based-rl.md` → 加入 RSSM 结构（GRU 确定/随机分离）、latent imagination 伪代码、DreamerV3 改进表格

### 完成标准
- 3 个新页面 + 3 个深化
- `make lint` 仍保持 ✅（0 新增真实 issues）
- wiki/queries/README.md 同步更新新增的 2 个 query 产物

---

## P3 · Search 增强（TF-IDF 排序）

**背景**：Karpathy 原文明确提到 *"qmd: BM25/vector hybrid search with LLM re-ranking"* 作为 wiki 搜索工具。我们的 search_wiki.py 是纯关键词字符串匹配，无排序，短词查询会返回大量噪声结果。

### 3.1 search_wiki.py TF-IDF 排序

- [x] `scripts/search_wiki.py` 升级：TF × coverage × title_boost 评分，按分排序输出，显示 score 标注

### 3.2 `--json` 输出模式

- [x] `--json` 参数：输出 JSON 格式（path / score / type / tags / snippet）

### 3.3 Makefile 增强

- [x] `make search Q=<关键词>` 自动调用 TF-IDF 版本（Makefile 已有此 target）

### 完成标准
- `python3 scripts/search_wiki.py "MPC solver"` 时 mpc-solver-selection.md 排第一
- `--json` 输出格式正确
- `make search Q=<词>` 调用更新后的版本

---

## P4 · 前端体验提升（V5 P5 延续）

### 4.1 搜索体验

- [ ] `docs/index.html` 搜索结果键盘导航：↑↓ 选中高亮，Enter 打开详情页，Esc 清空搜索
- [ ] `docs/index.html` 搜索框下方标签云：统计高频 tag（Top 20），点击直接过滤

### 4.2 detail.html 增强

- [ ] 关联页面列表改为卡片式（显示 type badge + summary 片段）
- [ ] `og:image` 动态设置（基于 page_type 选择预设图标 URL）

### 4.3 tech-map.html 增强

- [ ] 层级筛选器多选（Ctrl+Click 多层级同时显示）
- [ ] 页面内节点搜索（实时过滤匹配节点）

> 注：前端改动需要 browser 验证，每项完成后使用 `make export && open docs/index.html` 测试，无法验证则标记 `[-]`。

### 完成标准
- 键盘导航可用（至少 ↑↓/Enter）
- 标签云正常渲染

---

## P5 · Karpathy 高级特性（wiki 复利最大化）

**背景**：Karpathy 原文提到的若干高级特性，我们尚未实现：Marp 幻灯片生成、wiki 内链图谱自动生成、Web 资料快速导入。

### 5.1 Marp 幻灯片生成

> 用途：从 wiki 页面一键生成课程 / 会议报告幻灯片，让 wiki 内容可复用为演示文稿。

- [x] `scripts/wiki_to_marp.py`：每个 H2 → 一张幻灯片，跳过导航节（参考来源/关联页面），输出 exports/slides/

- [x] `Makefile` 新增：`make slides F=<wiki-path>`（已加入）

### 5.2 Wiki 内链图谱自动生成

- [x] `scripts/generate_link_graph.py`：61 nodes / 342 edges → `exports/link-graph.json`
- [-] `docs/graph.html`：需要 browser 验证，推迟至 v7
- [x] `Makefile` 新增：`make graph`

### 5.3 Web 资料快速导入（fetch_to_source.py）

- [x] `scripts/fetch_to_source.py`：urllib 抓取 → 提取 title/description → 生成 sources/blogs/ 模板

- [x] `Makefile` 新增：`make fetch URL=<url> NAME=<stem>`

### 完成标准
- `make slides F=wiki/methods/model-predictive-control.md` 生成 Marp 文件
- `make graph` 生成 link-graph.json，docs/graph.html 可渲染
- `make fetch URL=<url>` 生成 sources/ 模板文件

---

## P6 · Lint 深化与 CI 增强

### 6.1 CANONICAL_FACTS 扩展（2 → 10 条）

- [x] `scripts/lint_wiki.py`：将 CANONICAL_FACTS 从 2 条扩展至 6 条（2→10 目标，v7 继续补全）

  已新增检测规则：

  | 规则 ID | 主题 | 正面断言 | 负面断言 |
  |--------|------|---------|---------|
  | 3 | DR 必要性 | DR 是 sim2real 必须步骤 | DR 会降低 in-distribution 性能 |
  | 4 | RL 推理速度 | RL policy 推理速度快 | RL policy 推理延迟高 |
  | 5 | WBC 计算复杂度 | WBC 可实时运行 | WBC 计算量大无法实时 |
  | 6 | 接触力估计精度 | 接触力估计精确 | 接触力仿真 sim2real gap 大 |

  推迟至 v7 的规则（需要更多 wiki 内容积累）：

  | 规则 ID | 主题 |
  |--------|------|
  | 7 | TSID vs QP 等价性 |
  | 8 | 全身控制必要性 |
  | 9 | 接触模型精度（MuJoCo） |
  | 10 | 仿真频率稳定性 |

### 6.2 GitHub Actions auto-export

- [ ] `.github/workflows/export.yml`：push 到 main 时自动运行 `make catalog && make export`，提交更新的 JSON 回仓库

### 6.3 覆盖率目标 Badge

- [ ] `README.md` 中加入动态 Sources 覆盖率 badge（目标：75%+ 时为绿色）

### 完成标准
- CANONICAL_FACTS 增至 8 条规则
- export.yml 触发并成功运行

---

## 维护操作标准（V6 更新版）

### Op 1：Ingest（添加新资料）—— 目标 8–12 页面覆盖
```bash
1. make ingest NAME=xxx TITLE="..." DESC="..."  # 生成 sources/papers/xxx.md 模板
2. 编辑模板，填写论文摘录（至少 3 条核心论文 + wiki 映射）
3. make coverage F=sources/papers/xxx.md       # 审计覆盖范围（V6 新步骤）
4. 在报告建议的 wiki 页面中补充 cross-reference 和 ingest 链接
5. 在对应 wiki 页面 ## 参考来源 加入 ingest 档案链接
6. make lint       # 确认 0 issues
7. make catalog    # 更新 index.md
8. make export     # 同步 JSON + sitemap
9. make log OP=ingest DESC="sources/papers/xxx.md — 简述，覆盖 N 个页面"
```

### Op 2：Query（知识查询）
```bash
1. make search Q=<关键词>                       # TF-IDF 排序搜索（V6 升级版）
2. python3 scripts/search_wiki.py <关键词> --related  # 加载邻居页面
3. 综合多页面分析，得出结论
4. 如有独立价值 → 保存为 wiki/queries/xxx.md
5. 更新 wiki/queries/README.md 的表格
6. make lint && make catalog && make export
7. make log OP=query DESC="关键词 → wiki/queries/xxx.md"
```

### Op 3：Lint（健康检查）
```bash
make lint                              # 完整健康检查（10 项 + 覆盖率报告）
make log OP=lint DESC="0 issues，覆盖率 XX%，矛盾 N 个"
```

### Op 4：Index（索引更新）
```bash
make catalog    # 刷新 index.md（Page Catalog）
make export     # 更新 exports/ JSON + sitemap.xml
make graph      # 更新 link-graph.json（V6 新增）
```

---

## Karpathy 对齐度评估（V6 目标）

| Karpathy 原则 | V5 末状态 | V6 目标 |
|-------------|----------|--------|
| Raw sources（不可变 sources 层） | ✅ 16 文件，53% 覆盖 | **21+ 文件，75% 覆盖** |
| Wiki（LLM 维护的 md 文件集） | ✅ 96 页，互联完整 | **100+ 页** |
| Schema（配置与规范文档） | ✅ schema/ 5 文件 | 同步更新 |
| Ingest "1 source → 10–15 pages" | ⚠️（平均 2–3 页） | **✅（目标 8–12 页，coverage checker 辅助）** |
| Query 产物 | ✅ 9 个 | **12 个** |
| Lint（矛盾/陈旧/孤儿/覆盖率） | ✅ 10 项检测，2 条 CANONICAL_FACTS | **8 条 CANONICAL_FACTS** |
| Search（BM25/vector） | ⚠️ 纯关键词匹配 | **✅ TF-IDF 排序 + `--json` 输出** |
| log.md | ✅ 持续追加 | ✅ 持续追加 |
| Marp 幻灯片生成 | ❌ | **✅ wiki_to_marp.py** |
| 图谱视图（wiki 内链自动生成） | ⚠️ 手动维护 tech-map | **✅ generate_link_graph.py + graph.html** |
| 前端键盘导航 | ❌ | **✅** |
| Web 资料快速导入 | ❌ | **✅ fetch_to_source.py** |

---

## 状态约定
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[-]` 暂缓
