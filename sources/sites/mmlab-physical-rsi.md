# Physical RSI 1.0 — MMLab 项目页

> 来源归档（site / project page）

- **标题：** Physical RSI 1.0: Recursive Self-Harness for Scaling Embodied Skills
- **类型：** site
- **URL：** <https://mmlab.hk/research/PhysicalRSI>
- **机构叙事：** [HKU MMLab](https://mmlab.hk/) 研究页；页脚链 [Kinetix AI](https://www.kinetixai.tech/) 合作伙伴；演示视频 CDN `assets.kinetixai.cn/0928/`
- **入库日期：** 2026-09-28（2026-11-07 重新抓取整页并复核，见下「2026-11-07 复核」）
- **核对范围：** 2026-11-07 抓取整页 HTML（约 22 MB，含内联脚本），去标签后通读全部可见正文（标题与引言、Embodied Self-Harness 公式与伪代码、最小实现、四类 harness 修订示例、麻将 kong / 叠衣 / 技能继承 / 解方程等技能示例、分类对比与 52 个设置的任务级结果、进化轨迹、技能库说明）；另解析页内内嵌 JSON（`benchmarkEntries` 共 54 条榜单、`category-analysis.json` 五维类别对比）。**未读** 页内引用的视频（evolution-sequence.mp4 与 rollout 视频）和 `skill-clips.json` / `settings-gallery.json` / `fold-clothes-evolution.json` 的逐条内容；代码仓库 README 另读，见下
- **通读：** 已完成
- **一句话说明：** **Physical RSI 1.0** 提出 **Darwinian Self-Harness**：System 2 多模态 agent **F** 根据具身反馈 **τ** 改写 System 1 可执行 harness **H**（路由、code-policy skills、工具与 π₀.₅ 等 motor tools），在候选 agent 间 **评测–选择–继承**；截至 **2026-09-28** 在 **RoboDojo-Sim** 官方 Overall **#1**（Score **36**、SR **31%**）。

## 2026-11-07 复核（整页重新抓取，与 2026-09-28 版有出入）

- **榜单口径更新：** 页内 Official Overall 现为 **Score 36.27 / SR 31.38%**，标注「As of October 8, 2026 ranks #1 on the RoboDojo-Sim leaderboard」；数据注「官方榜单 2026-10-03 更新，2026-10-08 核验」。内嵌 `benchmarkEntries`（54 条）前几名：PhysicalRSI 36.27 / 31.38、Awomo-0.5 35.34 / 29.64、Simate-beta 33.95 / 27.96、Rex-M1 Preview 33.79 / 27.44、VPP2-Preview 31.40 / 25.62。
- **五维类别（Score / SR，页内 `category-analysis.json`）：** Generalization 21.30 / 15.62；Precision 38.05 / 32.50；Long horizon 46.37 / 37.33；Memory 46.74 / 46.56；Open 28.88 / 24.92。同页对照：Open 上 GPT-6 Astra 34.36 / 31.0 高于 PhysicalRSI；Generalization 上 Simate-beta 35.09 / 27.95、GPT-6 Astra 33.36 / 30.50 均高于 PhysicalRSI；Long horizon 上 Simate-beta 57.84 / 43.42 高于 PhysicalRSI。
- **代码已公开：** 页头 GitHub 链与「Architecture Code / Baseline Code · XPolicyLab」指向 <https://github.com/PhysicalRSI/PhysicalRSI>（Apache-2.0；README 写「Status: Core release」，pip 包 `physicalrsi` 0.3.0；本次浅克隆 HEAD `2574f72b`，2026-11-02）与 <https://github.com/XPolicyLab/XPolicyLab> 的 `policy/PhysicalRSI/`（HEAD `6baccd57`，2026-11-04；含 `eval.sh`、`model.py`、`runtime/PhysicalRSI*`；`eval.sh` 要求设置 `PHYSICALRSI_AGENT_API_KEY` 与冻结的 `PHYSICALRSI_SKILL_CONFIG`）。
- **仓库 README 要点（已读全文，未读源码与测试）：** 主分支是通用核心——`SkillDescriptor` / `compose_sequence` / `SkillRegistry` 描述并组合带类型契约的技能；harness 是定义搜索面、评测协议、资源预算与证据的数据；Darwinian Self-Harness 由 System 2 提出候选、冻结对照集上评测并选出幸存者，成对检验默认推荐 paired-permutation（README 称在 RSI-Bench-0 intercept 任务上 600 次选择决策零误继承，15 轮达 86–87% oracle 成功率，对比 paired-hoeffding 的 42–52%，为 README 自述）；语言提议器只改 `ParameterSpace` 声明的 JSON 数字字段，模型文本不当代码执行；基准适配器与演示放在 `benchmarks-and-demos` 分支，不在主分支。
- **仍未见：** 项目页与仓库 README 都未列 arXiv 论文；权重 / 数据集链接未见。

## 开源状态（项目页核查 2026-09-28，已被上节部分取代）

| 项 | 结论 |
|----|------|
| **论文 / arXiv** | 项目页 **未列** arXiv 或 PDF 下载链接 |
| **代码 / 权重** | 项目页 **未列** GitHub、Hugging Face 或 ModelScope |
| **评测与媒体** | 链 [RoboDojo Leaderboard](https://robodojo-benchmark.com/leaderboard)；站内 **212** 段 skill clip、**40** 任务 rollout 视频（Kinetix CDN）；页内引用 `settings-gallery.json` / `skill-clips.json` 等相对路径（静态托管未单独暴露，以页面内嵌数据为准） |
| **综合** | **待发布** — 以 RoboDojo 官方 verified 上榜规则为准，完整训推复现入口尚未公开 |

## 页面核心主张（归纳）

1. **下一代具身 AI 路线：** Native **System 1–2 协同进化** — Planning（S2）→ Acting（S1）↺ Reflecting（S2→H）。
2. **相对基线：** 对比 **GPT-as-policy**（慢/贵）、**VLA/WAM**（复合误差与 OOD）、Physical RSI 强调 **高效 + 开放世界泛化**。
3. **主算法：** \(A_k=\mathrm{Agent}(F,H_k)\)，\(H_{k+1}=\mathrm{Improve}(A_k,H_k,\tau_k)\) — vary → evaluate → select → inherit。
4. **最小实现：** System 2 用 \(F(H_k,E^+,E^-)\) 修订 harness；System 1 用 \(H_k(o_t,g,m_t;S,T)\to a_t\) 执行 code-policy skills，并可调用 **π₀.₅** 等 VLA 作 motor tool。
5. **技能库：** 页内展示 **212** clips / **40** tasks 的共享 skill space；示例含麻将 **make kong**、叠衣 **harness memory**、skill 继承与组合（方程求解 \(\pi_{eq}=\sigma_{place}\circ\cdots\)）。

## RoboDojo-Sim 榜单（页内 Official Overall，2026-09-28）

| 指标 | Physical RSI |
|------|----------------|
| **Overall 排名** | **#1** |
| **Score** | **36** |
| **SR** | **31%** |

页内对照模型包括 **π₀.₅**（motor tool）、**DM0.5**、**Liber-0 Lite**、**GPT-6 Astra**、**Simate-beta** 等；分任务 SR/Score 表覆盖 Generalization / Long horizon / Memory / Open / Precision 五维（部分均值为 **≈** 估计）。

## 对 wiki 的映射

- 主实体：[physical-rsi](../../wiki/entities/physical-rsi.md)
- 基准：[robodojo](../../wiki/entities/robodojo.md)
- 对照 harness 线：[HarnessPAI](../../wiki/entities/paper-harnesspai.md)、[Harness VLA](../../wiki/entities/paper-harness-vla.md)
- RSI 选型：[rsi-four-tier-five-pushes](../../wiki/queries/rsi-four-tier-five-pushes.md)
