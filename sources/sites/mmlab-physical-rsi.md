# Physical RSI 1.0 — MMLab 项目页

> 来源归档（site / project page）

- **标题：** Physical RSI 1.0: Recursive Self-Harness for Scaling Embodied Skills
- **类型：** site
- **URL：** <https://mmlab.hk/research/PhysicalRSI>
- **机构叙事：** [HKU MMLab](https://mmlab.hk/) 研究页；页脚链 [Kinetix AI](https://www.kinetixai.tech/) 合作伙伴；演示视频 CDN `assets.kinetixai.cn/0928/`
- **入库日期：** 2026-09-28
- **一句话说明：** **Physical RSI 1.0** 提出 **Darwinian Self-Harness**：System 2 多模态 agent **F** 根据具身反馈 **τ** 改写 System 1 可执行 harness **H**（路由、code-policy skills、工具与 π₀.₅ 等 motor tools），在候选 agent 间 **评测–选择–继承**；截至 **2026-09-28** 在 **RoboDojo-Sim** 官方 Overall **#1**（Score **36**、SR **31%**）。

## 开源状态（项目页核查 2026-09-28）

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
