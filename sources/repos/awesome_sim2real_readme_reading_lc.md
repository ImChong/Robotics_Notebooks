# AwesomeSim2Real README 通读（上游清单视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/lc-awesome-sim2real-technology-map.md`）

- **标题：** AwesomeSim2Real（README.md 全文）
- **作者 / 维护方：** Longchao Da（LongchaoDa）等
- **类型：** repo / awesome-list / curated-index
- **原文：** <https://github.com/LongchaoDa/AwesomeSim2Real>（2026-11 通过 raw.githubusercontent.com 获取 main 分支 README.md 全文，285 行）
- **一句话说明：** 按「Surveys and Simulators」与「Technique Papers」两大块组织的 RL 视角 Sim2Real 论文清单；技法部分按 MDP 要素 Observation / Action / Transition / Reward 分组，每条给标题、作者、会议/年份与链接。
- **核对范围：** 通读 README 全部 285 行：研究趋势说明、目录、Survey Papers（7 条）、Environments and Benchmarks 的 8 个子类（Robotics / Transportation / Recommender / Other 的 Environments 与 Benchmarks）、Technique Papers 的 4 大块及其子分组、Cite 段。未逐条打开清单所链接的 139 篇论文原文（那些由各自的 `lc_awesome_sim2real_*` 归档负责）；README 内的图片（year_distribution、framework、table）为图片文件，未读取其像素内容。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位：** 收集近年顶会/期刊上的 RL 型 sim-to-real 论文，并欢迎通过 issue 补充；与综述 arXiv:2502.13187 配套（Cite 段）。
- **结构：** Surveys and Simulators（Survey Papers + Environments and Benchmarks）与 Technique Papers 两大块。
- **环境/基准分域：** Robotics、Transportation、Recommender System、Other，每域再分 Environments 与 Benchmarks（含 MuJoCo、PyBullet、robosuite、CARLA、SUMO、RecSim 等）。
- **技法分组：** Observation（Domain Randomization / Domain Adaptation / Sensor Fusion / Foundation Models）；Action（Action Space Scale / Action Delay / Action Uncertainty / Foundation Models）；Transition（Domain Randomization / Domain Adaptation / Grounding Methods / LLM-Enhanced Approaches）；Reward（Reward Shaping / LLM-Based Reward Design）。
- **条目特征：** 每条是「加粗标题 + 作者 + 会议/年份 + [link]」，部分带 GitHub badge；部分条目的归类较宽松（例如 Observation/Foundation Models 下含 GPT-4 技术报告、医学图像分割，Reward 下含数学奥赛 MCTS 论文），且同一论文（如 robosuite）可在多处出现；Transition 的 Domain Adaptation 下含 *Overcoming the Sim-to-Real Gap* 一条。
- **原文无评分/对比：** README 不提供论文间的定量比较，仅作导航。

## 对 wiki 的映射

- [`wiki/overview/lc-awesome-sim2real-technology-map.md`](../../wiki/overview/lc-awesome-sim2real-technology-map.md)
- [`wiki/entities/awesome-sim2real.md`](../../wiki/entities/awesome-sim2real.md)
