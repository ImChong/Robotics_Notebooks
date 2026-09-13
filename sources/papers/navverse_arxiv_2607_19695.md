# NavVerse: Benchmarking Indoor-to-Outdoor Embodied Navigation in Continuous Robot Simulation

> 来源归档

- **标题：** NavVerse: Benchmarking Indoor-to-Outdoor Embodied Navigation in Continuous Robot Simulation
- **类型：** paper / benchmark / VLN / ObjNav / navigation / Isaac Sim
- **arXiv：** <https://arxiv.org/abs/2607.19695>（v1，2026-07-22）
- **PDF：** <https://arxiv.org/pdf/2607.19695>
- **项目页：** <https://umich-curly.github.io/NavVerse-Benchmark/>
- **GitHub：** <https://github.com/UMich-CURLY/NavVerse-Benchmark>
- **作者：** Junzhe Wu, Yue Hu, Zeyu Han, Po-Hsun Chang, Yinan Dong, Behrad Rabiei, Maani Ghaffari
- **机构：** 密歇根大学（University of Michigan / UMich）；CURLY 实验室
- **入库日期：** 2026-09-13
- **一句话说明：** 物理启用的室内–室外连通具身导航基准：100 室内 + 50 城市户外 + 50 室内外连通场景，10k episode 覆盖 ObjNav / PlaceNav / VLN，在 Isaac Sim 可执行 rollout 上同时报告成功、效率与安全指标。

---

## 摘要级要点

1. **问题：** 配送、校园、应急响应等场景要求机器人在**单次连续 episode** 内从建筑内部走到街道；既有 benchmark 多把室内/户外分开评，且常抽象掉机器人执行，出口寻找、边界穿越、域迁移与运动学失败未被系统度量。
2. **场景规模：** **100** 室内、**50** 城市户外、**50** 室内–户外连通（scene-disjoint split）；城市资产含道路/地形/车辆，室内布局通过门–立面装配接到户外。
3. **Episode 套件（10,000）：** ObjNav **4,027**、PlaceNav **2,973**、VLN **3,000**。
4. **任务：** **ObjNav**（物体类别目标）、**PlaceNav**（语义地点/POI，如餐厅/银行）、**VLN**（自然语言路线指令）。
5. **评测：** 可执行机器人接口 rollout；指标除 **SR / SPL** 外含 **CE**（coverage efficiency）、**CR**（碰撞率）、**ADO**（平均障碍距离）、**NSR**（可通行表面比）及 transition 诊断（室内-only / reach-outside / pre-exit / post-exit 等）。
6. **仿真栈：** **Isaac Sim** 物理启用 rollout。
7. **零样本基线（论文摘要）：** RL（PoliFormer）、VLA（UniNaVid）、模块化（SGImagineNav）、VLA-RL（LongNav-R1）；**端到端 VLA 零样本 SR 最高**，模块化方法 **安全指标最强**；PlaceNav 在 outdoor → indoor-to-outdoor 上跌幅最明显，适应仍是瓶颈。

## 开源边界（步骤 2.5，截至 2026-09-13）

| 已发布 | 备注 |
|--------|------|
| arXiv PDF | abs + PDF 可下载 |
| 项目页 | 榜单、transition 诊断、定性 rollout 视频 |
| GitHub 仓 | 公开，但默认分支 `website` **仅静态项目站** |
| Isaac Sim 仿真 / 评测代码 | 项目页标注 **Code Coming soon**；仓内未见 |
| 场景 / episode 数据 | **未见** 公开下载入口 |

**综合判定：** **待发布** — 论文与项目站已上线，可复现 benchmark 代码与数据仍待官方发布。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-navverse.md`](../../wiki/entities/paper-navverse.md)
- 项目页：[`sources/sites/navverse-benchmark-github-io.md`](../sites/navverse-benchmark-github-io.md)
- 仓库：[`sources/repos/navverse-benchmark.md`](../repos/navverse-benchmark.md)
- 任务交叉：[视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md)、[零样本物体导航](../../wiki/tasks/zero-shot-object-navigation.md)
