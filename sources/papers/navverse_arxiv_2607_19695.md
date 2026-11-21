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
- **最近核对：** 2026-11-21
- **通读：** 已完成
- **一句话说明：** 基于 Isaac Sim 5.1.0 + Isaac Lab 2.3.0 的室内–室外连通具身导航基准：100 室内 GRScenes + 50 Virtual Community 城市户外 + 50 门–立面混合场景，10,000 episode（ObjNav / PlaceNav / VLN），暴露出口寻找（25–48% 无法走到室外）与过渡后性能骤降；已于 2026-09-30 开源。

---

## 原文通读要点（arXiv:2607.19695）

1. **问题：** 配送、校园、应急响应等场景要求机器人在**单次连续 episode** 内从建筑内部走到街道；既有 benchmark 多把室内/户外分开评，且常抽象掉机器人执行，出口寻找、边界穿越、域迁移与运动学失败未被系统度量。
2. **场景规模与构造：** **100** GRScenes 室内、**50** Virtual Community 城市户外（OSM 道路下沉 0.2 m + 121 个 Objaverse 资产 + 101 个 Gemini3-Pro 店面外观）、**50** 门–立面（door-to-facade）室内–户外连通场景（scene-disjoint split）。
3. **Episode 套件与本体（10,000）：** ObjNav **4,027**、PlaceNav **2,973**（餐厅、银行等 POI）、VLN **3,000**；主评测以 **Boston Dynamics Spot** 四足控制接口执行（Table 7 另含 Spot 与 Turtlebot 跨本体对照）。
4. **评测指标：** 除 **SR / SPL** 外包含 **CE**（coverage efficiency）、**CR**（碰撞率）、**ADO**（平均障碍距离）、**NSR**（可通行表面比）及 transition 阶段诊断。
5. **仿真栈：** **Isaac Sim 5.1.0 + Isaac Lab 2.3.0** 物理启用 rollout。
6. **核心实验结论（§5）：** RL（PoliFormer）、VLA（UniNaVid）、模块化（SGImagineNav）、VLA-RL（LongNav-R1）全零样本评测显示 **UniNaVid** SR 最高（ObjNav 11.62%、PlaceNav 11.38%、VLN 10.67%），但其 ObjNav 在室内→室外（I2O）场景由 17.65% 跌至 **3.64%**，且 25–48% 失败样本全程未能找到出口走出室外；**SGImagineNav** 安全与可通行表面比最佳（CR 0.21、NSR 0.16）。

## 开源与仓库核查（2026-11-21）

| 已发布组件 | 核查结论 |
|------------|----------|
| arXiv PDF & HTML | 已公开发布（`arXiv:2607.19695`） |
| 项目页 | 含榜单（2026-11-09 接入 GPT-6 Astra）、transition 诊断与定性视频 |
| GitHub 主代码库 | **已开源**（2026-09-30 释放）：含 `navverse/`、`baselines/`、`scripts/`、`configs/`、`docker/`、`setup.py`、`tests/` 及 `navverse-run` 启动器 |
| 场景与 Episode 数据 | **已开源**：Hugging Face `tccoin/navverse-benchmark` 提供 `navverse_v1_july23.tar.gz`（含 `vc_plus`、`grscenes_commercial`、`navmesh`、`episodes/`）与 SHA-256 校验 |

**综合判定：** **已开源** — 代码、基线包装器、Isaac Sim Docker 与 v1 场景/episode 资产均已发布。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-navverse.md`](../../wiki/entities/paper-navverse.md)
- 项目页：[`sources/sites/navverse-benchmark-github-io.md`](../sites/navverse-benchmark-github-io.md)
- 仓库：[`sources/repos/navverse-benchmark.md`](../repos/navverse-benchmark.md)
- 任务交叉：[视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md)、[零样本物体导航](../../wiki/tasks/zero-shot-object-navigation.md)
