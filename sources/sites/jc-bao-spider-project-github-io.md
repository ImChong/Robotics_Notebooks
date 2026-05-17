# jc-bao.github.io/spider-project（SPIDER 项目页）

> 来源归档（ingest）

- **标题：** SPIDER — Scalable Physics-Informed DExterous Retargeting
- **类型：** site / project-page
- **官方入口：** <https://jc-bao.github.io/spider-project/>
- **入库日期：** 2026-05-17
- **一句话说明：** 论文配套站点：概括 **人体+物体运动学输入 → 运动学重定向 → 物理采样** 管线、**灵巧手 / 人形** 交互可视化（多机型 tab）、**真机部署** 与 **数据增强 / 接触引导** 对比区，以及 **BibTeX**；正文知识编译见 wiki 方法页与 `sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md`。

## 页面公开信息（检索自 2026-05-17）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://jc-bao.github.io/spider-project/> |
| 论文 abs | <https://arxiv.org/abs/2511.09484> |
| 网站源码（GitHub） | <https://github.com/jc-bao/spider-project> |

## 站点结构摘录（便于 wiki 溯源）

- **Pipeline 叙事：** Input = Human motion + Object motion（含 mesh）→ Step 1 Kinematic retargeting → Step 2 Physics-based sampling → Output = Dynamically feasible trajectory。
- **部署区：** 强调生成轨迹**动力学可行**故可直驱实机；展示多任务短视频（取勺、吉他、灯泡旋转、拔插、抓取等）。
- **灵巧手与人形分区：** 分 tab 的交互轨迹可视化（Allegro / Inspire / Schunk / XHand / Ability 等；Unitree G1/H1-2、Fourier N1、Booster T1 等）；支持 log_time / sim_time 切换。
- **数据增强：** 物体尺寸、地形、力等维度的 origin vs augmented 对照叙事。
- **接触引导：** With vs Without contact guidance 的并排演示（Allegro、约束 G1 等）。

## 对 wiki 的映射

- [`wiki/methods/spider-physics-informed-dexterous-retargeting.md`](../../wiki/methods/spider-physics-informed-dexterous-retargeting.md)
- [`sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md`](../papers/spider_scalable_physics_informed_dexterous_retargeting.md)
- [`sources/repos/jc-bao-spider-project.md`](../repos/jc-bao-spider-project.md)
