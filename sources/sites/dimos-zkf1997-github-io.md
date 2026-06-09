# DIMOS 项目页（zkf1997.github.io/DIMOS）

> 来源归档（ingest 配套站点）

- **URL：** <https://zkf1997.github.io/DIMOS/>
- **对应论文：** [DIMOS: Synthesizing Diverse Human Motions in 3D Indoor Scenes](https://arxiv.org/abs/2305.12411)（ICCV 2023）
- **机构：** ETH Zürich / Google
- **入库日期：** 2026-06-09
- **一句话说明：** 官方落地页：方法示意图、演示视频、OOD 泛化案例、训练过程可视化与 BibTeX。

## 页面要点（2026-06 快照）

- **核心叙述：** 将室内 human-scene interaction 合成表述为 **带潜动作空间的 MDP**；场景感知 + 目标驱动策略可生成 **漫游、坐/躺、序列组合** 等行为。
- **方法图：** 总框架（运动基元 + locomotion/interaction 策略 + NavMesh + 静态交互生成）；locomotion 局部可走性图；interaction 的 SDF 邻近特征。
- **定性结果：**
  - novel 椅形、低凳坐下的 OOD 泛化；
  - Replica / PROX 重建房间；
  - Paris-CARLA-3D 公交站长椅点云；
  - Shap-E 文本生成奇怪形状椅子的交互；
  - 策略学习 epoch 0→1000 的「随机乱走 → 走向椅子 → 自然坐下」演化。
- **局限展示：** 运动学穿透仍可见；躺姿数据不足导致不自然。
- **相关项目：** [GAMMA](https://yz-cnsdq-z.github.io/)（3D 场景 locomotion 基座）、[COINS](https://zkf1997.github.io/COINS/)（语义可控静态人–场景交互）。
- **视频：** YouTube / bilibili 外链。
- **BibTeX：** `@inproceedings{Zhao:ICCV:2023, ...}`

## 对 wiki 的映射

- 与 [sources/papers/dimos_arxiv_2305_12411.md](../papers/dimos_arxiv_2305_12411.md) 配对
- 实体页：[wiki/entities/paper-dimos-human-scene-motion-synthesis.md](../../wiki/entities/paper-dimos-human-scene-motion-synthesis.md)
