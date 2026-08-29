# CMoE 项目页（hoshi-no-ai.github.io）

- **标题：** CMoE — Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots
- **类型：** site / project-page
- **URL：** <https://hoshi-no-ai.github.io/CMoE/>
- **arXiv：** <https://arxiv.org/abs/2603.03067>
- **代码：** <https://github.com/Hoshi-No-Ai/CMoE>
- **视频：** <https://www.youtube.com/watch?v=Q95Ssg1FP7A>
- **机构：** 复旦大学（Fudan University）智能机器人与先进制造学院
- **平台：** Unitree G1
- **会议：** ICRA 2026
- **配套论文归档：** [`sources/papers/cmoe_contrastive_mixture_of_experts_icra_2026.md`](../papers/cmoe_contrastive_mixture_of_experts_icra_2026.md)
- **入库日期：** 2026-08-23
- **复核日期：** 2026-08-29

## 一句话摘要

复旦大学 CMoE 项目页：单阶段 MoE 人形 locomotion，用 **对比学习** 解决 Vanilla MoE 门控均匀激活；仿真 Isaac Gym + 真机 G1 高程图感知，展示 20 cm 台阶、80 cm 沟与混合跑酷。

## 开源状态（步骤 2.5，截至 2026-08-29）

| 资源 | 状态 |
|------|------|
| 项目页 + 摘要图 + Framework 图 + BibTeX | **已发布** |
| arXiv PDF | **已发布**（2603.03067） |
| YouTube 演示 | **已发布** |
| 官方 GitHub（`Hoshi-No-Ai/CMoE`） | **已发布**（`legged_gym` + `rsl_rl`，task `g1cmoe`，alg `cmoe`） |
| `Fudan-MAGIC-Lab/CMoE` | **空占位**（README clone 指令仍指向它，不可用） |
| 预训练 checkpoint / 权重托管 | **未列出**（README 仅说明 logs 路径） |
| 真机高程图 / 部署 | README 指向 [elevation_mapping_humanoid](https://github.com/smoggy-P/elevation_mapping_humanoid) 与 [rl_sar](https://github.com/fan-ziqi/rl_sar) |
| mjlab 移植 | 社区 [senlanke/mimic](https://github.com/senlanke/mimic) 任务 `CMoE-G1` |

**结论：已开源（仿真训练栈完整）；权重与真机雷达→高程图部署需自行接社区仓，官方仓不提供 onboard 包。**

## 公开信息要点

- **核心叙事：** Vanilla MoE 专家激活分散、缺乏地形聚类 → CMoE 用对比约束使门控输出与同地形高程表征对齐。
- **框架三件套：** MoE actor-critic、SwAV 式对比目标、本体 VAE + 地形 AE 双 estimator。
- **真机数字：** 连续 20 cm 台阶、80 cm 沟、30 cm 栏、17° 坡；混合地形与户外未训练台阶、扰动实验。
- **BibTeX：** `@inproceedings{ma2026cmoe, booktitle={ICRA 2026}, ...}`

## 为何值得保留

- 与 [MoRE](../../wiki/entities/paper-amp-survey-08-more.md)、[TRAMP](../../wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md) 形成 **MoE 人形感知** 对照轴：CMoE 专攻 **门控塌缩 / lazy gating**，而非步态命令或 AMP 先验。
- 官方代码与论文模块命名一致，适合补 **源码运行时序图**。
- 高程图 + 雷达真机管线是人形 perceptive locomotion 的工程锚点。

## 关联资料

- 论文归档：[`sources/papers/cmoe_contrastive_mixture_of_experts_icra_2026.md`](../papers/cmoe_contrastive_mixture_of_experts_icra_2026.md)
- 代码归档：[`sources/repos/cmoe.md`](../repos/cmoe.md)
- 沉淀实体：[`wiki/entities/paper-cmoe.md`](../../wiki/entities/paper-cmoe.md)
