# Learning Latent Dynamics for Planning from Pixels（PlaNet，arXiv:1811.04551）

> 来源归档（ingest）

- **标题：** Learning Latent Dynamics for Planning from Pixels
- **类型：** paper / PlaNet / RSSM / CEM planning / model-based RL / pixels
- **arXiv：** <https://arxiv.org/abs/1811.04551>（PDF：<https://arxiv.org/pdf/1811.04551.pdf>）
- **项目页：** <https://planetrl.github.io/>（另见 <https://danijar.com/project/planet/>）
- **代码：** <https://github.com/google-research/planet>（Apache-2.0；仓库已 archived）
- **作者：** Danijar Hafner、Timothy Lillicrap、Ian Fischer、Ruben Villegas、David Ha、Honglak Lee、James Davidson
- **机构：** 谷歌（Google Brain）等
- **入库日期：** 2026-07-27
- **一句话说明：** **PlaNet（Deep Planning Network）** 从像素学习带确定性与随机性的 **RSSM** 潜动态，用 **latent overshooting** 多步变分目标，并在潜空间用 **CEM** 在线规划选动作——纯模型基智能体，交互更少却接近强 model-free。

## 开源状态（项目页 + 仓库核查，2026-07-27）

- **已开源：** 项目页指向 [google-research/planet](https://github.com/google-research/planet)（**Apache-2.0**）。入口：`python3 -m planet.scripts.train --logdir … --params '{tasks: [cheetah_run]}'`。仓库状态为 **archived**（依赖 TensorFlow 1.x / 旧 dm_control），可作历史复现锚点；生产向潜空间规划更常接 Dreamer / TD-MPC 谱系。

## 摘要级要点

- **问题：** 图像域学到的动力学常不够准，难以支撑多步规划。
- **模型：** RSSM = 确定性路径 + 随机状态；观测编码进当前潜状态后，在潜空间预测未来回报。
- **目标：** latent overshooting — 多步变分推断，强化长期预测一致性。
- **规划：** CEM 在潜空间搜索动作序列，执行首步后重规划（MPC）。
- **结果：** 接触动力学、部分可观、稀疏回报的连续控制任务上，样本效率显著优于当时强 model-free，终局性能接近。

## 核心论文摘录（MVP）

### 1) RSSM 与潜空间规划

- **链接：** §3 Method；项目页动画
- **摘录要点：** 历史图像编码 → 当前潜状态；多条动作序列在潜空间并行展开估回报；执行最优序列首动作后重规划。
- **对 wiki 的映射：**
  - [PlaNet 实体页](../../wiki/entities/paper-planet-latent-dynamics.md)
  - [Latent Imagination](../../wiki/concepts/latent-imagination.md)
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)

### 2) Latent overshooting

- **链接：** 多步变分目标相关节
- **摘录要点：** 单步 ELBO 不足以约束长程；overshooting 用多步预测一致性提升规划可用的动力学。
- **对 wiki 的映射：**
  - [DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md) — 后续在 RSSM 上改为想象中训 actor-critic

### 3) 与纯随机 / 纯确定性消融

- **链接：** 实验消融；README 参数表
- **摘录要点：** 纯 SSM 或纯确定性路径均弱于混合 RSSM；随机数据收集基线更弱。
- **对 wiki 的映射：**
  - [world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md)

## BibTeX

```bibtex
@inproceedings{hafner2019planet,
  title     = {Learning Latent Dynamics for Planning from Pixels},
  author    = {Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and
               Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James},
  booktitle = {International Conference on Machine Learning},
  pages     = {2555--2565},
  year      = {2019}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-planet-latent-dynamics.md`](../../wiki/entities/paper-planet-latent-dynamics.md)
- 代码：[`sources/repos/google-research-planet.md`](../repos/google-research-planet.md)
- 项目页：[`sources/sites/planetrl-github-io.md`](../sites/planetrl-github-io.md)
- 博客策展：[`sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md`](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
