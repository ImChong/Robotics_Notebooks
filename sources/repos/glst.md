# GLST 官方仓库（Yousef-Liang/glst）

> 来源归档

- **标题：** glst
- **类型：** repo
- **URL：** <https://github.com/Yousef-Liang/glst>
- **论文：** <https://arxiv.org/abs/2610.17927>（归档 [glst_arxiv_2610_17927](../papers/glst_arxiv_2610_17927.md)）
- **许可：** BSD-3-Clause；`glst/actor_critic.py` 派生自 rsl_rl，保留其版权头
- **入库日期：** 2026-10-17
- **一句话说明：** GLST 的 PyTorch 模型发布：潜在全局地图与目标条件导航策略。

## 核查结果（2026-10-17）

- 结构：`glst/` 包、`checkpoints/`、`pyproject.toml`、`LICENSE`、`NOTICE`、`CITATION.cff`；4 次提交，0 star，无 release。
- 内容：潜在全局地图模型（输入局部高度图、可见性掩码与位姿，输出 9×32×32、0.4 m / 格的潜在地图）；局部 CNN + 目标条件注意力编码器接 MLP actor-critic，输出 3 维 SE(2) 速度指令。
- 入口：无 CLI；Python 调用 `LatentGlobalMapModel`、`LatentMapConfig`、`load_checkpoint`、`build_policy`，方法 `reset_inference`、`forward_step`、`act_inference`。
- 安装：Python 3.10+，`pip install torch`，`pip install -e .`。
- 状态：README 称训练与机器人部署代码计划后续发布；检查点说明前后不一致，未核实目录内文件。
- **结论：部分开源**。
