# Terrain-Grounded Imagination（TGI）

> 来源归档

- **类型：** repo（ETH Zürich 作者官方实现）
- **链接：** <https://github.com/leggedrobotics/terrain-grounded-imagination>
- **作者：** Matthias Heyrman、Marco Hutter
- **核查版本：** [a64911f](https://github.com/leggedrobotics/terrain-grounded-imagination/tree/a64911fea91848189f4863af8ae004c2e376836c)
- **许可证：** [MIT](https://github.com/leggedrobotics/terrain-grounded-imagination/blob/a64911fea91848189f4863af8ae004c2e376836c/LICENSE)；第三方机器人模型另有许可，ANYmal-D 与 ANYdrive 仿真资源从 Isaac Lab 获取。
- **核查日期：** 2026-11-27
- **沉淀到 wiki：** [Terrain-Grounded Imagination](../../wiki/entities/terrain-grounded-imagination.md)
- **通读：** 已完成
- **核对范围：** 通读该提交官方 README；交叉核对 ANYmal-D 的训练配置、采集与训练循环、地形距离查询、接触动力学网络、奖励与终止、评测入口。此条目归档官方项目，不把 BibTeX 中的仓库引用当作独立论文。

## 原始资料与口径

- [ANYmal-D 训练入口](https://github.com/leggedrobotics/terrain-grounded-imagination/blob/a64911fea91848189f4863af8ae004c2e376836c/tgi_anymal/scripts/train.py)与[按地形等级评测脚本](https://github.com/leggedrobotics/terrain-grounded-imagination/blob/a64911fea91848189f4863af8ae004c2e376836c/tgi_anymal/scripts/anymal_term_level.py)分别负责训练与检查点评估；G1 使用独立的 `tgi/` 训练入口。
- 仓库列出两种默认训练配方及运行耗时；未给出可重算 ANYmal-D「约 6 倍」的 PPO 训练曲线、收敛阈值与比较单位。该数字不是 README 中 5.6 小时与另一公开数字的比值。
- README 演示描述包含 MuJoCo 中的 ANYmal-D；仓库提供的 ANYmal-D 训练及上述评测则用 Isaac Lab / Newton。这不能当作真机验证。本次未重训或复现对照。
