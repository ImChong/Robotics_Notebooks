# BiCAT 官方 tokenizer / BAR Policy 源码

- **类型：** repo / VLA / bimanual
- **代码：** https://github.com/MAGICLAB-NUS/BiCAT
- **论文：** https://arxiv.org/abs/2611.23499
- **项目：** https://magiclab-nus.github.io/BiCAT/
- **核查日期：** 2026-11-20
- **核对范围：** README 全文与 src/bicat/scripts/train.py 训练入口、官方 CLI/serve/eval 文档
- **通读：** 已完成

## 原文记录

官方 README 中 codec `bicat-train --config-name train_codecs codecs@codec=tokens/bicat dataset=<name>`，策略 `bicat-train policy=bar_bicat_pali dataset=<name>`；入口 `src/bicat/scripts/train.py` 在 Hydra 配置下选择 `CodecsTrainer` / `PolicyTrainer`。数据要求 LeRobot v3，双臂动作按 [左臂+夹爪, 右臂+夹爪] 排列；右臂 `right_arm_sign` 与共享范围必须与机器人匹配。部署 `bicat-serve` gRPC，再用 `scripts/robotwin/run_eval.py` 在指定 checkpoint 评测。**注意：** `run_mode=overwrite` 会清空输出目录；策略 checkpoint 不内含冻结 tokenizer 权重。License Apache-2.0。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/paper-bicat.md)
