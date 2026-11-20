# HINT-WAM 官方训练代码

- **类型：** repo / robot learning
- **代码：** https://github.com/tuandattt/hintwam_code
- **论文：** https://arxiv.org/abs/2611.24789
- **项目：** https://tuandattt.github.io/hintwam/
- **评测依赖：** https://arxiv.org/abs/2603.16666
- **核查日期：** 2026-11-20
- **核对范围：** 官方 README 全文，训练/数据/权重/配置/ablation 与 CLI；代码目录映射来自 README，未逐模块源码审计
- **通读：** 已完成

## 原文记录

README 明确仓库**仅提供训练**；闭环 LIBERO/RoboTwin 评测使用 Fast-WAM 代码。训练入口 `scripts/train_zero2.sh` → `scripts/train.py`，配置 `task=libero90_sft`、`libero90_opsd_teacher`、`libero90_opsd_distill_energy`，teacher checkpoint 通过 `opsd.teacher_ckpt` 传入；教师带额外 hindsight adapter，学生 Stage B 训练 action expert；每轮 10 sampler steps，4 student samples，BC anchor 权重 1。README 环境 8× A5000、Python 3.10、CUDA 12.8。训练日志需检查 teacher loaded 与 phi_gate，避免加载错误教师。**源码逐函数运行尚未复测**。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/paper-hint-wam.md)
