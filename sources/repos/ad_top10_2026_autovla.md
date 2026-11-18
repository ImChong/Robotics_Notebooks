# AutoVLA 官方源码及复现入口

- **官方仓库：** https://github.com/ucla-mobility/AutoVLA
- **原论文：** https://arxiv.org/abs/2506.13757
- **核对日期：** 2026-11-18
- **核对范围：** 官方 README 与下列可用源码/文档入口，静态核验而非真实训练执行
- **通读：** 已完成

## 官方执行入口

1. ```bash scripts/action_token_cluster.sh```
2. ```bash scripts/run_sft.sh```
3. ```bash scripts/run_rft.sh```
4. ```bash navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh```

## 已知边界

动作token的离散误差及额外CoT的延时需单独评估；离线规划指标不是实车安全证明。

- [独立节点](../../wiki/entities/paper-rcl-2506-13757-autovla-a-vision-language-action-model-for-end-t.md)
- [原文记录](../papers/ad_top10_2026_autovla.md)
