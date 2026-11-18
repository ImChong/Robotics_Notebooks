# SGDrive 官方源码及复现入口

- **官方仓库：** https://github.com/LogosIntelligenceLab/SGDrive
- **原论文：** https://arxiv.org/abs/2601.05640
- **核对日期：** 2026-11-18
- **核对范围：** 官方 README 与下列可用源码/文档入口，静态核验而非真实训练执行
- **通读：** 已完成

## 官方执行入口

1. ```sh scripts/training/run_sgdrive_train_multi_node_2b.sh```
2. ```sh scripts/training/run_sgdrive_train_multi_node_2b_rl.sh```
3. ```sh scripts/evaluation/run_sgdrive_agent_pdm_score_evaluation.sh```

## 已知边界

部分权重仍有coming soon标记，需核对HF可得资产；训练缓存需求可达1–2TB；无法由PDMS推导真实道路安全。

- [独立节点](../../wiki/entities/paper-sgdrive-2601-05640.md)
- [原文记录](../papers/ad_top10_2026_sgdrive.md)
