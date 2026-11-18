# ReCogDrive 官方源码及复现入口

- **官方仓库：** https://github.com/xiaomi-research/recogdrive
- **原论文：** https://arxiv.org/abs/2506.08052
- **核对日期：** 2026-11-18
- **核对范围：** 官方 README 与下列可用源码/文档入口，静态核验而非真实训练执行
- **通读：** 已完成

## 官方执行入口

1. ```cd internvl_chat && sh ./shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh```
2. ```sh scripts/training/run_recogdrive_train_multi_node_2b.sh```
3. ```sh scripts/training/run_recogdrive_train_multi_node_rl_2b.sh```
4. ```sh scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh```

## 已知边界

部分第三方数据受许可限制；VLM隐状态缓存约占1–2TB磁盘；不同数据集分数不可混合。

- [独立节点](../../wiki/entities/paper-recogdrive-2506-08052.md)
- [原文记录](../papers/ad_top10_2026_recogdrive.md)
