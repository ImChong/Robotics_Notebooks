# DriveVLA-W0 官方源码及复现入口

- **官方仓库：** https://github.com/BraveGroup/DriveVLA-W0
- **原论文：** https://arxiv.org/abs/2510.12796
- **核对日期：** 2026-11-18
- **核对范围：** 官方 README 与下列可用源码/文档入口，静态核验而非真实训练执行
- **通读：** 已完成

## 官方执行入口

1. ```bash scripts/misc/download_emu3_pretrain.sh```
2. ```bash inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh```
3. ```bash inference/vla/eval_navsim_metric_from_json.sh```

## 已知边界

世界模型在训练期辅助学习，不能把视觉生成指标直接当成驾驶规划安全指标。

- [独立节点](../../wiki/entities/paper-sa-2510-12796-drivevla-w0-world-models-amplify-data-scaling-la.md)
- [原文记录](../papers/ad_top10_2026_drivevla-w0.md)
