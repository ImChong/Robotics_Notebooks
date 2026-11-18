# Alpamayo 1 官方源码及复现入口

- **官方仓库：** https://github.com/NVlabs/alpamayo
- **原论文：** https://arxiv.org/abs/2511.00088
- **核对日期：** 2026-11-18
- **核对范围：** 官方 README 与下列可用源码/文档入口，静态核验而非真实训练执行
- **通读：** 已完成

## 官方执行入口

1. ```python src/alpamayo_r1/test_inference.py```

## 已知边界

官方仓已转为有限维护；HF权重和数据需单独申请；推理要求GPU显存至少24GB；公共10B不接收显式导航路线输入。

- [独立节点](../../wiki/entities/paper-alpamayo-r1-2511-00088.md)
- [原文记录](../papers/ad_top10_2026_alpamayo-r1.md)
