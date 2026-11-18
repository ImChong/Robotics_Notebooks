# DriveWAM 官方源码及复现入口

- **官方仓库：** https://github.com/chenshi3/DriveWAM
- **原论文：** https://arxiv.org/abs/2605.28544
- **核对日期：** 2026-11-18
- **核对范围：** 官方 README 与下列可用源码/文档入口，静态核验而非真实训练执行
- **通读：** 已完成

## 官方执行入口

1. ```python -m src.navsim.process_data --output-path ./data/navsim/trainval```
2. ```bash scripts/drivewam_navsim_train.sh```
3. ```python -m src.navsim.eval```

## 已知边界

生成画面不是物理精确的交通仿真；KV记忆预算与自回归累积误差会限制长期性能。

- [独立节点](../../wiki/entities/paper-rcl-2605-28544-drivewam-video-generative-priors-enable-scalable.md)
- [原文记录](../papers/ad_top10_2026_drivewam.md)
