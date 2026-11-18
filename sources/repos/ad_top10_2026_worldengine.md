# WorldEngine 官方源码及复现入口

- **官方仓库：** https://github.com/OpenDriveLab/WorldEngine
- **原论文：** https://arxiv.org/abs/2606.19836
- **核对日期：** 2026-11-18
- **核对范围：** 官方 README 与下列可用源码/文档入口，静态核验而非真实训练执行
- **通读：** 已完成

## 官方执行入口

1. ```export WORLDENGINE_ROOT=$(pwd)```
2. ```bash scripts/closed_loop_test.sh```
3. ```bash scripts/multigpu_closed_loop_test.sh```

## 已知边界

官方README把Behavior World Model integration仍列为待办；simengine和algengine需分开装依赖；3DGS重建画面不保证物理交互真实。

- [独立节点](../../wiki/entities/paper-worldengine-2606-19836.md)
- [原文记录](../papers/ad_top10_2026_worldengine.md)
