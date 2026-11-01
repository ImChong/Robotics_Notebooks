# SSIO 官方仓库

- **代码：** <https://github.com/amazon-far/SSIO>
- **项目页：** <https://ssio-imu.github.io/>；[项目归档](../sites/ssio.md)
- **论文：** <https://arxiv.org/abs/2610.25988>；[论文归档](../papers/ssio_arxiv_2610_25988.md)
- **权重：** <https://huggingface.co/Open-IO/SSIO>
- **核查日期：** 2026-10-29

已公开积分自监督、位姿图蒸馏、评估与六个数据集加载器；README 提供 EuRoC 小样本 CPU 流程：`train_ssio.py` → 可选 `train_pgd.py` → `evaluate.py`。G1 与转换后的 MIPO 数据公开；MIPO rosbag 转换器未包含在仓库中。已发布 PGD checkpoint 与论文对应行的结果不完全一致，需参照官方 recipes。
