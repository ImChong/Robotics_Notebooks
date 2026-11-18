# xmimic-xsoma-2611-20452 官方代码

- **类型：** repo
- **地址：** https://github.com/nico-bohlinger/xsoma
- **论文：** https://arxiv.org/abs/2611.20452
- **源码状态：** 已开源；仓库含真实运行入口。
- **核对范围：** 官方 README、相关训练/CLI 入口与配置说明（静态核对，不含实际 GPU 运行）。
- **通读：** 已完成
- **wiki：** [论文详情](../../wiki/entities/paper-xmimic-xsoma-2611-20452.md)

## 入口与证据

XSoma 主仓：python train.py --layout AL --log_dir runs/x4 → xsoma.train.main（xsoma/train.py）；训练包括仿真采样、回放、世界模型更新、可微想象及策略更新。配套 XMimic 在 https://github.com/nico-bohlinger/xmimic，train.py 选择 single_gpu.run 或 multi_gpu.run；旧 standalone XMimic 代码归档 sources/repos/xmimic-jax.md 不重复建立实体。

## 复现边界

训练时间和实机成绩来自作者论文与官方结果，不宣称本次实测复现。源码可能更新，使用前固定 commit、软件版本与数据许可。
