# iPPLab/HOG-WAM 官方实现

- **类型：** 代码仓库
- **代码：** [iPPLab/HOG-WAM](https://github.com/iPPLab/HOG-WAM)
- **论文：** [arXiv:2611.34811](https://arxiv.org/abs/2611.34811)（[论文归档](../papers/hog-wam-arxiv-2611-34811.md)）
- **核对日期：** 2026-11-27

核对 README、`src/fastwam_tpu/models/mot.py`、`src/fastwam_tpu/inference/policy.py` 与 `eval/scripts/fastwam_model_server.py`。仓库发布 HOG 训练、导出和评估代码；训练完成的策略检查点、数据集和仿真资源需单独准备。主方法的 RGB 输入和名为 `rgb_to_hog` 的实验消融是不同配置。未运行训练或仿真。

对应 [HOG-WAM 节点](../../wiki/entities/paper-hog-wam.md)。
