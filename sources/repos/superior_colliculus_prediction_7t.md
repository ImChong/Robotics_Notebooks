# prediction-in-Superior-Colliculus-7T

- **类型：** repo
- **代码：** https://github.com/danlei-chen/prediction-in-Superior-Colliculus-7T
- **论文：** https://doi.org/10.1038/s41593-026-02480-0
- **数据：** https://openneuro.org/datasets/ds008447/versions/1.1.0
- **入库日期：** 2026-11-02
- **开放状态：** 公开分析代码和 ROI 掩膜；论文提供原始与预处理数据。

## 已核对入口

- `GLM/level1_model.py` 建立 Nipype 一级模型，读取 BOLD、任务事件和混杂变量，输出对比图。
- `GLM/level1_warping.py` 使用 DARTEL 将图像配准到上丘相关空间。
- `signal_comparison/signal_extration.py` 按 ROI 体素提取三个任务阶段的信号。
- `MVPA/MVPA_analysis.py` 读取预先准备的阶段事件 CSV 和上丘图像数组，使用线性 SVM 分类并实施受试者标签置换。

README 未提供完整逐步复现命令；脚本包含固定数据路径，MVPA 所需中间 CSV 和 NPY 的完整生成流程尚待核实，不应视为开箱即用。

## 对 wiki 的映射

- [人类上丘的感觉预测与反应](../../wiki/entities/paper-human-superior-colliculus-predictive-7t.md)
