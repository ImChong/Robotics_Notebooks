# Real2SimLab 官方仓库

- **类型：** repo
- **链接：** <https://github.com/Real2SimHub/Real2SimLab>
- **项目页：** <https://real2simhub.github.io/Real2SimLab/>
- **论文：** <https://arxiv.org/abs/2611.34849v1>
- **许可：** Apache-2.0。
- **核对范围：** README、实验入口与编排、冻结重跑检查；源码阅读，不声明运行结果。

- [实验 CLI](https://github.com/Real2SimHub/Real2SimLab/blob/1c36025f79f9f071fc419adb37166a8aefc3efdf/real2simlab/cli/experiments.py)：`r2s run` 解析 TOML，调用实验运行。
- [实验编排](https://github.com/Real2SimHub/Real2SimLab/blob/1c36025f79f9f071fc419adb37166a8aefc3efdf/real2simlab/orchestrate/experiment.py)：解析任务、调用方法适配器、核对产物指纹并组织评分。
- [常规冻结重跑](https://github.com/Real2SimHub/Real2SimLab/blob/1c36025f79f9f071fc419adb37166a8aefc3efdf/real2simlab/runner/run.py)：只读程序、传入 seed、检查约定输出并写运行记录。
- [交付检查](https://github.com/Real2SimHub/Real2SimLab/blob/1c36025f79f9f071fc419adb37166a8aefc3efdf/real2simlab/runner/check.py)：对可执行交付按同 seed 重跑两次并比较输出；基准自带评估器执行的包不走该重跑路径。

[论文归档](../papers/real2simlab_arxiv_2611_34849.md) · [官方文档归档](../sites/real2simlab.md)
