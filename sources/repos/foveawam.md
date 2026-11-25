# FoveaWAM 与 OnTimeBench | Official code release

- **类型：** repo
- **代码：** [FoveaWAM 固定提交 9f52a34](https://github.com/ArrebolBlack/FoveaWAM/tree/9f52a3454f9d1ddfb6a54c662941a0e747262ffe)、[OnTimeBench 固定提交 cf57767](https://github.com/ArrebolBlack/OnTimeBench/tree/cf57767e9435115c07aa3b071677e667aca2db8d)
- **项目页：** <https://foveawam.github.io/>
- **论文：** <https://arxiv.org/abs/2611.31705v1>
- **许可：** 两仓代码 Apache-2.0；OnTimeBench 数据 CC BY 4.0。仿真权重作者参数仅供非商业研究和评估，内含 MAE 编码器另受 CC BY-NC 4.0 约束，不继承代码许可。
- **发布物核对：** Hub 清单有 [仿真 checkpoint（固定版本 06e5a81）](https://huggingface.co/JiaqiYin/FoveaWAM/tree/06e5a813996e5a08acd024fa183ca24565c410d3) 与 [OnTimeBench 的 Harmonic grasp 样本（固定版本 f6d60b8）](https://huggingface.co/datasets/JiaqiYin/OnTimeBench/tree/f6d60b879c263a28b65b45fa2063acec25f6e674)；只核对文件清单，未下载大文件。
- **核对范围：** README、许可、CLI、policy、solver、评测适配与基准接口；未执行 GPU 推理或完整仿真。
- **通读：** 已完成
- **入库日期：** 2026-11-25
- **对应节点：** [FoveaWAM](../../wiki/entities/paper-foveawam.md)

| 入口 | 核对结果 |
|---|---|
| `foveawam.cli.infer` | `load_policy` 加载仿真 bundle；HDF5 读八帧并 reset，逐帧送入历史，最后一帧触发动作采样，输出 `actions_e7` |
| `inference/policy.py` 与 `inference/solver.py` | 观察窗口 → 一次世界流条件计算 → 五步 Euler → 反归一化；动作 API 不跑预测头，`predict_motion` 可显式请求世界读出 |
| `foveawam.cli.evaluate` 与 OnTimeBench adapter | 使用内置时延分布或指定设备测得的分布；仿真目标位姿为可配置的观测来源 |
| OnTimeBench 生成及录制入口 | 可运行内置 oracle 采集单场示教，不依赖已发布模型权重；完整分数仍须加载固定拆分和对应策略 |

[论文归档](../papers/foveawam-arxiv-2611-31705.md) · [项目页归档](../sites/foveawam.md)
