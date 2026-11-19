# RDT-1B 项目页（rdt-robotics.github.io）

- **URL：** <https://rdt-robotics.github.io/rdt-robotics/>
- **论文：** [rdt_1b_arxiv_2410_07864.md](../papers/rdt_1b_arxiv_2410_07864.md)
- **代码：** <https://github.com/thu-ml/RoboticsDiffusionTransformer> — [`sources/repos/thu_ml_robotics_diffusion_transformer.md`](../repos/thu_ml_robotics_diffusion_transformer.md)
- **入库日期：** 2026-09-28
- **最近复核：** 2026-11-19
- **核对范围：** 项目页完整静态文本（`rdt_site.txt`）

## 一句话说明

RDT-1B 官方项目页：展示双臂操作架构总览、6 大真机泛化与灵巧性视频对比，以及代码/权重/微调数据下载入口。

## 项目页要点与开源核查（2026-11-19）

- **开源与发布状态：** 页眉提供 Paper（arXiv）、Code（GitHub `thu-ml/RoboticsDiffusionTransformer`）、Data（HF `rdt-ft-data`）、Model（HF `rdt-1b`）全套链接，均**已开源 / 已发布**。
- **展示维度：**
  - **Dexterity：** 摇杆直行遥控机器狗（对比 ACT 偏航、OpenVLA 抓空、RDT Scratch 走歪）；
  - **Zero-Shot 对象与场景：** 未见杯具与未见房间；
  - **Instruction Following：** 未见水深词汇（`left/right hand`, `1/3`, `2/3` 量杯倒水，对比 OpenVLA 与 Octo）；
  - **Few-Shot Learning：** 1-shot 叠短裤与 5-shot 双臂交接马克笔；
  - **Real-Time Inference：** 声明移除限速后可达 6 action chunks/sec（381 actions/sec），支持接近人类操作速度的双臂执行。
