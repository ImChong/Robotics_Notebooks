# Cortical Labs CL SDK（CL1 官方开发与模拟接口）

> 官方代码 / 开发文档来源归档；作为 2025 年 CL1 论文的**后续补充材料**。此处的 API 说明来自目前可访问的官方文档，不倒推为 2025 年论文原文内容。

- **类型：** repo / SDK / simulation / neural-interface
- **机构：** Cortical Labs
- **代码：** <https://github.com/Cortical-Labs/cl-sdk>
- **开发文档：** <https://docs.corticallabs.com/>
- **教程仓库：** <https://github.com/Cortical-Labs/cl-api-doc>
- **平台页：** <https://corticallabs.com/cl1.html>
- **入库日期：** 2026-11-19
- **核对范围：** 已阅读 cl-sdk 官方 README、开发文档 Quick start / Detecting and Reacting to Spikes / Stimulating / Simulator，以及 src/cl/__init__.py、src/cl/neurons.py 的部分实现；未运行模拟器、未接入 CL1 真机，未通读仓库所有模块。
- **通读：** 未完成
- **一句话说明：** Python SDK 为本地应用提供与 CL1 设备上 CL API 兼容的接口，可用模拟数据或历史数据回放验证闭环应用；模拟数据不等同真实神经元学习。

## 入口与运行路径（官方文档与代码）

- Python 3.12+；`pip install cl-sdk`；推荐通过 `with cl.open() as neurons:` 开启会话。
- `neurons.loop(ticks_per_second=...)` 逐 tick 提供 `tick.analysis.spikes`；应用逻辑可以据此构造 `StimDesign`，调用 `neurons.stim(channel, design)`。
- `neurons.record()` 可记录 spike、刺激与其他事件；开发文档给出 HDF5 存储格式。
- 模拟器默认生成伪随机数据，也能通过 `CL_SDK_REPLAY_PATH` 回放记录；官方明确模拟器**不提供响应刺激的真实学习动力学**，不可用其数据证实生物学习效果。
- `src/cl/neurons.py` 提供模拟端 `Neurons` 的 loop、stim 等接口，并关联数据生产器与刺激队列；真机上采用预装的设备版 CL API，不能把模拟端实现当作硬件驱动全部实现。

## 与论文的关系

- [CL1 平台论文来源](../papers/kagan_2025_cl1_biological_neural_platform.md)；论文发表于 2025-07-07，当前官方开发文档引用 2026 年 CL API（arXiv:2602.11632），版本与时间均有区别。
- [独立论文详情页](../../wiki/entities/paper-cl1-biological-neural-platform-kagan-2025.md)
