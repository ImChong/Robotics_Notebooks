# ActionCodec2: A Streaming Action Tokenizer｜流式动作分词器

- **类型：** 论文
- **作者：** Zibin Dong、Yicheng Liu、Shiduo Zhang、Tianyuan Yuan、Baijun Ye、Jinqiu Wang、Hang Zhao
- **原文：** https://arxiv.org/abs/2611.15490 （v1）
- **HTML：** https://arxiv.org/html/2611.15490
- **PDF：** https://arxiv.org/pdf/2611.15490
- **项目：** https://zibindong.github.io/actioncodec2-page/
- **代码：** https://github.com/ZibinDong/actioncodec2
- **核对范围：** arXiv:2611.15490v1 正文的动作编码动机、量化/Set-BPE 方法、流式执行、仿真及 DROID 真机实验、局限与相关附录；交叉核读官方 README 与实际训练/评测/推理入口。未进行模型训练和真机复现。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [ActionCodec2](../../wiki/entities/paper-actioncodec2-streaming-action-tokenizer.md)

## 原文摘录与映射

1. **误差反馈量化：** 量化参考的是解码器已重建的前一步动作状态，由下一步更新抵消累积量化偏差。误差约束作用于重建轨迹而非真机伺服跟踪。映射：方法与误读边界。
2. **Set-BPE：** 跨动作分量和时间步合并离散符号，在较少 token 下继续支持逐步释放动作。映射：流程图与 tokenization 选型。
3. **Streaming：** 流式动作 token 允许边解码边执行，但动态重规划仍需要实时观测、策略训练及执行接口配合。映射：工程实践和闭环链路。
4. **评测：** 相同 G0.5 骨干在 DROID 训练后的四组未见真机任务，每策略合计 50 次评估，ACodec2+ 平均任务进度 73.9%，FAST 50.8%、X-Tokenizer 52.8%；**进度是里程碑比例，不是成功率**。映射：实验口径。
5. **代码边界：** 官方 tokenizer、G0.5 训练与 WebSocket 推理 benchmark 代码可得；主要代码 Apache-2.0，G0.5 适配及派生策略遵循单独非商用许可。映射：源码归档和运行时序图。
