# Figure Helix 官方技术发布

- **类型：** 官方技术博客
- **Helix：** <https://www.figure.ai/news/helix>（2025-02-20）
- **Helix 02：** <https://www.figure.ai/news/helix-02>（2026-01-27）
- **核查日期：** 2026-10-05（2026-11-21 全文复核）
- **核对范围：** 抓取并通读 Figure 官方技术发布完整正文：<https://www.figure.ai/news/helix>（2025-02-20，Introducing Helix 至 Conclusion，含 7B S2 @ 7–9 Hz、80M S1 @ 200 Hz 35-DoF、~500 小时遥操数据、双嵌入式 GPU 流式推理）与 <https://www.figure.ai/news/helix-02>（2026-01-27，System 0 10M 参数 1 kHz 与 1,000+ 小时 / 200,000+ 并行环境、System 1 200 Hz 掌部相机与 3 g 触觉、System 2 语义规划、4 分钟 61 动作洗碗机任务与四项灵巧任务）；两篇均未提供模型权重、训练数据或可运行源码下载入口。
- **通读：** 已完成
- **一句话说明：** 对照两代视觉语言动作系统的输入、层级接口与全身控制边界。

## 技术摘录

1. **Helix 初代：** System 2 处理场景和语言，通过 latent 条件化 System 1；后者把视觉与本体状态转成上半身动作，强调陌生物体操作和双机器人协作。
2. **Helix 02：** System 1 扩展到全身关节目标（200 Hz），System 0 执行平衡与接触协调（1 kHz）。System 0 为约 10M 参数网络，训练使用超过 1000 小时重定向人动作与大规模并行仿真。
3. **硬件与评测：** Helix 02 使用 Figure 03 的掌部相机和触觉；展示约四分钟、61 个动作的厨房连续任务及灵巧操作。演示长度不能代替重复试验成功率或陌生家庭泛化评测。

## 开源核查

打开两篇官方发布页后，未见 Helix / Helix 02 训练代码、模型权重或训练数据下载入口。可用于机制对照，不能作为可本地复现的模型发布；**源码运行时序图不适用**。

## 对 wiki 的映射

- [Helix 初代](../../wiki/entities/paper-rcl-ref-cb61c489d1333f433fc4-helix-a-vision-language-action-model-for-general.md)
- [Helix 02](../../wiki/entities/helix-02.md)
- [Figure 公司概述](../../wiki/entities/figure-ai.md)
