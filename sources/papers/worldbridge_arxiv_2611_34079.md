# WorldBridge: Bridging Scattered Views for Coherent 3D World Generation

- **类型：** arXiv 论文（v1，2026-11-25）
- **作者：** Weilong Yan、Jingyu Hu、Hongli Xu、Yifan Zhan、Beibei Lin、Anbang Wang、Qinqian Lei、Jingwei Huang、Robby T. Tan
- **机构：** 新加坡国立大学、腾讯、香港中文大学、东京大学
- **原文：** [arXiv 2611.34079v1](https://arxiv.org/abs/2611.34079v1) · [HTML（含补充材料）](https://arxiv.org/html/2611.34079v1) · [PDF](https://arxiv.org/pdf/2611.34079v1)
- **项目页：** [作者公开演示](https://davidyan2001.github.io/WorldBridge/) · [项目核验](../sites/worldbridge-scattered-views.md)
- **通读：** 已完成
- **核对范围：** v1 正文 §1–6；补充材料 A–E 的训练、采样、LRC、基准构建、指标、原生深度额外评估与失败案例；对照项目页。

## 原文证据索引

- **§3，图 2–3，式 1–8：** 位姿已知的参考图与待生成相机是输入；场景读出和逐图潜变量各保留一条条件路径。只有双方均有序列位置的注意力才启用时间 RoPE，所有注意力对保留空间位置与相机编码。每个目标视角输出 RGB、深度与基于相机射线的点图。
- **§3.5，附录 A.3–A.4：** MoGe-3 估计参考深度，遮挡处理后将颜色重投影到目标；有支持的像素监督当前干净潜变量，每步更新一次并重新加噪。41 帧、H200、50 采样步、CFG 2 的报告配置下，LRC 相对不启用耗时增加 18.3%。
- **表 2–5、附录 B–C：** Short 总体 PSNR 18.35 dB，对照 GLD 16.13 dB；Long 17.68 对比 16.43 dB，AbsRel 0.345 对比 0.454（与 DA3/ViPE 伪深度、每场景尺度对齐）。Agent 同一路线分 On/Off，当前没有真实目标 RGB 或真实几何深度进行对照。
- **附录 D–E：** 补充原生深度在 ScanNet v2 和 Oxford Spires 传感器选定样例上另测，仍有每场景尺度对齐。椅框、墙缘等细结构会断裂，空间参考与目标低重叠时周边细节可漂移；补出的区域不宜当实测。正文与附录的数据记录分辨率不一致（§4.2 写 672×378、A.1 写 1280×720），因此不将二者混写为同一项训练输入规格。

- **沉淀到 wiki：** [WorldBridge 三维世界生成](../../wiki/entities/paper-worldbridge-scattered-views.md)
