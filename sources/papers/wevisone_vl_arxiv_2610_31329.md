# WeVisOne-VL: Unified Visual Perception from Objects to Pixels

- **类型：** 技术报告，arXiv 预印本
- **机构：** 腾讯微信视觉团队（WeChat Vision, Tencent）
- **作者：** Tianming Liang、Junyi Pan、Yuxi Li、Canmiao Fu、Zhipeng Huang、Chong Sun、Chen Li、Jing Lyu
- **版本：** [arXiv v2，2026-10-24](https://arxiv.org/abs/2610.31329v2) · [PDF](https://arxiv.org/pdf/2610.31329v2) · [HTML](https://arxiv.org/html/2610.31329v2)
- **官方代码：** <https://github.com/Tencent/WeVisOne-VL>
- **入库日期：** 2026-10-27；**复核日期：** 2026-11-27
- **通读：** 已完成
- **核对范围：** v2 正文 §§1–9、方法与数据构建、训练课程、实验表 2–8 及附录的训练数据、超参和奖励说明；未跑模型推理、训练或论文基准。

## 方法与数字的定位

- §2：WDE 标注库涉及逾 1 亿图像与约 36 亿框标注，含难正负样本和 alpha 数据构造，不是公开下载量。
- §3：结构化自回归输出结合约 9.6M 参数像素解码器；分割和抠图共享多数组件，后者用额外精细化恢复边界和分数透明度。
- §4：三阶段 SFT 分别对齐任务格式、提升分辨率/密集度、联合 token 与像素训练。混合任务 GRPO 用结构化定位奖励，不对目标掩码/alpha 像素直接计算 RL 奖励。
- §§5.1.1–5.1.7：报告 4B 的 LVIS F1 Mean 55.3、Dense200 71.3（表 2）；WeRef-Bench F1 Mean 78.7、缺失目标拒答 96.4%，RefCOCOg-Rej 拒答 67.4%（表 3）；ReasonSeg val gIoU 71.2（表 7）；WeRefMat 指代提示 MAD×10³ 为 9.8（表 8）。不同任务和量纲不可直接合并。

## 对 wiki 的映射

- [WeVisOne-VL](../../wiki/entities/paper-wevisone-vl.md)
- [官方仓库归档](../repos/wevisone-vl.md)
