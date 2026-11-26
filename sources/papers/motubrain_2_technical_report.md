# Motubrain 2.0：数据对齐与规模化的智能体世界动作模型 | An Agentic World Action Model with Aligned and Scaled Data

> 来源归档 / 官方技术报告。只保存出处与核验摘要，不转存 PDF。

- **原文：** [官方技术报告 PDF](https://www.motubrain.cn/_cms/assets/5584f6c8-19d4-4901-9836-d63f2fd28769_38184155-9315-4ba6-98fc-91aa37d1fd3f/assets/motubrain-2-tech-report.pdf?v=98b12305)（91 页，站点文件版本 `98b12305`）
- **作者 / 机构：** Motubrain Team / 生数科技（报告署名 GensPI Technology）
- **发布日期：** 官方发布博客标注 2026-11-24；PDF 未标 arXiv 号或版本号。
- **官方入口：** [项目页归档](../sites/motubrain-2-technology.md) · [发布博客归档](../blogs/motubrain-2-release.md) · [代码入口核验](../repos/motubrain-2-0.md) · [模型卡核验](../models/motubrain-2-robodojo.md)
- **核对范围：** 以官网 91 页 PDF 为主，核对 §§1–10、Canonical55 附录 A、数据质量附录 B、评测设置与逐任务表附录 C、真机配对与版本耗时附录 D、数据分布附录 E；对照官方博客与 2.0 项目页。未将视频演示当作独立实验。
- **通读：** 已完成

## 原文核验

| 锚点 | 记录 |
|---|---|
| §3，表 1–4 / 附录 A、B | 34,656,294 段注册具身记录、499,773.45 小时；按主视角计时，不按摄像头叠加。另约 200 万小时是视频基础模型的预训练规模。缺失与坏标签按窗口和通道屏蔽。 |
| §§4–5，图 6、表 7–9 | 20B 四流网络，前 6 层动作自注意，第 7–24 层可读取图像与文本；冻结 DA3 的 KV 注入指定层。分别采样视频和动作噪声时刻。 |
| §§6–7，图 10–18 | 分开记录少样本后训练、冻结模型的参考视频 ICL、四步蒸馏，以及固定四步 Euler 的本地推理加速消融；四者不共用一个数字。 |
| §8，表 14、26 | RoboTwin 2.0 的 DMD 为 91.42% / 83.16%（C2C / C2R）；人类 Ego 数据消融 52.2% → 73.4% 是四任务等权、目标本体暴露量近似匹配，非总训练步数匹配。 |
| §9，表 29–32 / 附录 D | seen-command 桌面任务 WAM 单独、无 WAM 的 Agent、完整 Harness 分别 7/32、25/32、29/32。十任务 V1→V6 为 31/71→45/71，同时中位耗时 144→487 s。 |

报告的 ICL 小节涉及独立论文 [Entangle（arXiv:2611.28732）](https://arxiv.org/abs/2611.28732)，不是 2.0 总报告的 arXiv 身份；前代 Motubrain 的 arXiv:2604.27792 也是不同工作。

## 对 Wiki 的映射

- [Motubrain 2.0 技术报告](../../wiki/entities/paper-motubrain-2.md)
