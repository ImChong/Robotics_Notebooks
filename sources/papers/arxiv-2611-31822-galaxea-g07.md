# G0.7：长上下文快速世界动作模型 | Scaling Up a Fast World Action Model with Long Context

- **原文：** [arXiv:2611.31822](https://arxiv.org/abs/2611.31822) · [v2 HTML](https://arxiv.org/html/2611.31822v2) · [v2 PDF](https://arxiv.org/pdf/2611.31822v2)
- **机构：** 星海图（Galaxea Team）
- **日期：** v1 于 2026-11-24 提交；v2 于 2026-11-25 修订。v2 补入项目地址及贡献名单；报告正文机制和实验指标未见变化。
- **项目页：** [英文报告](https://www.galaxea-ai.com/en/model-g07.html) · [中文版](https://www.galaxea-ai.com/zh/model-g07.html) · [发布核验](../sites/galaxea-g07.md)
- **核对范围：** 通读 arXiv v1 与 v2 共 13 页的正文 §§1–6、图 1–13、贡献与参考文献；对照官方中英文技术页，发布状态另列。未运行模型或机器人。
- **通读：** 已完成

## 原文证据定位

| 位置 | 核对结果 |
|---|---|
| §2.1、图 1–2 | 因果 VAE 编码真实观测，干净历史 latent 写入滑窗 KV；可前置人类演示，预训练改变序列长度及位置偏移。 |
| §2.2、图 3 | 同 checkpoint 的 Fast / IDM 分别跳过 / 先生成当前 chunk 未来视频 latent；RTX 5090 上 BF16 eager 为 156 / 457 ms，部署 FP8 Fast 为 60 ms。 |
| §3、图 4–6 | R1 Pro/Lite 与公开遥操作轨迹，加头戴第一视角视频；人手采用对齐 R1 TCP 的腕位姿与五指端点。 |
| §4、图 7–13 | 仿真综合分、真机零样本取放、经训练的 ICL 两任务，以及真机后训练阶段完成度分别报告；指标不能互换。 |
| §§5–6 | 与 Fast-WAM、Long-WAM、RTC 和第一视角训练路线的关系；训练视觉监督与部署生成依赖分开。 |

## 知识层

[阅读 G0.7 机制与评测笔记](../../wiki/entities/paper-galaxea-g07.md)
