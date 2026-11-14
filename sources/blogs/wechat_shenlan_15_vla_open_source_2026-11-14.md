# 15 个开源项目，看透 2026 年 VLA 的三个关键转变｜来源导读

- **资料类型：** blog / 综述（非 15 篇论文原文）
- **中文标题：** 15 个开源项目，看透 2026 年 VLA 的三个关键转变
- **作者与发布方：** 深蓝学院·具身君 / 深蓝具身智能
- **发布时间：** 2026-11-14 18:08（北京时间，附件页首）
- **原文：** https://mp.weixin.qq.com/s/Jgh3OTMK8gcsQYqX_fOp-Q
- **附件：** 用户提供的同题 25 页 PDF（2026-11-14 截图打印版）
- **核对范围：** 附件第 1–25 页全文（第 1–23 页正文；第 24 页作者列出的 7 个 GitHub 引用；第 25 页结束页），并对照本仓库已录入节点及下述官方仓库 README。
- **通读：** 已完成
- **重要声明：** 本文件仅做原创提炼及来源定位，不转载公众号文章、配图；项目所属机构、数据、开源状态和 GitHub Star 随时间变化，须以官方一手资料为准。

## 三条线索（文章观点，不是行业定论）

1. **机器人本体与数据：** 从桌面操作扩展到移动操作、全身控制和多本体共享表示；代表为 GR00T N1.7、Ψ₀、LingBot-VLA 2.0、MolmoAct2、Qwen-VLA、Xiaomi-Robotics-1、UnifoLM-VLA-0、InternVLA-A1.5、Hy-Embodied-0.5-VLA。
2. **控制频率与任务长度：** 世界–动作耦合的 OpenWAM、低时延 TurboVLA、使用高低层分解与世界模型的 τ₀-VLA。
3. **基础设施：** StarVLA（训练/模型适配）、vla-evaluation-harness（隔离评测）、verl-vla（人介入与 SFT/RL 后训练链路）。

以上是原文**分组**，不能以此推导所有项目有相同开放程度，也不能跨硬件/基准比较百分比。

## 结合一手资料进行的更正

- **vla-evaluation-harness：** 文章来源前段归于 MAUM.AI，但后文明确写 Ai2。实际公开仓库为 [allenai/vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness)；仓库主体以 Ai2 / Allen Institute for AI 为准。
- **Qwen-VLA：** 截至本次官方 README 核对仍以论文、博客、结果与 Demo 为主；未找到训练/推理主代码与模型权重发布入口。它是研究成果公开，不等于权重可复现。
- **TurboVLA：** 官方 README 已公开训练/评测代码及权重；在 RTX 4090 上报告约 31.2 ms 单次策略推理、0.9 GiB 推理显存，**32 Hz 不等于真机控制整环频率**。官方 2026-09-27 更新 RoboTwin 2.0 结果 88.06%，不能与早期 LIBERO 97.6% 混为同一协议。
- **Hy-VLA：** 官方写完整 UMI 预训练使用 10,000+ 小时，**开放数据集为 2,000+ 小时**；不可把全文 10k h 理解为全部数据已下载。
- **源论文与完整可运行仓库：** 来源采用不同验证层级；本次针对**综述**入库，并未重新通读 15 个项目的全部论文及代码，未开展 15 套复现实验。
- **Star 截面：** 原文统计截至 2026-11-11，本文不固化任何 Star 排名作选型证据。

## 对 wiki 的映射

- [2026 VLA 开源生态三转变对比](../../wiki/comparisons/vla-open-source-ecosystem-2026-three-shifts.md)
- [2025 VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md)

## 官方仓库补充资料

- [TurboVLA](../repos/turbovla.md)
- [Hy-Embodied-0.5-VLA](../repos/hy-embodied-05-vla.md)
- [vla-evaluation-harness](../repos/vla-evaluation-harness.md)
- [verl-vla](../repos/verl-vla.md)
