# GR00T 2 白皮书（2026，原始论文归档）

- **正式标题：** GR00T 2: Scaling Robot Foundation Models for Generalization, Dexterity, and Adaptation
- **作者/机构：** NVIDIA et al. / 英伟达（NVIDIA）
- **类型：** NVIDIA 技术白皮书（2026 年 11 月）
- **原文：** <https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf>
- **项目：** <https://research.nvidia.com/labs/gear/gr00t-2/>（官方项目页归档：[gr00t-2-gear.md](../sites/gr00t-2-gear.md)）
- **代码：** 2026-11-10 项目页注明 `Code Available in December`，尚无 GR00T 2 公布的实现入口
- **入库日期：** 2026-11-10
- **核对范围：** 已读取官方项目页全部可抓取文本，以及搜索索引中 PDF 的目录、引言和 §2 Overview 可见段落；这些段落涉及输入输出、视频 VAE、语言编码器、状态/动作本体头和 joint flow matching。**未能获取并完整阅读白皮书 PDF（网页读取被文件体积限制，容器下载失败），§3–§5 全文、附录 A–D 及其统计表格未逐页核对，图片/曲线未逐点检验。不得视为论文通读或全表复核。**
- **通读：** 未完成

## 已见一手文本：架构与方法（部分）

- §2 概述：0.8 秒视频历史、最多四路相机、5 fps、每帧 224×320，加机器人本体状态与自然语言指令；输出未来帧与动作块。
- §2 描述：基于视频 diffusion transformer / Cosmos-Predict2.5 系列，冻结 Wan 2.1 视频 VAE 压缩视频；冻结 Cosmos-Reason1 VLM 编码语言；本体专属 MLP 编码状态/动作并解码动作。视频 token 跨时间与视角注意，动作 token 可读未来生成的视频 token。
- §2 描述：未来视频和未来动作共用 flow-matching 训练目标，各有损失项；部署采用少量去噪步。这些架构细节是 PDF 可见段落，未核对完整训练超参、后续修订与附录。
- 官方项目页补充：X-Adapt（人→机器人、UMI→机器人、DexUMI→灵巧手），少样本适配、真机 RL 与 Agentic/Visual Steering，G1 Turbo + SONIC 全身控制。

## 论文目录（PDF 索引可见）

1. Introduction（p.3）
2. GR00T 2 Overview（p.5）
3. GR00T 2 Capabilities（p.6，含 Few-shot、X-Adapt、RL、Agentic、Loco-manipulation）
4. How We Made GR00T 2（p.14，含训练、数据、评测、采集管线、仿真、部署和基础设施）
5. Conclusion and Open Problems（p.26）
6. 附录 A–D（p.38 起，零/少样本、X-Adapt、数据、GearArena）

## 需补读与验收

- 完整 PDF 正文（含结果表、图注及作者声明的限制）
- 附录 A–D 的对照口径、训练与推理详细设置
- NVIDIA 公布 GR00T 2 代码后，固定提交并阅读实际入口，再补 **源码运行时序图**
- **本文件不摘录全文**；只记录已核实内容与通读缺口

## 对 wiki 的映射

- [GR00T 2：跨本体世界动作基础模型](../../wiki/entities/paper-gr00t-2.md)
