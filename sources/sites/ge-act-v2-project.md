# GE-Act 2.0 项目页（ge-act-v2.github.io）

- **类型**：项目静态站点
- **收录日期**：2026-09-09
- **站点**：<https://ge-act-v2.github.io/>
- **论文**：<https://arxiv.org/abs/2609.05588>
- **代码：** 页内标注 **Code · Coming soon**（无 URL）

## 一句话

**Genie Envisioner Act 2.0（GE-Act 2.0）** 展示 **CoAE + SVP + IDM** 模块化 WAM 预训练、**KASO** 联合对齐、**300→30,000 h** 操作数据缩放曲线，以及 **G1-OP / G2-90D** 零样本 OOD 真机评测与 **104 ms** 部署延迟。

## 开源核查（2026-09-09）

| 项 | 结论 |
|----|------|
| **代码** | **待发布** — 页眉 **Code · Coming soon**，无 GitHub 链接 |
| **权重** | **未列出** |
| **数据** | 披露预训练/共训小时数与评测协议；无公开下载 |

## 站点摘录要点

- **机构**：AgiBot Research（智元机器人）。
- **模块**：CoAE（24 tokens/帧）· SVP（单步 MeanFlow）· IDM（逆动力学）；KASO 连接联合训练。
- **缩放**：共训 300 / 1.2k / 5k / 30k h；G1-OP **44.1%**、G2-90D **31.1%** 均值（30k 档）。
- **指令**：杂乱场景选物、对抗动作偏置、组合新长程任务。
- **部署**：单张 RTX 5090，**104 ms** 完整管线，**52×30 Hz** 动作块。

## 对 wiki 的映射

- 主沉淀：[GE-Act 2.0](../../wiki/entities/paper-ge-act-2.md)
- 原始论文档：[ge_act_2_arxiv_2609_05588.md](../papers/ge_act_2_arxiv_2609_05588.md)
- 平台姊妹：[Genie Envisioner](../../wiki/entities/paper-sa-2508-05635-genie-envisioner-a-unified-world-foundation-plat.md)、[GE-Sim 2.0](../../wiki/entities/ge-sim-2.md)
