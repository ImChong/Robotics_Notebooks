# motubrain_arxiv_2604_27792

> 来源归档（ingest）

- **标题：** Motubrain: An Advanced World Action Model for Robot Control
- **短名：** Motubrain
- **类型：** paper
- **来源：** arXiv technical report
- **原始链接：**
  - <https://arxiv.org/abs/2604.27792>
  - <https://arxiv.org/pdf/2604.27792>
- **项目页：** <https://www.motubrain.com/zh/> · <https://www.genspi.com/zh/motubrain/>
- **代码仓（占位）：** <https://github.com/shengshu-ai/Motubrain> — [`sources/repos/motubrain.md`](../repos/motubrain.md)
- **作者：** Motubrain Team / Chendong Xiang, Fan Bao, Haitian Liu, Hengkai Tan, Hongzhe Bi, … / Jun Zhu
- **机构：** 生数科技（Shengshu Technology）；清华大学
- **版本：** arXiv:2604.27792（2026-04，2026-07 更新）
- **入库日期：** 2026-08-13
- **一句话说明：** 生数科技 Joint WAM：UniDiffuser 式统一 video–action，三流 MoT + H-bridge；RoboTwin 2.0 Clean/Random **95.8 / 96.1**；50–100 条同本体轨迹适配新人形。官方仓截至入库日仅报告 PDF，**无训练/推理代码**。

## 核心摘录（编译自官方 README / 项目页，细节以 PDF 为准）

1. **定位：** Motus（arXiv:2512.13030）验证 WAM 范式；Motubrain 做规模、多视角、统一 action、实时闭环与推理加速，面向真机「通用大脑」。
2. **建模：** 一次训练五种推理模式（VLA / WM / 视频生成 / IDM / 联合预测）；relative EEF 统一动作；独立 text 流；任意视角 token 拼接 + view-dependent RoPE；中间层 H-bridge attention。
3. **后训练 / 推理：** Teacher Forcing AR + Diffusion；DiT cache / FP8 / CUDA graph ≈ **5 Hz**（相对 Motus ~10×）；IDM / Video-to-Action 可到 **~11 Hz**；部署用 RTC + 动作平滑。异步策略对照见 [2608.01880](./wam_realtime_async_arxiv_2608_01880.md)。
4. **数字（README）：** RoboTwin 2.0 **95.8 / 96.1**；WorldArena 表内 EWMScore **63.77**（README 导语另写 64.87，以表为准）；宣称 50–100 条同本体轨迹适配；不依赖额外 VLM 规划器 / 双系统 / 外部 memory / retry 数据。
5. **开源：** 仓内无脚本；Modified MIT（超大规模商用需 UI 展示 MotuBrain）。前作 Motus 官网称已开源，**不可**等同 Motubrain 可复现。

## 对 wiki 的映射

- 升格 [Motubrain 论文实体](../../wiki/entities/paper-motubrain.md)
- 部署实证 [WAM 实时异步](../../wiki/entities/paper-wam-realtime-async.md)
- 前作索引 [Motus](../../wiki/entities/paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md)
