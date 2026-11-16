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

## 全文通读复核（2026-11-13）

- **核对范围：** arXiv:2604.27792v5 PDF（2026-07-15），§1–§5 全文（方法、加速与异步执行、仿真、世界模型、真机、局限和贡献者）；图表按文字层核对，参考文献浏览；官方 README 与资源树；未观看演示或运行模型。
- **通读：** 已完成
- §2：text stream 无输出头，H-bridge 仅中间 50% 层联合注意力。Stage 1 只更新视频；Stage 2 冻结视频、只更新动作支，但保留 video/action 两项 loss。每臂相对 EEF 为 3位置+6旋转+1夹爪=10D，仅夹爪归一化。
- Non-AR 与 AR 后训练分别训练；V2A 掩码是 video 不读 action，action 读 video/text。推理先短 joint prefix，再冻结视频并缓存 K/V，仅更新 action；不能笼统称完全不执行视频支。
- 表 2：Non-AR 基准 4.90s→噪声采样 2.90→compile 0.98→FP8 0.88→cache 0.20→V2A 0.09，累计 54.4×。不能把 5Hz 缓存阶段当作完整联合推理的固定速度。
- 表 3–4：Clean 95.80、Rand 96.08；训练 2,500 clean+25,000 randomized，5Hz 视频/10Hz 动作。表 5 EWMScore 63.77，README 导语 64.87 与其表不一致。
- §3.3：关东煮 5 次、7 动作、98.54；调酒 7 次、15 动作、97.34；插花 10 次、10 动作、83.30。数值是按重试次数折扣的步骤评分，不是完整任务二值成功率。
- `shengshu-ai/Motubrain` 固定提交 `a845f4b93f430c578398bc65b1614b79f17088cd` 只有文档/PDF/图片等，无可运行训练推理脚本；前作 Motus 代码不能充作此模型实现。
