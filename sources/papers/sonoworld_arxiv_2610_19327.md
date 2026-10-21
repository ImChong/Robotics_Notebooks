# SonoWorld: Towards Interactive Pose-Controllable Ultrasound World Models（arXiv:2610.19327）

> 来源归档（ingest）

- **标题：** SonoWorld: Towards Interactive Pose-Controllable Ultrasound World Models
- **arXiv：** <https://arxiv.org/abs/2610.19327>（cs.CV，2026-10-14 提交）
- **代码：** 无（项目页标注 Coming Soon）
- **项目页：** <https://iclr2027anonymous.github.io/SonoWorld/>
- **作者：** Zefan Yang, Zheyuan Zhang, Boris Maihe, Yikang Liu, Hang Yu, Ankush Mukherjee, Xiao Chen, Pingkun Yan, Terrence Chen, Shanhui Sun
- **单位（论文 HTML）：** United Imaging Intelligence（美国波士顿）；Rensselaer Polytechnic Institute
- **入库日期：** 2026-10-17
- **来源线索：** 具身智能小站 2026-10-17 文章（<https://mp.weixin.qq.com/s/I4XP5JSG9QNmm4hd1nk2JQ>）
- **一句话说明：** 因果、可由 6-DoF 探头位姿控制的超声世界模型，三阶段：潜动作预训练、位姿落地、因果蒸馏。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 匿名投稿版（Under Review），论文与 GitHub 链接均为 "Coming Soon"；无代码、数据、权重链接；BibTeX 作者为 Anonymous。
- **结论：** **宣称将发布但未列链接**；无可运行实现。

## 摘录 1：方法

- Wan2.1-T2V-1.3B 骨干；阶段 1 潜动作条件视频扩散（无位姿标注），阶段 2 用 EM 追踪甲状腺扫描把潜动作换为度量 6-DoF，阶段 3 DMD 蒸馏为因果生成器 + 选择性 KV 淘汰；40.6 fps（256×256，4 步，H100）。

**对 wiki 的映射：** [`paper-sonoworld`](../../wiki/entities/paper-sonoworld.md)「核心原理」。

## 摘录 2：数据与结果

- 过滤后 153,892 段 / 6,925,140 帧；EM 数据 42 名受试者（33/9）。项目页写约 6.7M 帧，与论文不一致。
- 潜动作预训练 PSNR 17.78 → 21.38；动作判别阈值 15 平均 74.76%；用户研究 5 位评分者 55 段。

**对 wiki 的映射：** 同页「实验与评测」。

## 摘录 3：局限

- 位姿数据小且仅甲状腺；接触力、形变等未观测；像素指标为间接证据；无共享基准。

**对 wiki 的映射：** 同页「局限与风险」。

## 交叉链接

- 项目页归档：[sonoworld-project.md](../sites/sonoworld-project.md)
