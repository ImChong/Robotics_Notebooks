# vision-locomotion.github.io（CoRL 2022 视觉四足 Locomotion 项目页）

- **标题：** Legged Locomotion in Challenging Terrains using Egocentric Vision — 官方项目页
- **类型：** site / project-page
- **URL：** <https://vision-locomotion.github.io/>
- **PDF：** <https://arxiv.org/pdf/2211.07638.pdf>
- **arXiv：** <https://arxiv.org/abs/2211.07638>
- **OpenReview：** <https://openreview.net/forum?id=Re3NjSwf0WF>
- **YouTube：** <https://youtu.be/5sRqythe6TE>
- **入库日期：** 2026-09-21
- **配套论文：** [Legged Locomotion in Challenging Terrains using Egocentric Vision（CoRL 2022）](../papers/vision_locomotion_corl_2022_arxiv_2211_07638.md)
- **代码：** 截至入库日，项目页 **未列 GitHub / Hugging Face / 数据集链接**（仅有 PDF、OpenReview、视频与媒体报道）

## 一句话摘要

CMU Pathak Lab × UC Berkeley Malik 组的 **CoRL 2022 Best Systems Paper**：在 **Unitree A1 级小型四足** 上，用 **单前向 Intel RealSense 深度相机 + 端到端 RNN 策略**（无高程图 / 无 VIO 融合），实机穿越 **楼梯、路缘、踏石、沟隙** 与户外非结构化地形；项目页含 **CoRL 2022 / CVPR 2022 live demo**、失败案例（高路缘跌落、后足记忆误差）与多家媒体报道。

## 公开信息要点（截至入库日）

- **会议标签：** CoRL 2022 **Best Systems Paper Award**；PMLR v205 pp.403–415。
- **作者：** Ananye Agarwal*、Ashish Kumar*、Jitendra Malik†、Deepak Pathak†（CMU + UC Berkeley）。
- **平台：** Unitree **A1**（站立约 40 cm；髋高约 28 cm）；**单 D435 前向深度**；UPboard + Jetson NX；策略 **50 Hz**，PD **400 Hz**。
- **方法主张（页面 / 论文一致）：**
  - 反对「高程图 + 落脚点规划」分模块栈（噪声、专用硬件、生物不合理）；
  - **两阶段训练**：Phase 1 用廉价 **scandots** PPO；Phase 2 **DAgger** 蒸馏到 **深度 + 本体 GRU**（Monolithic 或 **RMA 解耦** 架构）；
  - **无步态先验**，小机体上 **自发 hip abduction** 攀 stair/curb（相对身高可达 ~89% 台阶高度）。
- **实机演示（项目页视频）：**
  - **Bar stools 踏石 / 大 gap**：后足须 **记忆** 前相机已见障碍位置；16 次配置中仅 1 次后足踩空失败；
  - **楼梯**：最高 **24 cm** 高、最窄 **30 cm** 宽；**路缘 26 cm**；多种光照与 OOD 楼梯可 **涌现 climbing** 恢复；
  - **非结构化户外**：泥阶、树根、河滩岩石；红外深度 **弱光可用**；
  - **鲁棒性**：5 kg 重物投掷、塑料布泼水滑地；
  - **失败**：过高路缘 **dip 超出前视** 会跌落；无顶视相机。
- **Baselines 叙事（论文）：** 盲走 upstairs **0%**；带噪高程图 student 在踏石上几乎不动；本文在 sim 总 mean time to fall 比 blind/noisy **高约 60–90%**。
- **后续谱系：** 同作者线 → [RMA](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)、[Extreme Parkour](../../wiki/entities/extreme-parkour.md)（项目页将其作为楼梯失败对照）；Ashish Kumar → [CMS / antonilo vision_locomotion](https://antonilo.github.io/vision_locomotion/)（ICRA 2023，**不同论文**，有独立代码栈）。

## 为何值得保留

- **端到端 ego-depth loco 的早期系统论文 + Best Systems 奖**，定义「前视深度 + 短时记忆 → 后足落脚」范式，影响后续跑酷 / 人形感知 locomotion 叙事。
- **项目页失败视频** 比 PDF 更直观：前视-only 的 **记忆边界** 与 **Extreme Parkour 对照** 是选型关键证据。
- **开源状态明确为未发布**：避免与 ICRA 2023 CMS 的 [`vision_locomotion`](https://github.com/antonilo/vision_locomotion) 仓库混淆。

## 关联资料

- 论文归档：[`sources/papers/vision_locomotion_corl_2022_arxiv_2211_07638.md`](../papers/vision_locomotion_corl_2022_arxiv_2211_07638.md)
- Wiki 实体：[`wiki/entities/paper-vision-locomotion-egocentric.md`](../../wiki/entities/paper-vision-locomotion-egocentric.md)
