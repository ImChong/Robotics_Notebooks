# Astribot SmoothRL 项目页（归档）

- **标题：** SmoothRL: Online Reinforcement Learning During Asynchronous Execution
- **类型：** site / project-page（**不可达**）
- **URL：** <https://www.astribot.com/research/SmoothRL>（论文所列；截至 2026-09-02 返回 **HTTP 404**）
- **arXiv：** <https://arxiv.org/abs/2608.29768>
- **HTML：** <https://arxiv.org/html/2608.29768>
- **入库日期：** 2026-09-02
- **配套论文：** [SmoothRL（arXiv:2608.29768）](../papers/smoothrl_arxiv_2608_29768.md)

## 一句话摘要

星尘智能（Astribot）提出的 **异步推理环内在线 RL 微调 VLA** 工作；论文 Project Page 指向 `astribot.com/research/SmoothRL`，但入库日复核 **404**。方法细节以 arXiv HTML/PDF 为准。

## 公开信息要点（arXiv + 入库日复核）

- **机构：** Astribot Team（research@astribot.com）。
- **平台：** Astribot S1 移动双臂人形（25 DoF）；冻结 **π₀.₅** 为 base policy；三相机 224² + 本体；30 Hz 控制、5 Hz 推理、latency budget **n=6** 帧。
- **三任务：** 动态投掷入筐（39%→94%）；双臂笔帽装配（8%→83%）；开箱切胶带（30%→90%）；各 250 rollout episodes 在线微调。
- **方法：** chunk 三区划分（committed / execution / discarded）；梯度仅经 execution region 回传；训练环嵌入异步推理；残差干预优于 VR 直接遥操作（远距投掷 30% vs 80%）。
- **代码 / 数据（步骤 2.5）：** 项目页 **404**；arXiv 正文 **未列** GitHub / Hugging Face → 按 **确认未开源**（官方入口仅 arXiv）处理。

## 关联

- Wiki：[paper-smoothrl](../../wiki/entities/paper-smoothrl.md)
- 交叉：[ARLI](../../wiki/entities/paper-arli.md)、[action-chunking](../../wiki/methods/action-chunking.md)
