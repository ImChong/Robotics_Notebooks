# Qwen-VLA 原文核对 / Unifying VLA Modeling across Tasks, Environments, and Robot Embodiments

- **类型：** paper
- **论文：** https://arxiv.org/html/2605.30280
- **官方代码入口：** https://github.com/QwenLM/Qwen-VLA （截至 2026-11-14 官方 README 以技术报告、Demo 与结果为主，未见完整训练/推理工程与权重可下载）
- **团队：** Qwen Team（通义千问）
- **核对范围：** arXiv HTML §1–6 与附录（骨干/动作专家、embodiment-aware prompt、动作布局、T2A-CPT-SFT-RL 阶段、操作/导航/ALOHA/OOD 实验、对照），官方 GitHub README 全文及开放入口。
- **通读：** 已完成

## 机制、指标与边界

- **Qwen3.5-4B VLM + 1.15B DiT flow-matching** 连续动作头，同一个模型训练操作、导航、轨迹等任务；文本 embodiment prompt 指定本体/控制语义，不对每个平台新开动作头。
- 训练阶段是 T2A 动作预测预训练、CPT 多模态持续预训练、SFT 和 RL；该统一表征有助于跨平台，但实际传感器/动作空间仍须逐一对齐。
- 作者报告 Qwen-VLA-Instruct：LIBERO **97.9**；RoboTwin Easy/Hard **86.1/87.2**；R2R 成功率 **57.5**，RxR **59.6**；ALOHA 实机 OOD 平均成功率 **76.9%**。任务和指标分开统计。
- **开源局限：** 公开论文/演示≠可直接训练和下载完整 checkpoint。应在 README 未更新前避免写“完全开源可复现”。

## 对 wiki 的映射

- [Qwen-VLA 唯一详情](../../wiki/entities/qwen-vla.md)
